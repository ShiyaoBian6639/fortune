"""
Turn the XGBRegressor's point estimate into calibrated probabilities.

The regressor emits one number (the predicted next-day pct_chg). Trading
usually wants a *probability* — e.g. "what's the chance this stock goes up
tomorrow?" or "probability of > +3% gain?".

We convert the point estimate into a probability by combining it with the
empirical distribution of out-of-sample residuals observed on the val set.
Under a Gaussian or Student-t error assumption:

    P(target > threshold | pred) = 1 - F((threshold - pred) / sigma)

where F is the Gaussian / Student-t CDF and sigma is the residual scale.

Rationale (Efron & Hastie 2016, "Computer Age Statistical Inference" §12):
point predictions from tree ensembles are well-calibrated for the *center*
of the distribution but not for tail events — matching them against the
empirical residual distribution gives a reasonable calibrated probability
without needing a second model. For higher-quality tail probabilities,
either train a dedicated binary classifier (`objective=binary:logistic`)
or quantile-regression models (`objective=reg:quantileerror`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from scipy import stats

from .config import MODEL_DIR


# ─── Residual-distribution fit from val/test OOF predictions ────────────────

@dataclass
class ResidualModel:
    """A Student-t fit of the OOF residuals (target - pred) across val.

    Student-t handles return fat tails much better than plain Gaussian. With
    df → ∞ it collapses back to Gaussian. Typical fit on A-share returns is
    df ≈ 4-6.
    """
    mu:    float
    sigma: float
    df:    float
    n_obs: int

    def p_greater_than(self, pred: np.ndarray, threshold: float) -> np.ndarray:
        """P(target > threshold | pred) as a vector of the same shape as `pred`."""
        z = (threshold - pred - self.mu) / self.sigma
        return 1.0 - stats.t.cdf(z, df=self.df)

    def p_less_than(self, pred: np.ndarray, threshold: float) -> np.ndarray:
        z = (threshold - pred - self.mu) / self.sigma
        return stats.t.cdf(z, df=self.df)

    def prediction_interval(self, pred: np.ndarray, coverage: float = 0.80):
        """Return (low, high) — symmetric equal-tailed PI around each pred."""
        alpha = (1 - coverage) / 2
        q_lo  = stats.t.ppf(alpha,     df=self.df)
        q_hi  = stats.t.ppf(1 - alpha, df=self.df)
        return pred + self.mu + self.sigma * q_lo, pred + self.mu + self.sigma * q_hi

    def summary(self) -> str:
        return (f"ResidualModel(mu={self.mu:+.4f}, sigma={self.sigma:.4f}, "
                f"df={self.df:.2f}, n={self.n_obs:,})")


def fit_residual_model(
    oof_preds: pd.DataFrame,
    pred_col:   str = 'pred',
    target_col: str = 'target',
    min_obs:    int = 1000,
) -> ResidualModel:
    """Fit a Student-t to `target - pred` on an OOF prediction DataFrame.

    If `scipy.stats.t.fit` fails to converge (can happen with degenerate
    residual distributions), fall back to Gaussian (df=inf) with method-of-moments.
    """
    if len(oof_preds) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} OOF predictions to fit a residual "
            f"distribution; got {len(oof_preds)}"
        )
    resid = (oof_preds[target_col].values - oof_preds[pred_col].values).astype('float64')
    # Drop non-finite
    resid = resid[np.isfinite(resid)]
    try:
        # scipy stats.t.fit returns (df, loc, scale)
        df, mu, sigma = stats.t.fit(resid, floc=None, fscale=None)
        # Guard against ridiculously small df (= Cauchy-like, unreliable)
        if df < 2.5:
            df = 2.5
    except Exception:
        df = float('inf')
        mu, sigma = float(resid.mean()), float(resid.std())
    return ResidualModel(mu=float(mu), sigma=float(max(sigma, 1e-6)),
                         df=float(df), n_obs=int(len(resid)))


# ─── Public API used by predict.py ──────────────────────────────────────────

def load_val_residual_model(cfg: dict) -> ResidualModel:
    """Reload the val OOF preds saved by train.py and fit a residual model.

    We prefer val OOF because (a) it was the split used for early stopping
    (so it best matches the model's calibration) and (b) it's typically much
    larger than test when the user ran walk-forward CV.
    """
    model_dir = cfg.get('model_dir', MODEL_DIR)
    val_path  = os.path.join(model_dir, 'xgb_preds', 'val.csv')
    test_path = os.path.join(model_dir, 'xgb_preds', 'test.csv')
    path = val_path if os.path.exists(val_path) else test_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No OOF predictions found at {val_path} or {test_path}. "
            f"Run `python -m xgbmodel.main --mode train` first."
        )
    oof = pd.read_csv(path)
    return fit_residual_model(oof)


def attach_probabilities(
    df:       pd.DataFrame,
    pred_col: str,
    resid:    ResidualModel,
    thresholds: Iterable[float] = (-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0),
    include_pi: bool = True,
) -> pd.DataFrame:
    """Add P(next-day return > threshold) columns for each listed threshold.

    Column naming:
        prob_gt_{N}pct  = P(pct_chg_next > N%)    for positive N
        prob_lt_{N}pct  = P(pct_chg_next < N%)    for negative N
        prob_up         = prob_gt_0pct (alias)
        prob_down       = prob_lt_0pct (alias)

    Optionally adds an 80% prediction interval (pi_lo_80, pi_hi_80).
    """
    pred = df[pred_col].values.astype('float64')
    new  = {}
    for th in thresholds:
        if th >= 0:
            col = f'prob_gt_{_fmt(th)}pct'
            new[col] = resid.p_greater_than(pred, th).astype('float32')
        else:
            col = f'prob_lt_{_fmt(-th)}pct'
            new[col] = resid.p_less_than(pred, th).astype('float32')
    # Convenient aliases
    new['prob_up']   = resid.p_greater_than(pred, 0.0).astype('float32')
    new['prob_down'] = resid.p_less_than    (pred, 0.0).astype('float32')

    if include_pi:
        lo, hi = resid.prediction_interval(pred, coverage=0.80)
        new['pi_lo_80'] = lo.astype('float32')
        new['pi_hi_80'] = hi.astype('float32')

    return pd.concat([df.reset_index(drop=True),
                      pd.DataFrame(new, index=df.reset_index(drop=True).index)],
                     axis=1, copy=False)


def _fmt(x: float) -> str:
    """Format 1.0 → '1', 2.5 → '2_5'."""
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace('.', '_')
