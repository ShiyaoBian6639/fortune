"""
Adversarial Validation for the deeptime cache.

Trains a binary classifier to distinguish train-period samples from test-period
samples. A feature with high importance in this classifier is leaking the
time period — i.e., its distribution has shifted between the two splits. These
are the features driving the val→test IC gap.

Interpretation guide:
  AUC ≈ 0.50 → no distribution shift detected (train/test are indistinguishable)
  AUC  0.60-0.75 → mild shift; inspect top features
  AUC  0.75-0.90 → significant shift; top features should be removed or
                   cross-sectionally ranked by day (config: DT_CS_NORMALIZE_TECH_FEATURES)
  AUC > 0.90     → severe shift; feature engineering likely has a leak or the
                   feature encodes absolute levels that differ between regimes

Usage:
    python -m deeptime.adversarial_validation [--top 30] [--n_samples 200000]

Output:
    plots/deeptime_results/adversarial_validation.csv
    (printed top-N shift features with importance + per-feature AUC)
"""

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deeptime.config import DEFAULT_CONFIG, DT_OBSERVED_PAST_COLUMNS


def _load_last_timestep(
    cache_dir: str,
    split: str,
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the LAST timestep features for each sequence in `split`, subsampled
    to at most `n_samples`. Also returns anchor dates so we can slice by date.
    """
    with open(os.path.join(cache_dir, 'metadata.json')) as f:
        meta = json.load(f)

    si = meta['splits'][split]
    n_total  = si['n_samples']
    seq_len  = meta['seq_length']
    n_past   = meta['n_past']

    obs_path   = os.path.join(cache_dir, f'{split}_obs.npy')
    dates_path = os.path.join(cache_dir, f'{split}_dates.npy')
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Missing {obs_path}")

    obs    = np.memmap(obs_path,   dtype='float32', mode='r', shape=(n_total, seq_len, n_past))
    dates  = np.memmap(dates_path, dtype='int32',   mode='r', shape=(n_total,))

    # Subsample
    rng = np.random.default_rng(seed)
    n_keep = min(n_samples, n_total)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    idx.sort()   # sorted → memmap reads closer to sequential

    # Take last timestep (most recent features, closest to anchor date)
    X = obs[idx, -1, :].copy()
    D = dates[idx].copy()
    return X, D


def _per_feature_auc(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Univariate Mann-Whitney AUC for each feature — how well can this feature
    *alone* distinguish train from test? Useful for ranking shift severity
    without needing a trained classifier.
    """
    from scipy.stats import rankdata
    n_tr, n_te = len(X_train), len(X_test)
    aucs = np.zeros(X_train.shape[1])
    for j in range(X_train.shape[1]):
        combined = np.concatenate([X_train[:, j], X_test[:, j]])
        # Replace nan/inf before ranking to avoid warnings
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        ranks = rankdata(combined)
        rank_sum_test = ranks[n_tr:].sum()
        u = rank_sum_test - n_te * (n_te + 1) / 2.0
        auc = u / (n_tr * n_te)
        # Fold so AUC is always ≥ 0.5 (direction-agnostic)
        aucs[j] = max(auc, 1 - auc)
    return aucs


def run_adversarial_validation(
    cache_dir: str,
    n_samples: int = 200_000,
    top_n:     int = 30,
    use_xgb:   bool = True,
    seed:      int = 42,
) -> pd.DataFrame:
    """
    Train a classifier to tell train-cache samples from test-cache samples.
    High classifier AUC means the two splits have drifted distributionally.
    """
    print(f"\n{'='*60}")
    print("Adversarial Validation: train-cache vs test-cache")
    print(f"{'='*60}\n")

    print(f"Loading last-timestep features from cache ({cache_dir})...")
    X_tr, dates_tr = _load_last_timestep(cache_dir, 'train', n_samples, seed)
    X_te, dates_te = _load_last_timestep(cache_dir, 'test',  n_samples, seed + 1)
    print(f"  train: {len(X_tr):,} samples (date range {dates_tr.min()}-{dates_tr.max()})")
    print(f"  test:  {len(X_te):,} samples (date range {dates_te.min()}-{dates_te.max()})")

    # Replace nan/inf (shouldn't be there post-normalization but guard anyway)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Per-feature univariate AUC (fast, no training needed) ────────────────
    print("\nComputing univariate per-feature AUC...")
    univ_aucs = _per_feature_auc(X_tr, X_te)

    # ── Train a classifier (multivariate view) ────────────────────────────────
    X = np.concatenate([X_tr, X_te], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))]).astype(np.int32)

    rng = np.random.default_rng(seed + 2)
    shuf = rng.permutation(len(X))
    X, y = X[shuf], y[shuf]

    # 80/20 internal split just to measure generalization of the AV classifier
    split = int(0.8 * len(X))
    X_fit, X_eval = X[:split], X[split:]
    y_fit, y_eval = y[:split], y[split:]

    clf_importances = None
    clf_auc         = None

    if use_xgb:
        try:
            import xgboost as xgb
            print("\nTraining XGBoost binary classifier...")
            clf = xgb.XGBClassifier(
                n_estimators     = 300,
                max_depth        = 6,
                learning_rate    = 0.1,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                tree_method      = 'hist',
                random_state     = seed,
                n_jobs           = -1,
                eval_metric      = 'auc',
            )
            clf.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=False)
            from sklearn.metrics import roc_auc_score
            clf_auc = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1])
            clf_importances = clf.feature_importances_
        except ImportError:
            print("  xgboost not installed — falling back to sklearn GradientBoosting")
            use_xgb = False

    if not use_xgb or clf_importances is None:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        print("\nTraining sklearn GradientBoosting classifier...")
        clf = GradientBoostingClassifier(
            n_estimators  = 100,
            max_depth     = 4,
            learning_rate = 0.1,
            subsample     = 0.8,
            random_state  = seed,
        )
        clf.fit(X_fit, y_fit)
        clf_auc = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1])
        clf_importances = clf.feature_importances_

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Classifier AUC: {clf_auc:.4f}")
    if   clf_auc < 0.60: verdict = "No meaningful distribution shift"
    elif clf_auc < 0.75: verdict = "Mild shift — inspect top features"
    elif clf_auc < 0.90: verdict = "Significant shift — remove or CS-rank top features"
    else:                verdict = "Severe shift — likely leak or absolute-level features"
    print(f"Verdict: {verdict}")
    print(f"{'='*60}\n")

    # Align column names with actual feature count
    n_feats    = X.shape[1]
    feat_names = list(DT_OBSERVED_PAST_COLUMNS[:n_feats])
    if len(feat_names) < n_feats:
        feat_names += [f'feature_{i}' for i in range(len(feat_names), n_feats)]

    df = pd.DataFrame({
        'feature':          feat_names,
        'classifier_imp':   clf_importances,
        'univariate_auc':   univ_aucs,
    }).sort_values('classifier_imp', ascending=False).reset_index(drop=True)

    # Flag whether each top feature is already in the CS-normalization list
    try:
        from deeptime.config import DT_CS_NORMALIZE_TECH_FEATURES as _CSNORM
        cs_set = set(_CSNORM)
    except Exception:
        cs_set = set()
    df['already_cs_normalized'] = df['feature'].isin(cs_set)

    print(f"Top {top_n} features driving the train/test split:")
    print(f"{'rank':>4} {'feature':<30} {'clf_imp':>10} {'uni_AUC':>10} {'CS-norm':>8}")
    print("-" * 70)
    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        cs_flag = '✓' if row['already_cs_normalized'] else ' '
        print(f"{rank:>4} {row['feature'][:30]:<30} "
              f"{row['classifier_imp']:>10.4f} "
              f"{row['univariate_auc']:>10.3f} "
              f"{cs_flag:>8}")

    # ── Save CSV ────────────────────────────────────────────────────────────
    out_dir = os.path.join(_ROOT, 'plots', 'deeptime_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'adversarial_validation.csv')
    df.to_csv(out_path, index=False)
    print(f"\nFull ranking saved to {out_path}")

    # ── Suggest action ──────────────────────────────────────────────────────
    not_normalized = df[~df['already_cs_normalized']].head(top_n)
    if len(not_normalized) > 0 and clf_auc > 0.65:
        print(f"\nSuggested action: add these top shift features to "
              f"DT_CS_NORMALIZE_TECH_FEATURES in deeptime/config.py:")
        print("    " + ", ".join(f"'{f}'" for f in not_normalized['feature'].head(10)))
        print("  Then rebuild the cache: rm -rf stock_data/deeptime_cache/")

    return df


def main():
    p = argparse.ArgumentParser(description='Adversarial validation on deeptime cache')
    p.add_argument('--cache_dir', default=DEFAULT_CONFIG['cache_dir'])
    p.add_argument('--n_samples', type=int, default=200_000,
                   help='Samples per split (default 200k; use smaller for speed)')
    p.add_argument('--top',       type=int, default=30,
                   help='Top N shift features to print (default 30)')
    p.add_argument('--no_xgb',    action='store_true',
                   help='Force sklearn GradientBoosting instead of XGBoost')
    p.add_argument('--seed',      type=int, default=42)
    args = p.parse_args()

    run_adversarial_validation(
        cache_dir = args.cache_dir,
        n_samples = args.n_samples,
        top_n     = args.top,
        use_xgb   = not args.no_xgb,
        seed      = args.seed,
    )


if __name__ == '__main__':
    main()
