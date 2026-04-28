"""Drop-in alternative regressors for the BTC ensemble pipeline.

NOTE: this file targets the **BTC pipeline** (btc/advanced_pipeline.py and
btc/multi_model_pipeline.py). It is intentionally separate from the rest of
the xgbmodel package, which is the A-share stock system.

All wrappers expose the same interface:
    .fit(X_full, y, end_idx, sample_weight=None, val_size=60)
    .predict(X_full, t) -> float
    .feature_importances_ -> np.ndarray | None

The wrappers handle their own model-specific input shaping. Tabular models
(LGB, MLP) consume a single feature row X_full[t]. Sequence models (DCNN,
Transformer) consume a window X_full[t-T+1:t+1]. y and target semantics are
identical across all models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# LightGBM — single-row input, drop-in for XGB
# ============================================================================

class LightGBMReg:
    def __init__(self, **params):
        import lightgbm as lgb
        self._lgb = lgb
        self.params = dict(
            objective="regression",
            n_estimators=500,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.03,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.params.update(params)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 30)
        self.model = None

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        X = X_full[:end_idx]
        if n - val < 50:
            self.model = self._lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y[:end_idx], sample_weight=sample_weight)
            return self
        X_tr, y_tr = X[:-val], y[:end_idx - val]
        X_val, y_val = X[-val:], y[end_idx - val:end_idx]
        sw_tr = sample_weight[:-val] if sample_weight is not None else None
        self.model = self._lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_tr, y_tr, sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[self._lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict(self, X_full, t):
        return float(self.model.predict(X_full[t:t + 1])[0])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.model is not None else None


# ============================================================================
# Torch utilities
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loaders(X, y, sw, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    sw_t = (torch.tensor(sw, dtype=torch.float32) if sw is not None
            else torch.ones(len(X), dtype=torch.float32))
    ds = torch.utils.data.TensorDataset(X_t, y_t, sw_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, pin_memory=False)


def _train_torch(model, loader_tr, loader_val, *, lr, weight_decay,
                 epochs, patience, min_epochs=10):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(epochs):
        model.train()
        for xb, yb, sw in loader_tr:
            xb, yb, sw = xb.to(DEVICE), yb.to(DEVICE), sw.to(DEVICE)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = (sw * (pred - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            tot = 0.0
            n = 0
            for xb, yb, sw in loader_val:
                xb, yb, sw = xb.to(DEVICE), yb.to(DEVICE), sw.to(DEVICE)
                pred = model(xb).squeeze(-1)
                tot += float((sw * (pred - yb) ** 2).sum().item())
                n += int(sw.sum().item())
            val_loss = tot / max(n, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if ep >= min_epochs and no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


# ============================================================================
# MLP — single-row input
# ============================================================================

class _MLPNet(nn.Module):
    def __init__(self, n_features, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class MLPReg:
    def __init__(self, hidden=128, dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 epochs=120, patience=15, batch_size=128, seed=42):
        self.hidden = hidden; self.dropout = dropout
        self.lr = lr; self.weight_decay = weight_decay
        self.epochs = epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed
        self.model = None
        self.feat_mu = self.feat_sd = None

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        _seed_all(self.seed)
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        X = X_full[:end_idx]
        valid = ~np.isnan(y[:end_idx])
        # Standardize on train portion (excluding val)
        cut = end_idx - val
        self.feat_mu = np.nanmean(X[:cut][valid[:cut]], axis=0)
        self.feat_sd = np.nanstd(X[:cut][valid[:cut]], axis=0) + 1e-6

        Xn = (X - self.feat_mu) / self.feat_sd
        Xn = np.nan_to_num(Xn, nan=0.0)
        sw = (sample_weight if sample_weight is not None
              else np.ones(end_idx, dtype=np.float32))

        X_tr, y_tr, sw_tr = Xn[:cut], y[:cut], sw[:cut]
        X_val, y_val, sw_val = Xn[cut:end_idx], y[cut:end_idx], sw[cut:end_idx]
        v_tr = ~np.isnan(y_tr); v_val = ~np.isnan(y_val)
        loader_tr = _make_loaders(X_tr[v_tr], y_tr[v_tr], sw_tr[v_tr],
                                  batch_size=self.batch_size, shuffle=True)
        loader_val = _make_loaders(X_val[v_val], y_val[v_val], sw_val[v_val],
                                   batch_size=self.batch_size, shuffle=False)

        self.model = _MLPNet(X.shape[1], hidden=self.hidden, dropout=self.dropout).to(DEVICE)
        _train_torch(self.model, loader_tr, loader_val,
                     lr=self.lr, weight_decay=self.weight_decay,
                     epochs=self.epochs, patience=self.patience)
        return self

    def predict(self, X_full, t):
        x = X_full[t:t + 1]
        x = (x - self.feat_mu) / self.feat_sd
        x = np.nan_to_num(x, nan=0.0)
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x).squeeze().item())

    @property
    def feature_importances_(self):
        return None


# ============================================================================
# Dilated 1D CNN — windowed input
# ============================================================================

class _DCNNNet(nn.Module):
    """Causal dilated 1D CNN over (window, n_features)."""

    def __init__(self, n_features, channels=(64, 64, 64), kernel=3, dropout=0.15):
        super().__init__()
        layers = []
        in_c = n_features
        dilation = 1
        for c in channels:
            pad = (kernel - 1) * dilation  # causal left-pad applied in forward
            layers.append(nn.Conv1d(in_c, c, kernel_size=kernel, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_c = c
            dilation *= 2
        self.convs = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(in_c, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.kernel = kernel

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        dilation = 1
        i = 0
        while i < len(self.convs):
            conv = self.convs[i]
            pad = (self.kernel - 1) * dilation
            x = F.pad(x, (pad, 0))
            x = conv(x)
            x = self.convs[i + 1](x)  # GELU
            x = self.convs[i + 2](x)  # Dropout
            dilation *= 2
            i += 3
        return self.head(x)


class DCNNReg:
    def __init__(self, window=20, channels=(64, 64, 64), kernel=3, dropout=0.15,
                 lr=1e-3, weight_decay=1e-4, epochs=80, patience=12,
                 batch_size=64, seed=42):
        self.window = window; self.channels = channels; self.kernel = kernel
        self.dropout = dropout; self.lr = lr; self.weight_decay = weight_decay
        self.epochs = epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed
        self.model = None
        self.feat_mu = self.feat_sd = None

    def _build_windows(self, X, indices):
        return np.stack([X[i - self.window + 1: i + 1] for i in indices], axis=0)

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        _seed_all(self.seed)
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        cut = end_idx - val
        valid = ~np.isnan(y[:end_idx])

        # Standardize features using train portion only
        train_mask = valid.copy()
        train_mask[cut:] = False
        self.feat_mu = np.nanmean(X_full[:end_idx][train_mask], axis=0)
        self.feat_sd = np.nanstd(X_full[:end_idx][train_mask], axis=0) + 1e-6

        Xn = (X_full - self.feat_mu) / self.feat_sd
        Xn = np.nan_to_num(Xn, nan=0.0)
        sw_full = (sample_weight if sample_weight is not None
                   else np.ones(end_idx, dtype=np.float32))

        # Eligible indices (need full window of past data)
        train_idx = np.array([i for i in range(self.window - 1, cut) if valid[i]])
        val_idx = np.array([i for i in range(max(cut, self.window - 1), end_idx) if valid[i]])
        if len(train_idx) < 50 or len(val_idx) < 5:
            self.model = None
            return self

        X_tr_w = self._build_windows(Xn, train_idx)
        X_val_w = self._build_windows(Xn, val_idx)
        y_tr = y[train_idx]; y_val = y[val_idx]
        sw_tr = sw_full[train_idx]; sw_val = sw_full[val_idx]

        loader_tr = _make_loaders(X_tr_w, y_tr, sw_tr,
                                  batch_size=self.batch_size, shuffle=True)
        loader_val = _make_loaders(X_val_w, y_val, sw_val,
                                   batch_size=self.batch_size, shuffle=False)
        self.model = _DCNNNet(X_full.shape[1], channels=self.channels,
                              kernel=self.kernel, dropout=self.dropout).to(DEVICE)
        _train_torch(self.model, loader_tr, loader_val,
                     lr=self.lr, weight_decay=self.weight_decay,
                     epochs=self.epochs, patience=self.patience)
        return self

    def predict(self, X_full, t):
        if self.model is None or t < self.window - 1:
            return float("nan")
        x = X_full[t - self.window + 1: t + 1]
        x = (x - self.feat_mu) / self.feat_sd
        x = np.nan_to_num(x, nan=0.0)
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x).squeeze().item())

    @property
    def feature_importances_(self):
        return None


# ============================================================================
# Transformer encoder — windowed input
# ============================================================================

class _TransformerNet(nn.Module):
    def __init__(self, n_features, window, d_model=64, nhead=4, n_layers=2,
                 ffn=128, dropout=0.15):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        h = self.proj(x) + self.pos
        h = self.enc(h)
        return self.head(h[:, -1, :])  # last position pooling


class TransformerReg:
    def __init__(self, window=20, d_model=64, nhead=4, n_layers=2, ffn=128,
                 dropout=0.15, lr=5e-4, weight_decay=1e-4, epochs=80,
                 patience=12, batch_size=64, seed=42):
        self.window = window; self.d_model = d_model; self.nhead = nhead
        self.n_layers = n_layers; self.ffn = ffn; self.dropout = dropout
        self.lr = lr; self.weight_decay = weight_decay
        self.epochs = epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed
        self.model = None
        self.feat_mu = self.feat_sd = None

    def _build_windows(self, X, indices):
        return np.stack([X[i - self.window + 1: i + 1] for i in indices], axis=0)

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        _seed_all(self.seed)
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        cut = end_idx - val
        valid = ~np.isnan(y[:end_idx])
        train_mask = valid.copy(); train_mask[cut:] = False
        self.feat_mu = np.nanmean(X_full[:end_idx][train_mask], axis=0)
        self.feat_sd = np.nanstd(X_full[:end_idx][train_mask], axis=0) + 1e-6
        Xn = (X_full - self.feat_mu) / self.feat_sd
        Xn = np.nan_to_num(Xn, nan=0.0)
        sw_full = (sample_weight if sample_weight is not None
                   else np.ones(end_idx, dtype=np.float32))

        train_idx = np.array([i for i in range(self.window - 1, cut) if valid[i]])
        val_idx = np.array([i for i in range(max(cut, self.window - 1), end_idx) if valid[i]])
        if len(train_idx) < 50 or len(val_idx) < 5:
            self.model = None
            return self
        X_tr_w = self._build_windows(Xn, train_idx)
        X_val_w = self._build_windows(Xn, val_idx)
        y_tr = y[train_idx]; y_val = y[val_idx]
        sw_tr = sw_full[train_idx]; sw_val = sw_full[val_idx]
        loader_tr = _make_loaders(X_tr_w, y_tr, sw_tr,
                                  batch_size=self.batch_size, shuffle=True)
        loader_val = _make_loaders(X_val_w, y_val, sw_val,
                                   batch_size=self.batch_size, shuffle=False)
        self.model = _TransformerNet(
            X_full.shape[1], self.window, d_model=self.d_model, nhead=self.nhead,
            n_layers=self.n_layers, ffn=self.ffn, dropout=self.dropout,
        ).to(DEVICE)
        _train_torch(self.model, loader_tr, loader_val,
                     lr=self.lr, weight_decay=self.weight_decay,
                     epochs=self.epochs, patience=self.patience)
        return self

    def predict(self, X_full, t):
        if self.model is None or t < self.window - 1:
            return float("nan")
        x = X_full[t - self.window + 1: t + 1]
        x = (x - self.feat_mu) / self.feat_sd
        x = np.nan_to_num(x, nan=0.0)
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x).squeeze().item())

    @property
    def feature_importances_(self):
        return None


# ============================================================================
# XGB wrapper to share the same interface
# ============================================================================

class XGBReg:
    def __init__(self, **params):
        import xgboost as xgb
        self._xgb = xgb
        self.params = dict(
            objective="reg:squarederror",
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=8,
            tree_method="hist", random_state=42, n_jobs=-1,
            early_stopping_rounds=30,
        )
        self.params.update(params)
        self.model = None

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        X = X_full[:end_idx]
        if n - val < 50:
            p = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
            self.model = self._xgb.XGBRegressor(**p)
            self.model.fit(X, y[:end_idx], sample_weight=sample_weight)
            return self
        X_tr, y_tr = X[:-val], y[:end_idx - val]
        X_val, y_val = X[-val:], y[end_idx - val:end_idx]
        sw_tr = sample_weight[:-val] if sample_weight is not None else None
        self.model = self._xgb.XGBRegressor(**self.params)
        self.model.fit(X_tr, y_tr, sample_weight=sw_tr,
                       eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X_full, t):
        return float(self.model.predict(X_full[t:t + 1])[0])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.model is not None else None


# ============================================================================
# CatBoost — single-row input, drop-in for XGB/LGB
# ============================================================================

class CatBoostReg:
    def __init__(self, **params):
        import catboost
        self._cb = catboost
        self.params = dict(
            loss_function="RMSE",
            iterations=500,
            depth=6,
            learning_rate=0.03,
            subsample=0.85,
            rsm=0.7,           # column subsample (CatBoost-specific)
            l2_leaf_reg=3.0,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        self.params.update(params)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 30)
        self.model = None

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        X = X_full[:end_idx]
        if n - val < 50:
            self.model = self._cb.CatBoostRegressor(**self.params)
            self.model.fit(X, y[:end_idx], sample_weight=sample_weight)
            return self
        X_tr, y_tr = X[:-val], y[:end_idx - val]
        X_val, y_val = X[-val:], y[end_idx - val:end_idx]
        sw_tr = sample_weight[:-val] if sample_weight is not None else None
        sw_val = sample_weight[-val:] if sample_weight is not None else None
        self.model = self._cb.CatBoostRegressor(**self.params)
        self.model.fit(
            X_tr, y_tr, sample_weight=sw_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        return self

    def predict(self, X_full, t):
        return float(self.model.predict(X_full[t:t + 1])[0])

    @property
    def feature_importances_(self):
        if self.model is None:
            return None
        return np.asarray(self.model.get_feature_importance())


# ============================================================================
# TabNet — single-row input, attentive tabular network
# ============================================================================

class TabNetReg:
    def __init__(self, n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-3,
                 lr=2e-3, epochs=80, patience=12, batch_size=256, seed=42):
        self.n_d = n_d; self.n_a = n_a; self.n_steps = n_steps
        self.gamma = gamma; self.lambda_sparse = lambda_sparse
        self.lr = lr; self.epochs = epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed
        self.model = None
        self.feat_mu = self.feat_sd = None

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        from pytorch_tabnet.tab_model import TabNetRegressor
        _seed_all(self.seed)
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        cut = end_idx - val
        valid = ~np.isnan(y[:end_idx])

        train_mask = valid.copy(); train_mask[cut:] = False
        self.feat_mu = np.nanmean(X_full[:end_idx][train_mask], axis=0)
        self.feat_sd = np.nanstd(X_full[:end_idx][train_mask], axis=0) + 1e-6
        Xn = (X_full - self.feat_mu) / self.feat_sd
        Xn = np.nan_to_num(Xn, nan=0.0).astype(np.float32)

        y32 = y.astype(np.float32)
        v_tr = valid[:cut]; v_val = valid[cut:end_idx]
        if v_tr.sum() < 100 or v_val.sum() < 5:
            self.model = None
            return self

        X_tr = Xn[:cut][v_tr]; y_tr = y32[:cut][v_tr].reshape(-1, 1)
        X_val = Xn[cut:end_idx][v_val]; y_val = y32[cut:end_idx][v_val].reshape(-1, 1)

        sw = (sample_weight[:cut][v_tr].astype(np.float32)
              if sample_weight is not None else None)

        self.model = TabNetRegressor(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, lambda_sparse=self.lambda_sparse,
            optimizer_params=dict(lr=self.lr),
            seed=self.seed, verbose=0,
            device_name="cuda" if DEVICE.type == "cuda" else "cpu",
        )
        self.model.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)],
            max_epochs=self.epochs, patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=min(self.batch_size, 64),
            weights=sw if sw is not None else 0,
            drop_last=True,
        )
        return self

    def predict(self, X_full, t):
        if self.model is None:
            return float("nan")
        x = X_full[t:t + 1]
        x = (x - self.feat_mu) / self.feat_sd
        x = np.nan_to_num(x, nan=0.0).astype(np.float32)
        return float(self.model.predict(x).reshape(-1)[0])

    @property
    def feature_importances_(self):
        if self.model is None:
            return None
        return self.model.feature_importances_


# ============================================================================
# Quantile LightGBM — predicts q25/q50/q75; mean = q50, spread = q75 - q25
# ============================================================================

class QuantileLGBReg:
    """Predicts three quantiles of the normalized target.

    .predict(X, t) returns the q50 (median) prediction — drop-in scalar.
    .predict_quantiles(X, t) returns (q25, q50, q75).
    .spread(X, t) returns q75 - q25 (uncertainty proxy for position sizing).
    """

    def __init__(self, quantiles=(0.25, 0.5, 0.75), **params):
        import lightgbm as lgb
        self._lgb = lgb
        self.quantiles = tuple(quantiles)
        base = dict(
            objective="quantile",
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.03,
            subsample=0.85, subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        base.update(params)
        self.early_stopping_rounds = base.pop("early_stopping_rounds", 30)
        self.params = base
        self.models = {}

    def fit(self, X_full, y, end_idx, sample_weight=None, val_size=60):
        n = end_idx
        val = min(val_size, max(n // 5, 10))
        X = X_full[:end_idx]
        for q in self.quantiles:
            params = dict(self.params, alpha=q)
            if n - val < 50:
                m = self._lgb.LGBMRegressor(**params)
                m.fit(X, y[:end_idx], sample_weight=sample_weight)
            else:
                X_tr, y_tr = X[:-val], y[:end_idx - val]
                X_val, y_val = X[-val:], y[end_idx - val:end_idx]
                sw_tr = sample_weight[:-val] if sample_weight is not None else None
                m = self._lgb.LGBMRegressor(**params)
                m.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[self._lgb.early_stopping(
                          self.early_stopping_rounds, verbose=False)])
            self.models[q] = m
        return self

    def predict(self, X_full, t):
        return self.predict_quantiles(X_full, t)[1]

    def predict_quantiles(self, X_full, t):
        x = X_full[t:t + 1]
        return tuple(float(self.models[q].predict(x)[0]) for q in self.quantiles)

    def spread(self, X_full, t):
        q25, _, q75 = self.predict_quantiles(X_full, t)
        return q75 - q25

    @property
    def feature_importances_(self):
        if 0.5 not in self.models:
            return None
        return self.models[0.5].feature_importances_


REGISTRY = {
    "xgb": XGBReg,
    "lgb": LightGBMReg,
    "mlp": MLPReg,
    "dcnn": DCNNReg,
    "transformer": TransformerReg,
    "catboost": CatBoostReg,
    "tabnet": TabNetReg,
    "quantile_lgb": QuantileLGBReg,
}


def make(name, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(REGISTRY)}.")
    return REGISTRY[name](**kwargs)
