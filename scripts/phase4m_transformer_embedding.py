"""Phase 4M.4: Transformer embedding from Alpha360 → XGB two-stage model.

Stage 1: Train Transformer on Alpha360 (60-day × 6-OHLCV sequences)
         → output 64-dim embedding per stock-day
Stage 2: Concat embedding to Alpha158 174-dim features → XGB on 238 dims

Key: uses DataLoader for batch loading to avoid OOM.
Only loads training window data per split (not full 5.9GB cache).

Usage:
    python scripts/phase4m_transformer_embedding.py
"""
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "phase4m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA360_PATH = DATA_DIR / "feature_cache_alpha360.parquet"
ALPHA174_PATH = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"

N_SPLITS = 4  # fewer splits — memory constrained
TEST_DAYS = 20
VALID_DAYS = 40
TRAIN_DAYS = 150  # shorter for memory (~7 months)

EMBEDDING_DIM = 64
N_DAYS = 60
N_FIELDS = 6  # CLOSE, OPEN, HIGH, LOW, VWAP, VOLUME
LABEL_COL = "__label_5d"

EXCLUDE_PREFIXES = ("__", "_", "hsi_", "hstech_", "nasdaq_")
EXCLUDE_EXACT = {"holder_num"}

XGB_PARAMS = {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
    "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
    "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SequenceDataset(Dataset):
    """Lazy dataset that reshapes flat 360-dim to (60, 6) sequences."""

    def __init__(self, X_flat: np.ndarray, y: np.ndarray):
        """
        Args:
            X_flat: (N, 360) flat features — CLOSE59..0, OPEN59..0, ..., VOLUME59..0
            y: (N,) labels
        """
        self.X = X_flat  # keep as numpy to save memory
        self.y = y
        self.n = len(X_flat)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_flat = self.X[idx]  # (360,)
        # Reshape to (60, 6): each field has 60 values (59=oldest, 0=newest)
        seq = np.zeros((N_DAYS, N_FIELDS), dtype=np.float32)
        for i in range(N_FIELDS):
            seq[:, i] = x_flat[i * N_DAYS:(i + 1) * N_DAYS]
        return torch.from_numpy(seq), torch.tensor(self.y[idx], dtype=torch.float32)


class SeqTransformerEncoder(nn.Module):
    """Transformer encoder that outputs a fixed-size embedding."""

    def __init__(self, n_fields=6, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_fields, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, N_DAYS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc_embed = nn.Linear(d_model, EMBEDDING_DIM)
        self.fc_pred = nn.Linear(EMBEDDING_DIM, 1)  # prediction head for training

    def forward(self, x):
        """x: (batch, 60, 6) → embedding: (batch, 64)"""
        x = self.proj(x) + self.pos_enc
        x = self.encoder(x)
        x = x[:, -1, :]  # last timestep
        embed = self.fc_embed(x)
        return embed

    def predict(self, x):
        """Full forward with prediction head."""
        embed = self.forward(x)
        return self.fc_pred(embed).squeeze(-1)


def train_transformer(model, train_loader, valid_loader, n_epochs=15, lr=1e-3,
                      early_stop=5):
    """Train transformer with early stopping."""
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model.predict(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model.predict(xb)
                val_loss += criterion(pred, yb).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"    Epoch {epoch+1}: train={total_loss/n_batches:.6f} val={val_loss:.6f}")

        if patience >= early_stop:
            logger.info(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def extract_embeddings(model, data_loader):
    """Extract embeddings from trained transformer."""
    device = get_device()
    model = model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            embed = model(xb)  # (batch, 64)
            embeddings.append(embed.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def evaluate_split(pred, label, index):
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 200:
        return None
    pred_s = pd.Series(pred[mask], index=index[mask])
    label_s = pd.Series(label[mask], index=index[mask])
    ric_list, spreads = [], []
    for dt, g in pred_s.groupby(level=0):
        gl = label_s.reindex(g.index).dropna()
        common = g.index.intersection(gl.index)
        if len(common) < 40:
            continue
        p, l = g.loc[common].values, gl.loc[common].values
        ric = stats.spearmanr(p, l).statistic
        if np.isfinite(ric):
            ric_list.append(ric)
        tmp = pd.DataFrame({"p": p, "l": l}).sort_values("p", ascending=False)
        spreads.append(tmp.head(20)["l"].mean() - tmp.tail(20)["l"].mean())
    if not ric_list:
        return None
    return {
        "rank_ic": float(np.mean(ric_list)),
        "spread_top20": float(np.mean(spreads)) if spreads else 0,
    }


def main():
    import xgboost as xgb

    t_start = time.time()
    device = get_device()
    logger.info(f"Device: {device}")

    # Step 1: Load 174 cache but only last ~1 year to save memory
    logger.info("Loading Alpha174 cache (recent only)...")
    cache_174_full = pd.read_parquet(ALPHA174_PATH)
    all_trade_dates = sorted(cache_174_full.index.get_level_values(0).unique())
    # Only keep what we need: N_SPLITS * TEST_DAYS + VALID + TRAIN + buffer
    needed_days = N_SPLITS * TEST_DAYS + VALID_DAYS + TRAIN_DAYS + 20
    cutoff_idx = max(0, len(all_trade_dates) - needed_days)
    cutoff_date = all_trade_dates[cutoff_idx]
    dates_full = cache_174_full.index.get_level_values(0)
    cache_174 = cache_174_full.loc[dates_full >= cutoff_date].copy()
    del cache_174_full; gc.collect()

    feat_174 = [c for c in cache_174.columns
                if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in EXCLUDE_EXACT]
    trade_dates = sorted(cache_174.index.get_level_values(0).unique())
    dates_level_174 = cache_174.index.get_level_values(0)
    logger.info(f"  174 cache: {cache_174.shape}, {len(feat_174)} features, from {cutoff_date}")

    # Step 2: Load 360 cache with same date filter
    logger.info("Loading Alpha360 cache (same date range)...")
    cache_360_full = pd.read_parquet(ALPHA360_PATH)
    dates_360_full = cache_360_full.index.get_level_values(0)
    cache_360 = cache_360_full.loc[dates_360_full >= cutoff_date].copy()
    del cache_360_full; gc.collect()
    feat_360 = [c for c in cache_360.columns if not c.startswith("__")]
    logger.info(f"  360 cache: {cache_360.shape}, {len(feat_360)} features")

    MODELS = [
        {"name": "xgb_174_baseline", "type": "xgb_only"},
        {"name": "xgb_174+embed64", "type": "two_stage"},
    ]

    all_results = {m["name"]: [] for m in MODELS}

    for split_idx in range(N_SPLITS):
        test_end_idx = len(trade_dates) - 1 - split_idx * TEST_DAYS
        test_start_idx = test_end_idx - TEST_DAYS
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - VALID_DAYS
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        if train_start_idx >= train_end_idx:
            break

        test_start = trade_dates[test_start_idx]
        test_end = trade_dates[test_end_idx]
        train_start = trade_dates[train_start_idx]

        logger.info(f"\nSplit {split_idx+1}/{N_SPLITS}: test {str(test_start)[:10]}~{str(test_end)[:10]}")

        # Date masks
        train_mask = (dates_level_174 >= train_start) & (dates_level_174 <= trade_dates[train_end_idx])
        valid_mask = (dates_level_174 >= trade_dates[valid_start_idx]) & (dates_level_174 <= trade_dates[valid_end_idx])
        test_mask = (dates_level_174 >= test_start) & (dates_level_174 <= test_end)

        # --- XGB baseline (174 features) ---
        t1 = time.time()
        X_tr_174 = cache_174.loc[train_mask, feat_174].values.astype(np.float32)
        y_tr = cache_174.loc[train_mask, LABEL_COL].values.astype(np.float32)
        X_va_174 = cache_174.loc[valid_mask, feat_174].values.astype(np.float32)
        y_va = cache_174.loc[valid_mask, LABEL_COL].values.astype(np.float32)
        X_te_174 = cache_174.loc[test_mask, feat_174].values.astype(np.float32)
        y_te = cache_174.loc[test_mask, LABEL_COL].values.astype(np.float32)
        test_idx = cache_174.index[test_mask]

        m_tr = np.isfinite(y_tr); X_tr_174, y_tr = X_tr_174[m_tr], y_tr[m_tr]
        m_va = np.isfinite(y_va); X_va_174, y_va = X_va_174[m_va], y_va[m_va]
        m_te = np.isfinite(y_te); X_te_174_c, y_te_c = X_te_174[m_te], y_te[m_te]
        test_idx_c = test_idx[m_te]

        dt = xgb.DMatrix(X_tr_174, label=y_tr)
        dv = xgb.DMatrix(X_va_174, label=y_va)
        model_xgb = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                               evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
        pred_baseline = model_xgb.predict(xgb.DMatrix(X_te_174_c))
        metrics_baseline = evaluate_split(pred_baseline, y_te_c, test_idx_c)
        if metrics_baseline:
            metrics_baseline["split"] = split_idx + 1
            all_results["xgb_174_baseline"].append(metrics_baseline)
            logger.info(f"  xgb_174_baseline:  RIC={metrics_baseline['rank_ic']:+.4f} ({time.time()-t1:.1f}s)")

        # --- Two-stage: Transformer embedding + XGB ---
        t2 = time.time()

        # Get Alpha360 data for same periods (already loaded in memory)
        dates_360_level = cache_360.index.get_level_values(0)

        tr_360_mask = (dates_360_level >= train_start) & (dates_360_level <= trade_dates[train_end_idx])
        va_360_mask = (dates_360_level >= trade_dates[valid_start_idx]) & (dates_360_level <= trade_dates[valid_end_idx])
        te_360_mask = (dates_360_level >= test_start) & (dates_360_level <= test_end)

        X_tr_360 = cache_360.loc[tr_360_mask, feat_360].values.astype(np.float32)
        y_tr_360 = cache_360.loc[tr_360_mask, LABEL_COL].values.astype(np.float32)
        X_va_360 = cache_360.loc[va_360_mask, feat_360].values.astype(np.float32)
        y_va_360 = cache_360.loc[va_360_mask, LABEL_COL].values.astype(np.float32)
        X_te_360 = cache_360.loc[te_360_mask, feat_360].values.astype(np.float32)
        y_te_360 = cache_360.loc[te_360_mask, LABEL_COL].values.astype(np.float32)

        # Filter NaN
        m_tr_360 = np.isfinite(y_tr_360)
        X_tr_360, y_tr_360 = X_tr_360[m_tr_360], y_tr_360[m_tr_360]
        m_va_360 = np.isfinite(y_va_360)
        X_va_360, y_va_360 = X_va_360[m_va_360], y_va_360[m_va_360]
        m_te_360 = np.isfinite(y_te_360)
        X_te_360_c, y_te_360_c = X_te_360[m_te_360], y_te_360[m_te_360]

        # NaN → 0
        X_tr_360 = np.nan_to_num(X_tr_360, nan=0.0)
        X_va_360 = np.nan_to_num(X_va_360, nan=0.0)
        X_te_360_c = np.nan_to_num(X_te_360_c, nan=0.0)

        logger.info(f"  Transformer: train={len(X_tr_360)}, valid={len(X_va_360)}, test={len(X_te_360_c)}")

        # Create DataLoaders
        train_ds = SequenceDataset(X_tr_360, y_tr_360)
        valid_ds = SequenceDataset(X_va_360, y_va_360)
        test_ds = SequenceDataset(X_te_360_c, y_te_360_c)

        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=4096, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

        # Train Transformer
        transformer = SeqTransformerEncoder(N_FIELDS, d_model=64, nhead=4, num_layers=2)
        logger.info(f"  Training Transformer on {device}...")
        transformer = train_transformer(transformer, train_loader, valid_loader,
                                         n_epochs=15, early_stop=5)

        # Extract embeddings for train, valid, test
        train_embed_loader = DataLoader(SequenceDataset(X_tr_360, y_tr_360),
                                         batch_size=4096, shuffle=False, num_workers=0)
        valid_embed_loader = DataLoader(SequenceDataset(X_va_360, y_va_360),
                                         batch_size=4096, shuffle=False, num_workers=0)

        embed_tr = extract_embeddings(transformer, train_embed_loader)
        embed_va = extract_embeddings(transformer, valid_embed_loader)
        embed_te = extract_embeddings(transformer, test_loader)

        logger.info(f"  Embeddings: train={embed_tr.shape}, test={embed_te.shape}")

        # Concat: Alpha174 + Transformer embedding → XGB
        # Need to align: 174 and 360 caches may have slightly different rows
        # Use min length (they should match if same date range)
        n_tr = min(len(X_tr_174), len(embed_tr))
        n_va = min(len(X_va_174), len(embed_va))
        n_te = min(len(X_te_174_c), len(embed_te))

        X_tr_concat = np.concatenate([X_tr_174[:n_tr], embed_tr[:n_tr]], axis=1)
        X_va_concat = np.concatenate([X_va_174[:n_va], embed_va[:n_va]], axis=1)
        X_te_concat = np.concatenate([X_te_174_c[:n_te], embed_te[:n_te]], axis=1)
        y_tr_concat = y_tr[:n_tr]
        y_va_concat = y_va[:n_va]
        y_te_concat = y_te_c[:n_te]
        test_idx_concat = test_idx_c[:n_te]

        # Train XGB on concat features
        dt2 = xgb.DMatrix(X_tr_concat, label=y_tr_concat)
        dv2 = xgb.DMatrix(X_va_concat, label=y_va_concat)
        model_xgb2 = xgb.train(XGB_PARAMS, dt2, num_boost_round=400,
                                evals=[(dv2, "valid")], early_stopping_rounds=50, verbose_eval=0)
        pred_two_stage = model_xgb2.predict(xgb.DMatrix(X_te_concat))

        metrics_two_stage = evaluate_split(pred_two_stage, y_te_concat, test_idx_concat)
        if metrics_two_stage:
            metrics_two_stage["split"] = split_idx + 1
            metrics_two_stage["n_features"] = X_tr_concat.shape[1]
            all_results["xgb_174+embed64"].append(metrics_two_stage)
            logger.info(f"  xgb_174+embed64:   RIC={metrics_two_stage['rank_ic']:+.4f} "
                        f"({X_tr_concat.shape[1]} feats, {time.time()-t2:.1f}s)")

        # Cleanup
        del X_tr_360, X_va_360, X_te_360_c, embed_tr, embed_va, embed_te
        del X_tr_concat, X_va_concat, X_te_concat
        gc.collect()

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"TRANSFORMER EMBEDDING + XGB: {N_SPLITS} splits")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<22} {'AvgRIC':>8} {'MedRIC':>8} {'Spread':>8} {'#Spl':>5}")
    logger.info("-" * 60)

    summary = {}
    for m in MODELS:
        name = m["name"]
        splits = all_results[name]
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s.get("spread_top20", 0) for s in splits]
        summary[name] = {
            "avg_rank_ic": round(float(np.mean(rics)), 6),
            "med_rank_ic": round(float(np.median(rics)), 6),
            "avg_spread": round(float(np.mean(spreads)), 6),
            "n_splits": len(splits),
        }
        logger.info(f"{name:<22} {np.mean(rics):>+8.4f} {np.median(rics):>+8.4f} "
                    f"{np.mean(spreads)*100:>+7.3f}% {len(splits):>5}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    output = {"evaluated_at": datetime.now().isoformat(timespec="seconds"),
              "n_splits": N_SPLITS, "summary": summary,
              "per_split": {k: v for k, v in all_results.items()}}
    out_path = OUTPUT_DIR / "transformer_embedding_xgb.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
