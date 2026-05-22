"""Step 1: Train Transformer on Alpha360 → save embeddings to parquet.

Only loads Alpha360 cache. Outputs embedding parquet with same index.
Run this FIRST, then run step2.

Usage:
    python scripts/phase4m_step1_train_transformer.py
"""
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
ALPHA360_PATH = DATA_DIR / "feature_cache_alpha360.parquet"
EMBED_PATH = DATA_DIR / "transformer_embeddings.parquet"
MODEL_PATH = DATA_DIR / "transformer_alpha360.pt"

EMBEDDING_DIM = 64
N_DAYS = 60
N_FIELDS = 6
LABEL_COL = "__label_5d"

# Only use last ~1 year for training (memory safe on 36GB)
MAX_TRAIN_DAYS = 250
VALID_DAYS = 40


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SequenceDataset(Dataset):
    def __init__(self, X_flat, y):
        self.X = X_flat
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        seq = np.zeros((N_DAYS, N_FIELDS), dtype=np.float32)
        for i in range(N_FIELDS):
            seq[:, i] = x[i * N_DAYS:(i + 1) * N_DAYS]
        return torch.from_numpy(seq), torch.tensor(self.y[idx], dtype=torch.float32)


class InferDataset(Dataset):
    """For inference only — no label needed."""
    def __init__(self, X_flat):
        self.X = X_flat

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        seq = np.zeros((N_DAYS, N_FIELDS), dtype=np.float32)
        for i in range(N_FIELDS):
            seq[:, i] = x[i * N_DAYS:(i + 1) * N_DAYS]
        return torch.from_numpy(seq)


class SeqTransformer(nn.Module):
    def __init__(self, n_fields=6, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_fields, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, N_DAYS, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc_embed = nn.Linear(d_model, EMBEDDING_DIM)
        self.fc_pred = nn.Linear(EMBEDDING_DIM, 1)

    def get_embedding(self, x):
        x = self.proj(x) + self.pos_enc
        x = self.encoder(x)
        return self.fc_embed(x[:, -1, :])

    def forward(self, x):
        return self.fc_pred(self.get_embedding(x)).squeeze(-1)


def main():
    t_start = time.time()
    device = get_device()
    logger.info(f"Device: {device}")

    # Load Alpha360 — only recent data
    logger.info("Loading Alpha360 cache...")
    cache = pd.read_parquet(ALPHA360_PATH)
    all_dates = cache.index.get_level_values(0)
    unique_dates = sorted(all_dates.unique())
    logger.info(f"  Full cache: {cache.shape}")

    # Split: train on older data, validate on recent, infer on ALL
    n_dates = len(unique_dates)
    train_end_idx = n_dates - VALID_DAYS - 1
    train_start_idx = max(0, train_end_idx - MAX_TRAIN_DAYS)
    valid_start_idx = train_end_idx + 1

    train_start = unique_dates[train_start_idx]
    train_end = unique_dates[train_end_idx]
    valid_start = unique_dates[valid_start_idx]
    valid_end = unique_dates[-1]

    logger.info(f"  Train: {train_start} ~ {train_end} ({train_end_idx - train_start_idx} days)")
    logger.info(f"  Valid: {valid_start} ~ {valid_end} ({VALID_DAYS} days)")

    feat_cols = [c for c in cache.columns if not c.startswith("__")]

    # Extract train/valid data
    train_mask = (all_dates >= train_start) & (all_dates <= train_end)
    valid_mask = (all_dates >= valid_start) & (all_dates <= valid_end)

    X_tr = cache.loc[train_mask, feat_cols].values.astype(np.float32)
    y_tr = cache.loc[train_mask, LABEL_COL].values.astype(np.float32)
    X_va = cache.loc[valid_mask, feat_cols].values.astype(np.float32)
    y_va = cache.loc[valid_mask, LABEL_COL].values.astype(np.float32)

    # NaN filter
    m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
    m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0)

    logger.info(f"  Train samples: {len(X_tr):,}, Valid: {len(X_va):,}")

    # Train
    train_ds = SequenceDataset(X_tr, y_tr)
    valid_ds = SequenceDataset(X_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=4096, shuffle=False, num_workers=0)

    model = SeqTransformer(N_FIELDS, d_model=64, nhead=4, num_layers=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience = 0
    best_state = None

    for epoch in range(20):
        model.train()
        total_loss, n_b = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        model.eval()
        val_loss, n_v = 0, 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
                n_v += 1
        val_loss /= max(n_v, 1)

        logger.info(f"  Epoch {epoch+1}: train={total_loss/n_b:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 5:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, str(MODEL_PATH))
    logger.info(f"  Model saved to {MODEL_PATH}")

    # Free training data
    del X_tr, y_tr, X_va, y_va, train_ds, valid_ds, train_loader, valid_loader
    gc.collect()

    # Inference: extract embeddings for ALL rows in cache (in batches)
    logger.info("Extracting embeddings for full cache...")
    model = model.to(device)
    model.eval()

    # Process in chunks to avoid OOM
    CHUNK_SIZE = 500_000
    all_embeddings = []
    n_total = len(cache)

    for chunk_start in range(0, n_total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
        X_chunk = cache.iloc[chunk_start:chunk_end][feat_cols].values.astype(np.float32)
        X_chunk = np.nan_to_num(X_chunk, nan=0.0)

        infer_ds = InferDataset(X_chunk)
        infer_loader = DataLoader(infer_ds, batch_size=4096, shuffle=False, num_workers=0)

        chunk_embeds = []
        with torch.no_grad():
            for xb in infer_loader:
                xb = xb.to(device)
                embed = model.get_embedding(xb).cpu().numpy()
                chunk_embeds.append(embed)

        all_embeddings.append(np.concatenate(chunk_embeds, axis=0))
        del X_chunk, infer_ds, infer_loader, chunk_embeds
        gc.collect()
        logger.info(f"  Chunk {chunk_start//CHUNK_SIZE + 1}: {chunk_end:,}/{n_total:,}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    # Save as parquet with same index as cache
    embed_cols = [f"embed_{i}" for i in range(EMBEDDING_DIM)]
    embed_df = pd.DataFrame(embeddings, index=cache.index, columns=embed_cols)
    embed_df.to_parquet(str(EMBED_PATH))
    logger.info(f"  Saved to {EMBED_PATH} ({os.path.getsize(EMBED_PATH) / 1e6:.0f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"\nStep 1 done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Next: python scripts/phase4m_step2_xgb_with_embedding.py")


if __name__ == "__main__":
    main()
