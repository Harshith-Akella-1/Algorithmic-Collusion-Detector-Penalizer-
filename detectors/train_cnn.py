"""
train_cnn.py -- 1D-CNN detector over raw event sequences.

Key design choices to prevent overfitting:
    - Episode-level train/val/test split (windows from one episode never cross splits)
    - Dropout (0.2 conv, 0.3 dense)
    - BatchNorm between conv layers (regularizing effect)
    - Weight decay (L2) on optimizer
    - Class-weighted cross-entropy loss
    - Early stopping on val loss (patience=8)
    - Best-model checkpoint saved by val macro-F1 (not loss -- F1 is the
      metric we actually care about, especially for paint)

Per-event features (6 channels):
    [normalized_ts, side, qty_normalized, is_CA, is_CB, type_onehot]
    - normalized_ts in [0, 1] across the window
    - side: -1 sell, +1 buy
    - qty_normalized: log1p(qty) / 6 (so range ~[0, 1])
    - is_CA, is_CB: binary
    - type_onehot: 3 channels (limit, market, cancel)

Sequence length: 1024 events per window (covers p99=940 with headroom).
Padding: zero-padded at the end. The CNN's GlobalMaxPool ignores padded zeros
naturally because they're below typical feature activations.

Run: python train_cnn.py
"""
import time
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DATA_DIR = Path('dataset')
FEATURES_FILE = DATA_DIR / 'features.parquet'
CHECKPOINT = Path('cnn_best.pt')

SEQ_LEN = 200
N_FEATURES = 6
SEQ_FILE = DATA_DIR / 'sequences.npy'
INDEX_FILE = DATA_DIR / 'seq_index.parquet'

CLASSES = ['none', 'wash', 'paint', 'spoof', 'mirror']
N_CLASSES = len(CLASSES)
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
PATIENCE = 8
SEED = 13

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)


# -------------------------------------------------------------------------
# Dataset (memmap-backed for fast random access)
# -------------------------------------------------------------------------
class SeqDataset(Dataset):
    """Loads pre-extracted sequences from a memmap. O(1) per item."""
    def __init__(self, indices, labels, seq_memmap):
        self.indices = indices.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.seq = seq_memmap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.seq[self.indices[i]]   # (N_FEATURES, SEQ_LEN)
        return torch.from_numpy(x.copy()), int(self.labels[i])


# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
class CNN1D(nn.Module):
    """1D-CNN with global max pool. Output: 5-class logits.

    Sized for CPU training: 32 -> 64 channels keeps params low (~25k) while
    still capturing enough patterns for 4 collusion types.

    Architecture choices:
      - 2 conv blocks (vs 3) to keep CPU compute low
      - BatchNorm + ReLU after each conv (regularizing effect)
      - Dropout 0.2 between conv blocks (sequence-level regularization)
      - GlobalMaxPool: translation-invariant, doesn't care WHERE in window
        the pattern appears
      - Dense(64) -> Dropout(0.3) -> Dense(N_CLASSES)
    """
    def __init__(self, in_ch=N_FEATURES, n_classes=N_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.drop_conv = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 64)
        self.drop_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 200 -> 100
        x = self.drop_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))              # 100
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)


# -------------------------------------------------------------------------
# Train / eval loop
# -------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            n += y.size(0)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    return total_loss / n, y, p


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f'Device: {DEVICE}')

    idx_df = pd.read_parquet(INDEX_FILE)
    seq_memmap = np.load(SEQ_FILE, mmap_mode='r')
    print(f'Loaded index: {len(idx_df):,} windows')
    print(f'Sequence memmap: {seq_memmap.shape}, dtype={seq_memmap.dtype}')
    print(f'Class balance: {idx_df["window_type"].value_counts().to_dict()}\n')

    # Episode-level split: train/val/test = 70/15/15
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    trainval_idx, holdout_idx = next(gss1.split(idx_df, groups=idx_df['episode_id']))
    df_tv = idx_df.iloc[trainval_idx]
    df_hold = idx_df.iloc[holdout_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
    val_idx_local, test_idx_local = next(gss2.split(df_hold, groups=df_hold['episode_id']))
    df_val = df_hold.iloc[val_idx_local]
    df_te = df_hold.iloc[test_idx_local]
    print(f'Train: {len(df_tv):,} windows / {df_tv.episode_id.nunique()} eps')
    print(f'Val  : {len(df_val):,} windows / {df_val.episode_id.nunique()} eps')
    print(f'Test : {len(df_te):,} windows / {df_te.episode_id.nunique()} eps\n')

    def make_ds(d):
        return SeqDataset(d['idx'].to_numpy(),
                          d['window_type'].map(CLS_TO_IDX).to_numpy(),
                          seq_memmap)

    train_ds = make_ds(df_tv)
    val_ds = make_ds(df_val)
    test_ds = make_ds(df_te)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Class weights (inverse frequency, normalized to sum = N_CLASSES)
    counts = Counter(df_tv['window_type'])
    weights = np.array([1.0 / counts[c] for c in CLASSES], dtype=np.float32)
    weights = weights / weights.sum() * N_CLASSES
    print(f'Class weights: {dict(zip(CLASSES, weights.round(3)))}\n')
    class_w = torch.from_numpy(weights).to(DEVICE)

    model = CNN1D().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}\n')

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_epoch = -1
    patience_left = PATIENCE
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_w)
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        train_loss /= n_seen

        val_loss, val_y, val_p = evaluate(model, val_loader)
        val_f1 = f1_score(val_y, val_p, average='macro')
        per_class_f1 = f1_score(val_y, val_p, average=None,
                                labels=list(range(N_CLASSES)))

        elapsed = time.time() - t0
        f1_str = ' '.join(f'{c[:3]}={f:.2f}' for c, f in zip(CLASSES, per_class_f1))
        print(f'epoch {epoch:3d}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'val_macroF1={val_f1:.4f}  [{f1_str}]  ({elapsed:.1f}s)')
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_macro_f1': val_f1,
                        'per_class_f1': dict(zip(CLASSES, per_class_f1.tolist()))})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT)
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f'  Early stopping (no val improvement for {PATIENCE} epochs)')
                break

    print(f'\nBest val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}')
    print(f'Loading checkpoint from epoch {best_epoch}\n')
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    print('=' * 60)
    print('TEST SET METRICS')
    print('=' * 60)
    _, te_y, te_p = evaluate(model, test_loader)
    print(classification_report(te_y, te_p, target_names=CLASSES, digits=3))
    print('Confusion matrix (rows=true, cols=pred):')
    cm = confusion_matrix(te_y, te_p)
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    bin_y = (te_y != CLS_TO_IDX['none']).astype(int)
    bin_p = (te_p != CLS_TO_IDX['none']).astype(int)
    print('\n--- Binary view: is_collusion ---')
    print(classification_report(bin_y, bin_p, target_names=['none', 'collusion'], digits=3))

    Path('cnn_history.json').write_text(json.dumps(history, indent=2))
    print('\nHistory written to cnn_history.json')


if __name__ == '__main__':
    main()
