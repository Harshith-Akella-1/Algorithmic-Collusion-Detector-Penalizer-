"""
eval_cnn.py -- run test-set evaluation using the saved cnn_best.pt checkpoint.
Reproduces the final test metrics without re-training.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, average_precision_score

from train_cnn import (
    CNN1D, SeqDataset, INDEX_FILE, SEQ_FILE, CHECKPOINT,
    CLASSES, CLS_TO_IDX, BATCH_SIZE, SEED, DEVICE
)
import torch.nn.functional as F

torch.manual_seed(SEED)
np.random.seed(SEED)

# Replicate split exactly
idx_df = pd.read_parquet(INDEX_FILE)
seq = np.load(SEQ_FILE, mmap_mode='r')
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
trainval_idx, holdout_idx = next(gss1.split(idx_df, groups=idx_df['episode_id']))
df_hold = idx_df.iloc[holdout_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1)
val_idx_local, test_idx_local = next(gss2.split(df_hold, groups=df_hold['episode_id']))
df_te = df_hold.iloc[test_idx_local]

ds = SeqDataset(df_te['idx'].to_numpy(),
                df_te['window_type'].map(CLS_TO_IDX).to_numpy(), seq)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = CNN1D().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

all_y, all_p, all_probs = [], [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_y.append(y.numpy())
        all_p.append(logits.argmax(1).cpu().numpy())
        all_probs.append(probs)
y = np.concatenate(all_y)
p = np.concatenate(all_p)
probs = np.concatenate(all_probs)

print(f'Test windows: {len(y)}')
print(f'\n{classification_report(y, p, target_names=CLASSES, digits=3)}')
print('Confusion matrix (rows=true, cols=pred):')
cm = confusion_matrix(y, p)
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

# Binary view
none_idx = CLS_TO_IDX['none']
bin_y = (y != none_idx).astype(int)
bin_p = (p != none_idx).astype(int)
collusion_prob = 1.0 - probs[:, none_idx]
print('\n--- Binary view: is_collusion ---')
print(classification_report(bin_y, bin_p, target_names=['none', 'collusion'], digits=3))
print(f'ROC-AUC : {roc_auc_score(bin_y, collusion_prob):.4f}')
print(f'PR-AUC  : {average_precision_score(bin_y, collusion_prob):.4f}')

print(f'\nMacro F1: {f1_score(y, p, average="macro"):.4f}')
print(f'Per-class F1: ' + ' '.join(f'{c}={v:.3f}' for c, v in
    zip(CLASSES, f1_score(y, p, average=None, labels=list(range(len(CLASSES)))))))
