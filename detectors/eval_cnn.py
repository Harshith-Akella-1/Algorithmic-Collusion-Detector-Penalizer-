"""
eval_cnn.py -- run test-set evaluation using the saved cnn_best.pt checkpoint.
Reproduces the final test metrics without re-training.
Saves confusion matrix as cnn_confusion_matrix.png.
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, roc_auc_score, average_precision_score)

# Fix imports: train_cnn.py lives in detectors/
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from train_cnn import (
    CNN1D, SeqDataset, INDEX_FILE, SEQ_FILE, CHECKPOINT,
    CLASSES, CLS_TO_IDX, BATCH_SIZE, SEED, DEVICE
)

CONFUSION_PNG = Path(__file__).resolve().parent / 'cnn_confusion_matrix.png'


def save_confusion_matrix_png(cm, classes, macro_f1, path):
    """Save a confusion matrix heatmap as a PNG file."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print('  [WARN] matplotlib/seaborn not installed -- skipping PNG')
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize for percentage annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Heatmap with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='white')

    # Add percentage annotations in smaller text
    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = cm_norm[i, j]
            ax.text(j + 0.5, i + 0.75, f'({pct:.0f}%)',
                    ha='center', va='center', fontsize=7,
                    color='gray' if pct < 50 else 'lightgray')

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(f'CNN Confusion Matrix -- Macro F1 = {macro_f1:.3f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  >> Confusion matrix saved to {path}')


def main():
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
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    all_y, all_p, all_probs = [], [], []
    with torch.no_grad():
        for x, y_batch in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_y.append(y_batch.numpy())
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

    macro_f1 = f1_score(y, p, average='macro')

    # Save confusion matrix PNG
    save_confusion_matrix_png(cm, CLASSES, macro_f1, CONFUSION_PNG)

    # Binary view
    none_idx = CLS_TO_IDX['none']
    bin_y = (y != none_idx).astype(int)
    bin_p = (p != none_idx).astype(int)
    collusion_prob = 1.0 - probs[:, none_idx]
    print('\n--- Binary view: is_collusion ---')
    print(classification_report(bin_y, bin_p,
                                target_names=['none', 'collusion'], digits=3))
    print(f'ROC-AUC : {roc_auc_score(bin_y, collusion_prob):.4f}')
    print(f'PR-AUC  : {average_precision_score(bin_y, collusion_prob):.4f}')

    print(f'\nMacro F1: {macro_f1:.4f}')
    print('Per-class F1: ' + ' '.join(
        f'{c}={v:.3f}' for c, v in
        zip(CLASSES, f1_score(y, p, average=None,
                              labels=list(range(len(CLASSES)))))))

    return y, p, probs, cm, macro_f1


if __name__ == '__main__':
    main()
