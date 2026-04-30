"""
train.py -- train XGBoost detector on window-level features.

Two heads trained:
    - binary  : is_collusion (0/1) -- the deployment-time signal
    - multi   : window_type (none/wash/paint/spoof/mirror) -- richer learning
                signal + diagnostic; sum of non-'none' probabilities recovers
                the binary score.

Episode-level split: all windows from a given episode go to the same fold,
so the model never sees test-episode windows during training.

Run: python train.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit('Install xgboost first: pip install xgboost')

FEATURES_FILE = Path('dataset/features.parquet')
TEST_SIZE = 0.2
SEED = 13

FEATURE_COLS = [
    'ab_trade_count', 'ab_trade_qty', 'ab_inter_trade_cv',
    'ab_gap_autocorr', 'ab_spectral_peak', 'ab_price_deviation',
    'A_cancel_ratio', 'B_cancel_ratio',
    'A_n_limits', 'A_n_cancels', 'B_n_limits', 'B_n_cancels',
    'sync_cancel_count', 'sync_cancel_ratio',
    'same_side_evap', 'A_cancel_burst',
]


def split(df, test_size=TEST_SIZE, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df['episode_id']))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_binary(df_tr, df_te):
    print('=' * 60)
    print('BINARY HEAD: is_collusion (0/1)')
    print('=' * 60)
    X_tr, y_tr = df_tr[FEATURE_COLS], df_tr['is_collusion']
    X_te, y_te = df_te[FEATURE_COLS], df_te['is_collusion']
    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=pos_w, eval_metric='aucpr',
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f'\nROC-AUC : {roc_auc_score(y_te, p):.4f}')
    print(f'PR-AUC  : {average_precision_score(y_te, p):.4f}')
    print(f'\n{classification_report(y_te, pred, digits=3)}')
    print('Top features by gain:')
    fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(fi.head(10).to_string())
    return clf


def train_multiclass(df_tr, df_te):
    print('\n' + '=' * 60)
    print('MULTI-CLASS HEAD: window_type')
    print('=' * 60)
    classes = sorted(df_tr['window_type'].unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f'Classes: {classes}')
    X_tr, y_tr = df_tr[FEATURE_COLS], df_tr['window_type'].map(cls_to_idx)
    X_te, y_te = df_te[FEATURE_COLS], df_te['window_type'].map(cls_to_idx)
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', eval_metric='mlogloss',
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print(f'\n{classification_report(y_te, pred, target_names=classes, digits=3)}')
    print('Confusion matrix (rows=true, cols=pred):')
    cm = confusion_matrix(y_te, pred)
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    return clf, classes


def main():
    if not FEATURES_FILE.exists():
        raise SystemExit(f'{FEATURES_FILE} not found. Run features.py first.')
    df = pd.read_parquet(FEATURES_FILE)
    print(f'Loaded {len(df):,} windows from {df["episode_id"].nunique()} episodes')
    print(f'Binary balance: {df["is_collusion"].value_counts().to_dict()}\n')

    df_tr, df_te = split(df)
    print(f'Train: {len(df_tr):,} windows / {df_tr["episode_id"].nunique()} eps')
    print(f'Test : {len(df_te):,} windows / {df_te["episode_id"].nunique()} eps\n')

    train_binary(df_tr, df_te)
    train_multiclass(df_tr, df_te)


if __name__ == '__main__':
    main()
