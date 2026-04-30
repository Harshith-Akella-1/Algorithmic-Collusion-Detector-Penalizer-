"""
prepare_sequences.py -- one-shot pre-extraction of event sequences.

Reads all (episode, window) pairs from features.parquet, extracts the
ordered event sequence for each window, and writes to a single memmap
file plus an index. Training then loads at native numpy speed instead
of re-reading parquet 19x per episode.

Output:
    dataset/sequences.npy       (N_windows, N_FEATURES, SEQ_LEN) float32
    dataset/seq_index.parquet   (episode_id, window_start, window_type, idx)

Run: python prepare_sequences.py
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('dataset')
SEQ_LEN = 200          # CA+CB events per window: max=112, p95=88
N_FEATURES = 6         # ts, side, qty, is_CA (vs is_CB), is_limit, is_market
                       # is_cancel implied by !is_limit & !is_market

OUT_NPY = DATA_DIR / 'sequences.npy'
OUT_IDX = DATA_DIR / 'seq_index.parquet'


def encode_window(orders_win, t0, t1):
    """Encode CA+CB events only. Other traders are noise the CNN doesn't need."""
    cab = orders_win[orders_win['trader'].isin(['CA', 'CB'])]
    n = min(len(cab), SEQ_LEN)
    x = np.zeros((N_FEATURES, SEQ_LEN), dtype=np.float32)
    if n == 0:
        return x
    sub = cab.iloc[:n]
    ts_norm = ((sub['ts'].to_numpy() - t0) / (t1 - t0)).astype(np.float32)
    side = np.where(sub['side'].to_numpy() == 'buy', 1.0, -1.0).astype(np.float32)
    qty_norm = (np.log1p(sub['qty'].to_numpy().astype(np.float32)) / 6.0)
    is_CA = (sub['trader'].to_numpy() == 'CA').astype(np.float32)
    otype = sub['type'].to_numpy()
    is_limit = (otype == 'limit').astype(np.float32)
    is_market = (otype == 'market').astype(np.float32)
    x[0, :n] = ts_norm
    x[1, :n] = side
    x[2, :n] = qty_norm
    x[3, :n] = is_CA   # 1 if CA, 0 if CB
    x[4, :n] = is_limit
    x[5, :n] = is_market
    return x


def main():
    feats = pd.read_parquet(DATA_DIR / 'features.parquet')[
        ['episode_id', 'window_start', 'window_end', 'window_type']
    ]
    feats = feats.reset_index(drop=True).copy()
    feats['idx'] = np.arange(len(feats), dtype=np.int64)
    n = len(feats)
    print(f'Pre-extracting {n} window sequences')
    print(f'Allocating: {n} x {N_FEATURES} x {SEQ_LEN} float32 = '
          f'{n * N_FEATURES * SEQ_LEN * 4 / 1e9:.2f} GB')

    seq = np.lib.format.open_memmap(OUT_NPY, mode='w+', dtype=np.float32,
                                    shape=(n, N_FEATURES, SEQ_LEN))

    t0 = time.time()
    cur_ep = None
    orders_cache = None
    written = 0
    for _, row in feats.iterrows():
        ep_id = row['episode_id']
        if ep_id != cur_ep:
            orders_cache = pd.read_parquet(
                DATA_DIR / 'orders' / f'{ep_id}.parquet'
            )
            cur_ep = ep_id
        ws, we = row['window_start'], row['window_end']
        win = orders_cache[(orders_cache['ts'] >= ws) &
                           (orders_cache['ts'] < we)]
        seq[row['idx']] = encode_window(win, ws, we)
        written += 1
        if written % 5000 == 0:
            rate = written / (time.time() - t0)
            print(f'  {written:6d}/{n}  ({rate:.0f} windows/s)')

    seq.flush()
    feats.to_parquet(OUT_IDX, index=False)
    print(f'\nWrote sequences -> {OUT_NPY}')
    print(f'Wrote index    -> {OUT_IDX}')
    print(f'Total time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
