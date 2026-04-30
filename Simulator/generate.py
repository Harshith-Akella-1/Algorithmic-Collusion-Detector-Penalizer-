"""
generate.py -- generate the full collusion dataset.

Output layout:
    dataset/
        labels.parquet            # one row per episode
        orders/ep_NNNNN.parquet   # orders for each episode
        trades/ep_NNNNN.parquet   # trades for each episode

Run: python generate.py
"""
import time
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from simulator import run_episode, COLLUSION_TYPES

OUT_DIR = Path('dataset')
N_PER_CLASS = 1000          # 5 classes -> 5000 total
SEED_BASE = 100_000


def generate(out_dir=OUT_DIR, n_per_class=N_PER_CLASS, seed_base=SEED_BASE):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / 'orders').mkdir(parents=True)
    (out_dir / 'trades').mkdir(parents=True)

    n_total = n_per_class * len(COLLUSION_TYPES)
    print(f'Generating {n_total} episodes '
          f'({n_per_class} per class x {len(COLLUSION_TYPES)} classes)')
    print(f'Output -> {out_dir}/\n')

    labels = []
    ep_idx = 0
    t_start = time.time()

    for class_idx, ctype in enumerate(COLLUSION_TYPES):
        class_t0 = time.time()
        for i in range(n_per_class):
            seed = seed_base + class_idx * 1_000_000 + i
            orders, trades, label = run_episode(seed=seed, collusion_type=ctype)
            ep_id = f'ep_{ep_idx:05d}'
            orders.to_parquet(out_dir / 'orders' / f'{ep_id}.parquet',
                              compression='snappy', index=False)
            trades.to_parquet(out_dir / 'trades' / f'{ep_id}.parquet',
                              compression='snappy', index=False)
            labels.append({
                'episode_id': ep_id,
                'collusion_type': label['collusion_type'],
                'colluder_A': label['pair'][0] if label['pair'] else None,
                'colluder_B': label['pair'][1] if label['pair'] else None,
                't_start': label['t_start'],
                't_end': label['t_end'],
                'seed': seed,
                'n_orders': len(orders),
                'n_trades': len(trades),
            })
            ep_idx += 1
            # Progress every 100 episodes
            if (i + 1) % 100 == 0:
                elapsed = time.time() - class_t0
                rate = (i + 1) / elapsed
                print(f'  [{ctype:6s}] {i+1:4d}/{n_per_class}  '
                      f'({rate:.1f} ep/s)')
        print(f'  [{ctype:6s}] done in {time.time()-class_t0:.1f}s\n')

    labels_df = pd.DataFrame(labels)
    labels_df.to_parquet(out_dir / 'labels.parquet',
                         compression='snappy', index=False)

    total_elapsed = time.time() - t_start
    print(f'=== Done in {total_elapsed/60:.1f} min ===')
    print(f'  labels.parquet  : {len(labels_df)} rows')
    print(f'  orders/         : {len(list((out_dir/"orders").iterdir()))} files')
    print(f'  trades/         : {len(list((out_dir/"trades").iterdir()))} files')

    # Sanity report
    print('\nClass balance:')
    print(labels_df['collusion_type'].value_counts().to_string())
    print('\nMean orders/trades per class:')
    print(labels_df.groupby('collusion_type')[['n_orders', 'n_trades']]
          .mean().round(0).astype(int).to_string())


if __name__ == '__main__':
    generate()
