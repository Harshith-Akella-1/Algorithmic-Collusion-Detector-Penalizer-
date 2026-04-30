"""
viz.py -- Visualize collusion evidence: one episode per type.

Picks episodes from available data (either dataset/ directory or root-level
parquet/json files), runs CNN inference, and generates a multi-panel figure
showing order flow + detector confidence.

Saves to collusion_evidence.png.

Run: python viz.py
"""
import json
import sys
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Fix imports
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, 'detectors'))
sys.path.insert(0, os.path.join(_project_root, 'data_prep'))

from predict import load_model, infer_episode, CLASSES, CLS_TO_IDX

DATA_DIR = Path('dataset')
OUTPUT_PNG = Path('collusion_evidence.png')

COLLUSION_TYPES = ['none', 'wash', 'paint', 'spoof', 'mirror']

# Color scheme for collusion types
TYPE_COLORS = {
    'none':   '#4CAF50',
    'wash':   '#F44336',
    'paint':  '#FF9800',
    'spoof':  '#9C27B0',
    'mirror': '#2196F3',
}

CA_COLOR = '#E91E63'
CB_COLOR = '#3F51B5'


def load_root_level_episodes():
    """Load episodes from root-level 'orders (N).parquet' + 'label (N).json' files."""
    label_files = sorted(glob.glob(str(Path(_project_root) / 'label (*).json')))
    if not label_files:
        return []

    episodes = []
    for lf in label_files:
        # Extract index from filename: 'label (N).json' -> N
        idx = Path(lf).stem.split('(')[1].rstrip(')')
        orders_path = Path(_project_root) / f'orders ({idx}).parquet'
        trades_path = Path(_project_root) / f'trades ({idx}).parquet'

        if not orders_path.exists():
            continue

        with open(lf) as f:
            label = json.load(f)

        episodes.append({
            'episode_id': f'root_ep_{idx}',
            'collusion_type': label.get('collusion_type', 'none'),
            't_start': label.get('t_start'),
            't_end': label.get('t_end'),
            'orders_path': orders_path,
            'trades_path': trades_path,
        })

    return episodes


def load_dataset_episodes():
    """Load episodes from dataset/ directory (labels.parquet)."""
    labels_path = DATA_DIR / 'labels.parquet'
    if not labels_path.exists():
        return []

    labels_df = pd.read_parquet(labels_path)
    rng = np.random.default_rng(42)
    episodes = []
    for ctype in COLLUSION_TYPES:
        eps = labels_df[labels_df['collusion_type'] == ctype]
        if len(eps) == 0:
            continue
        idx = rng.integers(0, len(eps))
        ep = eps.iloc[idx]
        episodes.append({
            'episode_id': ep['episode_id'],
            'collusion_type': ep['collusion_type'],
            't_start': ep.get('t_start'),
            't_end': ep.get('t_end'),
            'orders_path': DATA_DIR / 'orders' / f'{ep["episode_id"]}.parquet',
            'trades_path': DATA_DIR / 'trades' / f'{ep["episode_id"]}.parquet',
        })
    return episodes


def pick_episodes_by_type(all_episodes):
    """Pick one episode per available collusion type."""
    by_type = {}
    for ep in all_episodes:
        ctype = ep['collusion_type']
        if ctype not in by_type:
            by_type[ctype] = ep
    # Return in canonical order, only types that exist
    return [by_type[ct] for ct in COLLUSION_TYPES if ct in by_type]


def build_mid_series(orders, trades):
    """Build a mid-price time series from trade data."""
    if len(trades) == 0:
        return np.array([0.0]), np.array([100.0])
    t_sorted = trades.sort_values('ts')
    ts = t_sorted['ts'].to_numpy()
    if 'mid_at_trade' in t_sorted.columns:
        mids = t_sorted['mid_at_trade'].to_numpy()
        mask = np.isnan(mids)
        mids[mask] = t_sorted['price'].to_numpy()[mask]
    else:
        mids = t_sorted['price'].to_numpy()
    return ts, mids


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('ERROR: matplotlib required. Install with: pip install matplotlib')
        return

    # Try dataset/ first, fall back to root-level files
    episodes = load_dataset_episodes()
    if not episodes:
        print('  dataset/ not found or empty, trying root-level parquet files...')
        all_eps = load_root_level_episodes()
        episodes = pick_episodes_by_type(all_eps)

    if not episodes:
        print('ERROR: No episode data found. Generate dataset or provide parquet files.')
        return

    available_types = [ep['collusion_type'] for ep in episodes]
    print(f'  Found {len(episodes)} episodes: {available_types}')

    # Load model
    model = load_model()
    print(f'  Model loaded. Generating visualization...\n')

    n_rows = len(episodes)
    fig = plt.figure(figsize=(16, 4.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.35)

    for row_idx, ep in enumerate(episodes):
        ep_id = ep['episode_id']
        ctype = ep['collusion_type']
        t_start = ep.get('t_start')
        t_end = ep.get('t_end')

        print(f'  [{ctype:8s}] episode={ep_id}  '
              f'window=[{t_start} -> {t_end}]')

        # Load data
        orders = pd.read_parquet(ep['orders_path'])
        trades_path = ep['trades_path']
        if trades_path.exists():
            trades = pd.read_parquet(trades_path)
        else:
            trades = pd.DataFrame()

        # Run inference
        preds, ep_pred, ep_conf = infer_episode(model, orders, trades)

        # Create subplot with two axes (price + confidence)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[row_idx], height_ratios=[2, 1], hspace=0.15)
        ax_price = fig.add_subplot(inner_gs[0])
        ax_conf = fig.add_subplot(inner_gs[1], sharex=ax_price)

        # --- Top: Price + Order Flow ---
        ts_mid, mids = build_mid_series(orders, trades)
        ax_price.plot(ts_mid, mids, color='#455A64', alpha=0.6, linewidth=0.8,
                      label='Mid Price', zorder=1)

        # CA orders
        ca_orders = orders[orders['trader'] == 'CA']
        if len(ca_orders) > 0:
            ca_buys = ca_orders[ca_orders['side'] == 'buy']
            ca_sells = ca_orders[ca_orders['side'] == 'sell']
            if len(ca_buys) > 0:
                ax_price.scatter(ca_buys['ts'], ca_buys['price'],
                                marker='^', s=12, alpha=0.5, color=CA_COLOR,
                                label='CA buy', zorder=2)
            if len(ca_sells) > 0:
                ax_price.scatter(ca_sells['ts'], ca_sells['price'],
                                marker='v', s=12, alpha=0.5, color=CA_COLOR,
                                zorder=2)

        # CB orders
        cb_orders = orders[orders['trader'] == 'CB']
        if len(cb_orders) > 0:
            cb_buys = cb_orders[cb_orders['side'] == 'buy']
            cb_sells = cb_orders[cb_orders['side'] == 'sell']
            if len(cb_buys) > 0:
                ax_price.scatter(cb_buys['ts'], cb_buys['price'],
                                marker='^', s=12, alpha=0.5, color=CB_COLOR,
                                label='CB buy', zorder=2)
            if len(cb_sells) > 0:
                ax_price.scatter(cb_sells['ts'], cb_sells['price'],
                                marker='v', s=12, alpha=0.5, color=CB_COLOR,
                                zorder=2)

        # Shade collusion window
        if t_start is not None and t_end is not None and not (
                pd.isna(t_start) or pd.isna(t_end)):
            ax_price.axvspan(t_start, t_end, alpha=0.12,
                            color=TYPE_COLORS[ctype], zorder=0)
            ax_price.axvline(t_start, color=TYPE_COLORS[ctype],
                            linestyle='--', alpha=0.5, linewidth=1)
            ax_price.axvline(t_end, color=TYPE_COLORS[ctype],
                            linestyle='--', alpha=0.5, linewidth=1)

        ax_price.set_ylabel('Price', fontsize=9)
        ax_price.set_title(
            f'{ctype.upper()}  --  {ep_id}  |  '
            f'Detector: {ep_pred} (conf={ep_conf:.2f})',
            fontsize=11, fontweight='bold',
            color=TYPE_COLORS.get(ctype, '#333'))
        ax_price.legend(loc='upper right', fontsize=7, ncol=3)
        ax_price.grid(True, alpha=0.2)
        ax_price.tick_params(labelsize=8)
        plt.setp(ax_price.get_xticklabels(), visible=False)

        # --- Bottom: Detector Confidence ---
        window_starts = preds['window_start'].to_numpy()
        collusion_scores = preds['collusion_score'].to_numpy()
        pred_classes = preds['predicted_class'].to_numpy()
        bar_width = preds['window_end'].iloc[0] - preds['window_start'].iloc[0]

        bar_colors = [TYPE_COLORS.get(c, '#888') for c in pred_classes]

        ax_conf.bar(window_starts + bar_width / 2, collusion_scores,
                    width=bar_width * 0.9, color=bar_colors, alpha=0.7,
                    edgecolor='white', linewidth=0.5)
        ax_conf.axhline(0.5, color='#F44336', linestyle='--', alpha=0.5,
                       linewidth=1, label='Threshold')

        # Shade collusion window in confidence plot too
        if t_start is not None and t_end is not None and not (
                pd.isna(t_start) or pd.isna(t_end)):
            ax_conf.axvspan(t_start, t_end, alpha=0.08,
                           color=TYPE_COLORS[ctype], zorder=0)

        ax_conf.set_ylabel('Collusion\nScore', fontsize=9)
        ax_conf.set_xlabel('Time (s)', fontsize=9)
        ax_conf.set_ylim(0, 1.05)
        ax_conf.set_xlim(0, 600)
        ax_conf.legend(loc='upper right', fontsize=7)
        ax_conf.grid(True, alpha=0.2)
        ax_conf.tick_params(labelsize=8)

    fig.suptitle('Collusion Evidence -- Order Flow + Detector Confidence',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'\n  >> Saved to {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
