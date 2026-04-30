"""
predict.py -- Inference script for the collusion detector.

Loads the trained CNN (cnn_best.pt) and runs sliding-window classification
on any orders+trades file pair (CSV or Parquet).

Usage:
    python predict.py --orders orders.csv --trades trades.csv
    python predict.py --orders dataset/orders/ep_00042.parquet \\
                      --trades dataset/trades/ep_00042.parquet
    python predict.py --orders orders.csv --trades trades.csv \\
                      --output predictions.csv --threshold 0.3

Output: per-window predictions with class probabilities.
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Fix imports: modules live in subdirectories
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, 'detectors'))
sys.path.insert(0, os.path.join(_project_root, 'data_prep'))

from train_cnn import CNN1D, CLASSES, CLS_TO_IDX, DEVICE, N_FEATURES, SEQ_LEN
from prepare_sequences import encode_window

CHECKPOINT = Path(_project_root) / 'detectors' / 'cnn_best.pt'
WINDOW_LEN = 60.0
WINDOW_STRIDE = 30.0
EPISODE_DUR = 600.0
THRESHOLD = 0.5


def load_file(path):
    """Load a CSV or Parquet file, auto-detecting format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def load_model(checkpoint_path=None):
    """Load the trained CNN model."""
    cp = Path(checkpoint_path) if checkpoint_path else CHECKPOINT
    if not cp.exists():
        raise FileNotFoundError(
            f'Model checkpoint not found: {cp}\n'
            f'Run detectors/train_cnn.py first to train the model.'
        )
    model = CNN1D().to(DEVICE)
    model.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def infer_episode(model, orders, trades, pair=('CA', 'CB'),
                  window_len=WINDOW_LEN, stride=WINDOW_STRIDE,
                  duration=EPISODE_DUR, threshold=THRESHOLD):
    """Run sliding-window CNN inference on one episode.

    Args:
        model: loaded CNN1D model
        orders: DataFrame with columns [ts, oid, trader, side, type, price, qty]
        trades: DataFrame with columns [ts, buyer, seller, price, qty, ...]
        pair: tuple of trader names to monitor (default CA/CB)
        window_len: window length in seconds
        stride: stride between windows in seconds
        duration: total episode duration
        threshold: confidence threshold for collusion flag

    Returns:
        DataFrame with per-window predictions
    """
    results = []
    t = 0.0
    while t + window_len <= duration + 0.01:
        t_end = t + window_len

        # Slice window
        o_win = orders[(orders['ts'] >= t) & (orders['ts'] < t_end)].copy()
        n_events = len(o_win[o_win['trader'].isin(pair)]) if len(o_win) > 0 else 0

        # Remap trader names to CA/CB if needed (CNN was trained on CA/CB)
        if pair != ('CA', 'CB') and len(o_win) > 0:
            name_map = {pair[0]: 'CA', pair[1]: 'CB'}
            o_win['trader'] = o_win['trader'].map(
                lambda x: name_map.get(x, x))

        # Encode sequence (same as training pipeline)
        x = encode_window(o_win, t, t_end)  # (N_FEATURES, SEQ_LEN)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # (1, F, L)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        collusion_prob = float(1.0 - probs[CLS_TO_IDX['none']])

        row = {
            'window_start': round(t, 1),
            'window_end': round(t_end, 1),
            'n_pair_events': n_events,
            'predicted_class': pred_class,
            'confidence': round(confidence, 4),
            'collusion_score': round(collusion_prob, 4),
        }
        for cls in CLASSES:
            row[f'p_{cls}'] = round(float(probs[CLS_TO_IDX[cls]]), 4)

        results.append(row)
        t += stride

    df = pd.DataFrame(results)

    # Episode-level summary
    flagged = df[df['collusion_score'] >= threshold]
    if len(flagged) == 0:
        episode_pred = 'none'
        episode_confidence = float(df['p_none'].mean())
    else:
        # Majority class among flagged windows (exclude 'none')
        type_counts = flagged['predicted_class'].value_counts()
        type_counts = type_counts.drop('none', errors='ignore')
        if len(type_counts) > 0:
            episode_pred = type_counts.index[0]
            episode_confidence = float(
                flagged[flagged['predicted_class'] == episode_pred]
                ['confidence'].mean())
        else:
            episode_pred = 'none'
            episode_confidence = float(df['p_none'].mean())

    return df, episode_pred, episode_confidence


def main():
    parser = argparse.ArgumentParser(
        description='Run collusion detection on orders+trades files')
    parser.add_argument('--orders', required=True,
                        help='Path to orders file (CSV or Parquet)')
    parser.add_argument('--trades', required=True,
                        help='Path to trades file (CSV or Parquet)')
    parser.add_argument('--model', default=None,
                        help='Path to CNN checkpoint (default: detectors/cnn_best.pt)')
    parser.add_argument('--output', default=None,
                        help='Path to save predictions CSV')
    parser.add_argument('--window', type=float, default=WINDOW_LEN,
                        help=f'Window length in seconds (default: {WINDOW_LEN})')
    parser.add_argument('--stride', type=float, default=WINDOW_STRIDE,
                        help=f'Window stride in seconds (default: {WINDOW_STRIDE})')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help=f'Collusion detection threshold (default: {THRESHOLD})')
    parser.add_argument('--pair', nargs=2, default=['CA', 'CB'],
                        help='Trader pair to monitor (default: CA CB)')
    args = parser.parse_args()

    print('=' * 60)
    print('  COLLUSION DETECTOR -- INFERENCE')
    print('=' * 60)
    print(f'  Orders:    {args.orders}')
    print(f'  Trades:    {args.trades}')
    print(f'  Window:    {args.window}s / stride {args.stride}s')
    print(f'  Pair:      {args.pair}')
    print(f'  Threshold: {args.threshold}')
    print()

    # Load data
    orders = load_file(args.orders)
    trades = load_file(args.trades)
    print(f'  Loaded {len(orders):,} orders, {len(trades):,} trades')

    # Detect episode duration from data
    max_ts = max(orders['ts'].max(), trades['ts'].max()) if len(trades) > 0 else orders['ts'].max()
    duration = max(EPISODE_DUR, float(np.ceil(max_ts / 10) * 10))

    # Load model
    model = load_model(args.model)
    print(f'  Model loaded from {args.model or CHECKPOINT}')
    print()

    # Run inference
    df, ep_pred, ep_conf = infer_episode(
        model, orders, trades,
        pair=tuple(args.pair),
        window_len=args.window,
        stride=args.stride,
        duration=duration,
        threshold=args.threshold,
    )

    # Print per-window results
    print('Per-window predictions:')
    print('-' * 90)
    for _, row in df.iterrows():
        flag = '!!' if row['collusion_score'] >= args.threshold else '  '
        print(f'  {flag} [{row["window_start"]:5.0f}s - {row["window_end"]:5.0f}s]  '
              f'{row["predicted_class"]:8s}  '
              f'conf={row["confidence"]:.3f}  '
              f'collusion_score={row["collusion_score"]:.3f}  '
              f'events={row["n_pair_events"]:3d}')
    print('-' * 90)

    # Episode-level summary
    n_flagged = len(df[df['collusion_score'] >= args.threshold])
    print(f'\nEPISODE SUMMARY:')
    print(f'  Prediction:        {ep_pred}')
    print(f'  Confidence:        {ep_conf:.3f}')
    print(f'  Windows flagged:   {n_flagged}/{len(df)}')
    print(f'  Mean collusion:    {df["collusion_score"].mean():.3f}')

    # Save output
    if args.output:
        df.to_csv(args.output, index=False)
        print(f'\n  Predictions saved to {args.output}')

    return df, ep_pred, ep_conf


if __name__ == '__main__':
    main()
