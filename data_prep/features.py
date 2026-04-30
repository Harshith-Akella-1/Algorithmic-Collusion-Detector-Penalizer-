"""
features.py -- window-level feature extraction.

Slides a window over each episode's order/trade log, computing pair-level
features for the (CA, CB) pair. Produces a single features.parquet file
with one row per (episode, window_start) pair.

A window is labeled positive (collusion_type != 'none') if the labeled
collusion period overlaps the window by >= MIN_OVERLAP fraction.
Otherwise the window is labeled 'none', regardless of the episode's
overall class. This means colluding episodes have BOTH positive
(in-window) and negative (out-of-window) samples -- exactly what the
detector needs to learn that collusion is local, not episode-wide.

Run: python features.py
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('dataset')
OUT_FILE = Path('dataset/features.parquet')

WINDOW_LEN = 60.0           # seconds
WINDOW_STRIDE = 30.0        # seconds (50% overlap between consecutive windows)
EPISODE_DUR = 600.0
MIN_OVERLAP = 0.25          # min fraction of window covered by collusion period
                            # for window to be labeled positive

PAIR = ('CA', 'CB')


# =========================================================================
# Per-window feature computation
# =========================================================================
def compute_window_features(orders_w, trades_w, A='CA', B='CB'):
    """Compute pair-level features over a window slice of orders & trades."""
    f = {}

    # ---- Trade-based features ----
    if len(trades_w) == 0:
        ab = pd.DataFrame()
    else:
        ab_mask = (((trades_w['buyer'] == A) & (trades_w['seller'] == B)) |
                   ((trades_w['buyer'] == B) & (trades_w['seller'] == A)))
        ab = trades_w[ab_mask]

    f['ab_trade_count'] = float(len(ab))
    f['ab_trade_qty'] = float(ab['qty'].sum()) if len(ab) else 0.0

    # Inter-trade CV (rhythm signal -- low = regular cadence = paint)
    if len(ab) >= 3:
        gaps = np.diff(np.sort(ab['ts'].to_numpy()))
        f['ab_inter_trade_cv'] = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    else:
        f['ab_inter_trade_cv'] = 1.0

    # Inter-arrival regularity v2: lag-1 autocorrelation of gaps.
    # Periodic painters have low gap variance AND high temporal autocorrelation
    # (gap_i predicts gap_{i+1}). Honest crosses are Poisson-like: zero autocorr.
    if len(ab) >= 5:
        gaps = np.diff(np.sort(ab['ts'].to_numpy()))
        if gaps.std() > 1e-9:
            g0 = gaps[:-1]; g1 = gaps[1:]
            f['ab_gap_autocorr'] = float(np.corrcoef(g0, g1)[0, 1])
            if not np.isfinite(f['ab_gap_autocorr']):
                f['ab_gap_autocorr'] = 0.0
        else:
            f['ab_gap_autocorr'] = 1.0   # constant gaps = perfectly regular
    else:
        f['ab_gap_autocorr'] = 0.0

    # Spectral peak power: FFT of binned arrival series.
    # Bin AB trades into 1s buckets across the window, take FFT, normalize peak
    # power by total power. High value = sharp periodic component = paint.
    f['ab_spectral_peak'] = 0.0
    if len(ab) >= 4:
        ab_ts = ab['ts'].to_numpy()
        t_lo, t_hi = ab_ts.min(), ab_ts.max()
        if t_hi - t_lo > 4:
            n_bins = max(8, int(np.ceil(t_hi - t_lo)))
            counts, _ = np.histogram(ab_ts, bins=n_bins, range=(t_lo, t_hi))
            counts = counts.astype(float)
            counts -= counts.mean()
            spec = np.abs(np.fft.rfft(counts)) ** 2
            if len(spec) > 1 and spec[1:].sum() > 1e-9:
                f['ab_spectral_peak'] = float(spec[1:].max() / spec[1:].sum())

    # Price-deviation from smoothed mid (paint signal: prints near mid).
    # Use rolling pre-trade mid: average mid_at_trade over 5 trades just BEFORE
    # each AB trade. The painter's own trade moves mid, so comparing to the
    # same-instant mid undercounts deviation. Pre-trade smoothed mid sees the
    # market state the painter was trying to track.
    f['ab_price_deviation'] = 0.0
    if len(ab) > 0 and len(trades_w) > 0 and 'mid_at_trade' in trades_w.columns:
        trades_sorted = trades_w.sort_values('ts').reset_index(drop=True)
        trade_ts = trades_sorted['ts'].to_numpy()
        trade_mid = trades_sorted['mid_at_trade'].to_numpy()
        devs = []
        for _, row in ab.iterrows():
            idx = np.searchsorted(trade_ts, row['ts'], side='left')
            lo = max(0, idx - 5)
            window_mids = trade_mid[lo:idx]
            window_mids = window_mids[~np.isnan(window_mids)]
            if len(window_mids) > 0:
                smoothed = window_mids.mean()
                devs.append(abs(row['price'] - smoothed))
        if devs:
            f['ab_price_deviation'] = float(np.mean(devs))

    # ---- Order-based features ----
    A_orders = orders_w[orders_w['trader'] == A] if len(orders_w) else orders_w
    B_orders = orders_w[orders_w['trader'] == B] if len(orders_w) else orders_w

    A_limits = A_orders[A_orders['type'] == 'limit'] if len(A_orders) else A_orders
    A_cancels = A_orders[A_orders['type'] == 'cancel'] if len(A_orders) else A_orders
    B_limits = B_orders[B_orders['type'] == 'limit'] if len(B_orders) else B_orders
    B_cancels = B_orders[B_orders['type'] == 'cancel'] if len(B_orders) else B_orders

    A_lim_qty = float(A_limits['qty'].sum()) if len(A_limits) else 0.0
    A_can_qty = float(A_cancels['qty'].sum()) if len(A_cancels) else 0.0
    B_lim_qty = float(B_limits['qty'].sum()) if len(B_limits) else 0.0
    B_can_qty = float(B_cancels['qty'].sum()) if len(B_cancels) else 0.0

    f['A_cancel_ratio'] = A_can_qty / A_lim_qty if A_lim_qty > 0 else 0.0
    f['B_cancel_ratio'] = B_can_qty / B_lim_qty if B_lim_qty > 0 else 0.0

    f['A_n_limits'] = float(len(A_limits))
    f['A_n_cancels'] = float(len(A_cancels))
    f['B_n_limits'] = float(len(B_limits))
    f['B_n_cancels'] = float(len(B_cancels))

    # ---- Cancel synchrony (mirror signal) ----
    # Number of A cancels with a B cancel within +/- 1.0 s
    if len(A_cancels) > 0 and len(B_cancels) > 0:
        a_ts = A_cancels['ts'].to_numpy()
        b_ts = np.sort(B_cancels['ts'].to_numpy())
        lo = np.searchsorted(b_ts, a_ts - 1.0, side='left')
        hi = np.searchsorted(b_ts, a_ts + 1.0, side='right')
        matched = int((hi > lo).sum())
        f['sync_cancel_count'] = float(matched)
        f['sync_cancel_ratio'] = matched / len(a_ts)
    else:
        f['sync_cancel_count'] = 0.0
        f['sync_cancel_ratio'] = 0.0

    # ---- Same-side depth evaporation (mirror signal) ----
    # Max total CA+CB cancel qty on one side within any 2s sub-window.
    # Vectorized: per side, sort by ts, then for each event sum qty over
    # the trailing 2s using cumulative sum + searchsorted (O(n log n)).
    f['same_side_evap'] = 0.0
    if len(A_cancels) + len(B_cancels) > 0:
        ab_cancels = pd.concat([A_cancels, B_cancels])
        for side in ('buy', 'sell'):
            sub = ab_cancels[ab_cancels['side'] == side]
            if len(sub) == 0:
                continue
            sub = sub.sort_values('ts')
            ts = sub['ts'].to_numpy()
            qty = sub['qty'].to_numpy(dtype=float)
            csum = np.concatenate([[0.0], np.cumsum(qty)])
            # for each event at ts[i], find idx of first ts >= ts[i] - 2.0
            lo = np.searchsorted(ts, ts - 2.0, side='left')
            window_sums = csum[np.arange(1, len(ts) + 1)] - csum[lo]
            best = float(window_sums.max())
            if best > f['same_side_evap']:
                f['same_side_evap'] = best

    # ---- Cancel burst (spoof signal) ----
    # Max number of A cancel events in any 5s sub-window
    if len(A_cancels) > 0:
        ts_bins = (A_cancels['ts'] // 5.0).astype(int)
        f['A_cancel_burst'] = float(ts_bins.value_counts().max())
    else:
        f['A_cancel_burst'] = 0.0

    return f


# =========================================================================
# Window slicing & labeling
# =========================================================================
def slice_window(orders, trades, t0, t1):
    o = orders[(orders['ts'] >= t0) & (orders['ts'] < t1)]
    t = trades[(trades['ts'] >= t0) & (trades['ts'] < t1)]
    return o, t


def overlap_fraction(t0, t1, c_start, c_end):
    """Fraction of [t0, t1] covered by [c_start, c_end]."""
    if c_start is None or c_end is None or pd.isna(c_start) or pd.isna(c_end):
        return 0.0
    overlap = max(0.0, min(t1, c_end) - max(t0, c_start))
    return overlap / (t1 - t0)


# =========================================================================
# Main extraction loop
# =========================================================================
def extract_all(data_dir=DATA_DIR, out_file=OUT_FILE):
    data_dir = Path(data_dir)
    labels_df = pd.read_parquet(data_dir / 'labels.parquet')
    n_eps = len(labels_df)
    print(f'Extracting features from {n_eps} episodes')
    print(f'Window: {WINDOW_LEN}s, stride: {WINDOW_STRIDE}s, '
          f'min overlap: {MIN_OVERLAP}\n')

    rows = []
    t0 = time.time()
    for i, ep in labels_df.iterrows():
        ep_id = ep['episode_id']
        orders = pd.read_parquet(data_dir / 'orders' / f'{ep_id}.parquet')
        trades = pd.read_parquet(data_dir / 'trades' / f'{ep_id}.parquet')

        c_start, c_end = ep.get('t_start'), ep.get('t_end')
        ep_type = ep['collusion_type']

        wt = 0.0
        while wt + WINDOW_LEN <= EPISODE_DUR:
            ow, tw = slice_window(orders, trades, wt, wt + WINDOW_LEN)
            feats = compute_window_features(ow, tw)

            ovr = overlap_fraction(wt, wt + WINDOW_LEN, c_start, c_end)
            window_label = ep_type if (ep_type != 'none' and ovr >= MIN_OVERLAP) else 'none'
            window_binary = 0 if window_label == 'none' else 1

            row = {
                'episode_id': ep_id,
                'window_start': wt,
                'window_end': wt + WINDOW_LEN,
                'episode_type': ep_type,
                'window_type': window_label,
                'is_collusion': window_binary,
                'overlap_frac': ovr,
                **feats,
            }
            rows.append(row)
            wt += WINDOW_STRIDE

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1:5d}/{n_eps}  ({(i+1)/elapsed:.1f} ep/s)')

    feats_df = pd.DataFrame(rows)
    feats_df.to_parquet(out_file, compression='snappy', index=False)
    print(f'\nWrote {len(feats_df)} windows -> {out_file}')

    print('\nWindow-level class distribution:')
    print(feats_df['window_type'].value_counts().to_string())
    print('\nBinary collusion balance:')
    print(feats_df['is_collusion'].value_counts().to_string())
    return feats_df


if __name__ == '__main__':
    extract_all()
