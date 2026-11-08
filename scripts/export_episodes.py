"""
export_episodes.py — Convert Parquet episodes → JSON for the frontend simulator.

Picks 2 representative episodes per collusion type (10 total), runs predict.py
on each, then exports a compact JSON with:
  - ticks: [{t, mid, orders:[{trader, side, price, qty}]}]
  - predictions: [{window_start, window_end, predicted, confidence, collusion_score}]
  - mode, label_window, duration

Usage:
    python scripts/export_episodes.py
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import load_model, infer_episode

DATASET = ROOT / "dataset"
OUTPUT  = ROOT / "frontend" / "public" / "episodes"

# Pick 2 episodes per class
PICKS = {
    "none":   ["ep_00000", "ep_00050"],
    "wash":   ["ep_01000", "ep_01050"],
    "paint":  ["ep_02000", "ep_02050"],
    "spoof":  ["ep_03000", "ep_03050"],
    "mirror": ["ep_04000", "ep_04050"],
}


def load_episode(ep_id):
    orders = pd.read_parquet(DATASET / "orders" / f"{ep_id}.parquet")
    trades = pd.read_parquet(DATASET / "trades" / f"{ep_id}.parquet")
    return orders, trades


def build_ticks(orders, trades, bucket_size=1.0, duration=600):
    """Aggregate orders into 1-second ticks with mid-price."""
    ticks = []
    all_trades = trades.sort_values("ts") if len(trades) > 0 else pd.DataFrame()

    for t in np.arange(0, duration, bucket_size):
        t_end = t + bucket_size

        # Orders in this bucket
        bucket = orders[(orders.ts >= t) & (orders.ts < t_end)]
        tick_orders = []
        for _, row in bucket.iterrows():
            tick_orders.append({
                "trader": row.trader,
                "side": row.side,
                "price": round(float(row.price), 2),
                "qty": int(row.qty),
            })

        # Mid price from trades
        if len(all_trades) > 0:
            recent = all_trades[all_trades.ts <= t_end]
            if len(recent) > 0:
                mid = float(recent.iloc[-1].mid_at_trade)
            else:
                mid = 100.0
        else:
            mid = 100.0

        ticks.append({
            "t": int(t),
            "mid": round(mid, 3),
            "orders": tick_orders[:12],  # Cap at 12 per tick for JSON size
        })

    return ticks


def export_episode(ep_id, mode, label_window, model):
    """Export one episode to JSON."""
    orders, trades = load_episode(ep_id)
    duration = 600

    # Build ticks
    ticks = build_ticks(orders, trades, duration=duration)

    # Run CNN predictions
    pair = ("CA", "CB")
    df, ep_pred, ep_conf = infer_episode(
        model, orders, trades, pair=pair,
        window_len=60, stride=30, duration=duration, threshold=0.5,
    )

    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            "window_start": int(row["window_start"]),
            "window_end":   int(row["window_end"]),
            "predicted":    row["predicted_class"],
            "confidence":   round(float(row["confidence"]), 3),
            "collusion_score": round(float(row["collusion_score"]), 3),
        })

    episode = {
        "id": ep_id,
        "mode": mode,
        "duration": duration,
        "label_window": label_window,
        "episode_prediction": ep_pred,
        "episode_confidence": round(float(ep_conf), 3),
        "ticks": ticks,
        "predictions": predictions,
    }

    return episode


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load labels for collusion windows
    labels = pd.read_parquet(DATASET / "labels.parquet")
    labels_map = {}
    for _, row in labels.iterrows():
        t_start = None if pd.isna(row.t_start) else float(row.t_start)
        t_end   = None if pd.isna(row.t_end)   else float(row.t_end)
        labels_map[row.episode_id] = {
            "type": row.collusion_type,
            "window": [t_start, t_end] if t_start is not None else None,
        }

    # Load CNN model
    print("Loading CNN model...")
    model = load_model(None)
    print("  ✓ Model loaded\n")

    # Export manifest
    manifest = {}

    for mode, ep_ids in PICKS.items():
        manifest[mode] = []
        for ep_id in ep_ids:
            print(f"  Exporting {ep_id} ({mode})...")

            info = labels_map.get(ep_id, {"type": mode, "window": None})
            label_window = info["window"]

            try:
                episode = export_episode(ep_id, mode, label_window, model)

                out_path = OUTPUT / f"{ep_id}.json"
                with open(out_path, "w") as f:
                    json.dump(episode, f, separators=(",", ":"))  # compact

                size_kb = out_path.stat().st_size / 1024
                print(f"    ✓ {out_path.name} ({size_kb:.0f} KB) — pred: {episode['episode_prediction']}")

                manifest[mode].append({
                    "id": ep_id,
                    "file": f"{ep_id}.json",
                    "prediction": episode["episode_prediction"],
                    "confidence": episode["episode_confidence"],
                })
            except Exception as e:
                print(f"    ✗ Failed: {e}")

    # Write manifest
    manifest_path = OUTPUT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  ✓ Manifest: {manifest_path}")
    print(f"  ✓ {sum(len(v) for v in manifest.values())} episodes exported to {OUTPUT}")


if __name__ == "__main__":
    main()
