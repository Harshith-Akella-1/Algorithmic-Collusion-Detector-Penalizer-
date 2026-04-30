"""
infer_rl_episodes.py -- Run CNN detector on RL agent episodes.

The 'honesty proof': shows the detector correctly classifies RL agent
behavior as non-colluding. Two modes:

    python infer_rl_episodes.py --agent ppo     # single-agent PPO
    python infer_rl_episodes.py --agent mappo   # two-agent IPPO

For PPO: loads ppo_best.pt, runs 5 episodes recording orders/trades,
         runs CNN inference on each.
For MAPPO: reads pre-recorded episodes from rl_bots/mappo_episodes/.

Expected result: all windows classified as 'none'.
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Fix imports
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, 'rl_bots'))
sys.path.insert(0, os.path.join(_project_root, 'detectors'))
sys.path.insert(0, os.path.join(_project_root, 'data_prep'))

from predict import load_model, infer_episode


def run_ppo_episodes(n_episodes=5, seed_base=70000):
    """Run single-agent PPO and record episodes."""
    from market_env import MarketEnv
    from ppo import PPO

    checkpoint = Path(_project_root) / 'rl_bots' / 'ppo_best.pt'
    if not checkpoint.exists():
        print(f'ERROR: {checkpoint} not found. Train the PPO agent first.')
        return []

    env = MarketEnv(seed=seed_base)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPO(obs_dim, n_actions, normalize_obs=True)
    agent.load(checkpoint)
    print(f'  Loaded PPO from {checkpoint}')

    episodes = []
    for i in range(n_episodes):
        env = MarketEnv(seed=seed_base + i, agent_name='RL')
        obs, _ = env.reset()
        done = False

        while not done:
            a, _, _ = agent.act(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

        # Extract orders and trades
        # Rename RL agent orders to CA/CB pattern for detector compatibility
        orders_df = pd.DataFrame(env.lob.order_log)
        trades_list = []
        for tr in env.lob.trades:
            trades_list.append({
                'ts': tr['ts'], 'buyer': tr['buyer'], 'seller': tr['seller'],
                'price': tr['price'], 'qty': tr['qty'],
                'aggressor': tr.get('aggressor', ''),
                'mid_at_trade': tr.get('mid_at_trade'),
            })
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        episodes.append({
            'orders': orders_df,
            'trades': trades_df,
            'info': info,
            'label': f'ppo_ep_{i}',
        })
        print(f'  ep {i}: mark={info["mark"]:.1f}  pos={info["position"]:+d}  '
              f'fills={info["fills_taken"]}  orders={len(orders_df)}')

    return episodes


def load_mappo_episodes():
    """Load pre-recorded MAPPO episodes."""
    record_dir = Path(_project_root) / 'rl_bots' / 'mappo_episodes'
    if not record_dir.exists():
        print(f'ERROR: {record_dir} not found. Train MAPPO first.')
        return []

    episodes = []
    ep_dirs = sorted(record_dir.iterdir())
    for ep_dir in ep_dirs:
        if not ep_dir.is_dir():
            continue
        orders_path = ep_dir / 'orders.parquet'
        trades_path = ep_dir / 'trades.parquet'
        if not orders_path.exists():
            continue

        orders = pd.read_parquet(orders_path)
        trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
        episodes.append({
            'orders': orders,
            'trades': trades,
            'info': {},
            'label': ep_dir.name,
        })

    print(f'  Loaded {len(episodes)} MAPPO episodes from {record_dir}')
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description='Run CNN detector on RL agent episodes')
    parser.add_argument('--agent', choices=['ppo', 'mappo'], default='ppo',
                        help='Which agent to evaluate (default: ppo)')
    parser.add_argument('--n-episodes', type=int, default=5,
                        help='Number of PPO episodes to run (default: 5)')
    args = parser.parse_args()

    print('=' * 60)
    print(f'  RL -> DETECTOR CROSS-VALIDATION ({args.agent.upper()})')
    print('=' * 60)
    print()

    # Load detector
    model = load_model()
    print(f'  CNN detector loaded.')

    # Get episodes
    if args.agent == 'ppo':
        print(f'\n  Running {args.n_episodes} PPO episodes...')
        episodes = run_ppo_episodes(n_episodes=args.n_episodes)
        # For PPO, the agent trades as 'RL' — detector looks for CA/CB
        # So we need to map RL -> CA for the detector to pick it up
        pair = ('RL', 'NOBODY')  # RL is the only agent, no pair
    else:
        print(f'\n  Loading MAPPO episodes...')
        episodes = load_mappo_episodes()
        pair = ('RL_A', 'RL_B')

    if not episodes:
        print('\n  No episodes to analyze. Exiting.')
        return

    # Run inference on each episode
    print(f'\n  Running CNN inference...')
    print('-' * 70)

    all_preds = []
    for ep_data in episodes:
        orders = ep_data['orders']
        trades = ep_data['trades']
        label = ep_data['label']

        preds, ep_pred, ep_conf = infer_episode(
            model, orders, trades, pair=pair,
            threshold=0.5
        )

        n_flagged = len(preds[preds['collusion_score'] >= 0.5])
        mean_score = preds['collusion_score'].mean()

        status = 'OK' if ep_pred == 'none' else 'XX'
        print(f'  {status} {label:15s}  pred={ep_pred:8s}  conf={ep_conf:.3f}  '
              f'flagged={n_flagged}/{len(preds)}  mean_score={mean_score:.3f}')

        all_preds.append({
            'episode': label,
            'prediction': ep_pred,
            'confidence': ep_conf,
            'windows_flagged': n_flagged,
            'total_windows': len(preds),
            'mean_collusion_score': mean_score,
        })

    print('-' * 70)

    # Summary
    summary_df = pd.DataFrame(all_preds)
    n_correct = (summary_df['prediction'] == 'none').sum()
    n_total = len(summary_df)

    print(f'\n  SUMMARY:')
    print(f'  Episodes classified as "none": {n_correct}/{n_total}')
    if n_correct == n_total:
        print(f'  [PASS] All episodes correctly classified as non-colluding.')
        print(f'         The detector does not false-positive on legitimate RL trading.')
    else:
        false_pos = summary_df[summary_df['prediction'] != 'none']
        print(f'  [WARN] {n_total - n_correct} episode(s) falsely flagged:')
        for _, row in false_pos.iterrows():
            print(f'     {row["episode"]}: {row["prediction"]} '
                  f'(conf={row["confidence"]:.3f})')

    return summary_df


if __name__ == '__main__':
    main()
