"""
train_rl.py -- Train a PPO agent in MarketEnv (Phase 2 final prototype).

Phase 2 improvements over smoke test:
  - Stochastic eval (primary) — matches training policy behavior
  - Deterministic eval logged alongside for comparison
  - Linear LR decay over training
  - KL-based early stopping per iteration (handled in ppo.py)
  - Extended training: 300 iters, 4800 steps/rollout
  - Training curve plot saved as rl_training_curves.png
  - Enhanced logging: grad norm, LR, value loss, passive fills

Loop:
  for iter in range(N_ITERS):
      set LR via linear schedule
      collect ROLLOUT_STEPS transitions (multiple episodes if they fit)
      run PPO update (4 epochs over the rollout, KL early stop)
      log episode stats from the rollout
      every EVAL_EVERY iters: stochastic + deterministic eval
"""
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch

from market_env import MarketEnv
from ppo import PPO

# ---- Hyperparameters ----
N_ITERS = 300
ROLLOUT_STEPS = 4800          # ~8 episodes per rollout (595 steps each)
EVAL_EVERY = 10
EVAL_EPISODES = 5
SEED = 13

CHECKPOINT = Path('ppo_best.pt')
HISTORY = Path('rl_history.json')
CURVES_PNG = Path('rl_training_curves.png')


def evaluate(agent, n_episodes=5, seed_base=10000, deterministic=False):
    """Run evaluation episodes. Stochastic by default (Phase 2 fix)."""
    rewards, marks, positions, fills, invalids, passives = [], [], [], [], [], []
    for i in range(n_episodes):
        env = MarketEnv(seed=seed_base + i)
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            a, _, _ = agent.act(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
        marks.append(info['mark'])
        positions.append(info['position'])
        fills.append(info['fills_taken'])
        invalids.append(info['invalid_count'])
        passives.append(info.get('passive_fills', 0))
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_mark': float(np.mean(marks)),
        'mean_abs_pos': float(np.mean(np.abs(positions))),
        'mean_fills': float(np.mean(fills)),
        'mean_invalid': float(np.mean(invalids)),
        'mean_passive_fills': float(np.mean(passives)),
    }


def plot_training_curves(history, path):
    """Generate training curve plots. Fails silently if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not installed — skipping plot')
        return

    iters = [h['iter'] for h in history]
    rolling = [h.get('rolling20') for h in history]
    entropy = [h.get('entropy') for h in history]
    positions = [h.get('train_abs_pos') for h in history]
    grad_norms = [h.get('grad_norm') for h in history]
    lr_vals = [h.get('lr') for h in history]

    # Eval points
    eval_iters = [h['iter'] for h in history if h.get('eval_stoch')]
    eval_rewards = [h['eval_stoch']['mean_reward'] for h in history if h.get('eval_stoch')]
    eval_det_iters = [h['iter'] for h in history if h.get('eval_det')]
    eval_det_rewards = [h['eval_det']['mean_reward'] for h in history if h.get('eval_det')]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('RL Training Curves — PPO in MarketEnv', fontsize=14, fontweight='bold')

    # 1. Rolling reward + eval reward
    ax = axes[0, 0]
    valid_rolling = [(i, r) for i, r in zip(iters, rolling) if r is not None]
    if valid_rolling:
        ax.plot(*zip(*valid_rolling), color='#2196F3', alpha=0.7, label='Rolling-20 train')
    if eval_rewards:
        ax.plot(eval_iters, eval_rewards, 'o-', color='#4CAF50', markersize=4,
                label='Eval (stochastic)')
    if eval_det_rewards:
        ax.plot(eval_det_iters, eval_det_rewards, 's--', color='#FF9800', markersize=3,
                alpha=0.6, label='Eval (deterministic)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Entropy
    ax = axes[0, 1]
    valid_ent = [(i, e) for i, e in zip(iters, entropy) if e is not None]
    if valid_ent:
        ax.plot(*zip(*valid_ent), color='#9C27B0', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)

    # 3. Mean |position| at episode end
    ax = axes[1, 0]
    valid_pos = [(i, p) for i, p in zip(iters, positions) if p is not None]
    if valid_pos:
        ax.plot(*zip(*valid_pos), color='#F44336', alpha=0.7)
    ax.axhline(500, color='red', linestyle='--', alpha=0.5, label='Hard cap')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean |Position|')
    ax.set_title('Inventory Control')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Gradient norm
    ax = axes[1, 1]
    valid_gn = [(i, g) for i, g in zip(iters, grad_norms) if g is not None]
    if valid_gn:
        ax.plot(*zip(*valid_gn), color='#FF5722', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Grad Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)

    # 5. Learning rate
    ax = axes[2, 0]
    valid_lr = [(i, l) for i, l in zip(iters, lr_vals) if l is not None]
    if valid_lr:
        ax.plot(*zip(*valid_lr), color='#009688', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -3))

    # 6. Eval |position| over time
    ax = axes[2, 1]
    eval_pos_iters = [h['iter'] for h in history if h.get('eval_stoch')]
    eval_pos_vals = [h['eval_stoch']['mean_abs_pos'] for h in history if h.get('eval_stoch')]
    if eval_pos_vals:
        ax.plot(eval_pos_iters, eval_pos_vals, 'o-', color='#E91E63', markersize=4)
    ax.axhline(500, color='red', linestyle='--', alpha=0.5, label='Hard cap')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval Mean |Position|')
    ax.set_title('Eval Inventory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  >> Training curves saved to {path}')


def print_summary(history, best_eval, total_time):
    """Print a clear summary table at end of training."""
    print('\n' + '=' * 70)
    print('  TRAINING SUMMARY')
    print('=' * 70)
    print(f'  Total iterations:        {len(history)}')
    print(f'  Total time:              {total_time:.1f} min')
    print(f'  Best eval reward (stoch): {best_eval:.2f}')

    # Last 20 rolling reward
    recent = [h.get('rolling20') for h in history[-20:] if h.get('rolling20') is not None]
    if recent:
        print(f'  Final rolling-20 reward: {np.mean(recent):.2f}')

    # Final eval stats
    final_evals = [h for h in history if h.get('eval_stoch')]
    if final_evals:
        last = final_evals[-1]['eval_stoch']
        print(f'\n  Final eval (stochastic):')
        print(f'    Mean reward:    {last["mean_reward"]:8.2f} ± {last["std_reward"]:.2f}')
        print(f'    Mean mark:      {last["mean_mark"]:8.2f}')
        print(f'    Mean |pos|:     {last["mean_abs_pos"]:8.0f}')
        print(f'    Mean fills:     {last["mean_fills"]:8.0f}')
        print(f'    Mean invalid:   {last["mean_invalid"]:8.1f}')
        print(f'    Passive fills:  {last["mean_passive_fills"]:8.0f}')

    final_det = [h for h in history if h.get('eval_det')]
    if final_det:
        last = final_det[-1]['eval_det']
        print(f'\n  Final eval (deterministic):')
        print(f'    Mean reward:    {last["mean_reward"]:8.2f} ± {last["std_reward"]:.2f}')
        print(f'    Mean |pos|:     {last["mean_abs_pos"]:8.0f}')

    # Final entropy
    ent_vals = [h.get('entropy') for h in history[-5:] if h.get('entropy') is not None]
    if ent_vals:
        print(f'\n  Final entropy:   {np.mean(ent_vals):.3f}')

    print('=' * 70)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MarketEnv(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f'obs_dim={obs_dim}, n_actions={n_actions}')

    agent = PPO(obs_dim, n_actions,
                lr=3e-4,
                lr_end=5e-5,
                gamma=0.99, lam=0.95,
                clip_eps=0.2, entropy_coef=0.01,
                value_coef=0.5, grad_clip=0.5,
                update_epochs=4, minibatch_size=128, hidden=128,
                target_kl=0.02,
                normalize_obs=True)
    n_params = sum(p.numel() for p in agent.net.parameters())
    print(f'Agent params: {n_params:,}')
    print(f'LR schedule: {agent.lr_start:.1e} -> {agent.lr_end:.1e}')
    print(f'Target KL: {agent.target_kl}')
    print(f'Obs normalization: {agent.normalize_obs}')
    print(f'Rollout steps: {ROLLOUT_STEPS}')
    print(f'Total iterations: {N_ITERS}')
    print()

    history = []
    best_eval = -float('inf')
    recent_rewards = deque(maxlen=20)
    t_start = time.time()

    for it in range(1, N_ITERS + 1):
        t0 = time.time()

        # Linear LR schedule: frac_remaining goes from 1 -> 0
        frac_remaining = 1.0 - (it - 1) / N_ITERS
        lr = agent.set_lr(frac_remaining)

        buf = agent.collect_rollout(env, ROLLOUT_STEPS)
        eps = agent.pop_episodes()
        if eps:
            for e in eps:
                recent_rewards.append(e['reward'])

        metrics = agent.update(buf)
        elapsed = time.time() - t0

        train_mean = np.mean([e['reward'] for e in eps]) if eps else float('nan')
        train_pos = np.mean([abs(e['position']) for e in eps]) if eps else 0.0
        rolling = np.mean(recent_rewards) if recent_rewards else float('nan')
        train_passive = np.mean([e.get('passive_fills', 0) for e in eps]) if eps else 0.0

        # Compact log line
        kl_marker = '*' if metrics.get('early_stopped') else ' '
        line = (f'iter {it:3d}{kl_marker} '
                f'eps={len(eps):2d}  '
                f'R={train_mean:8.2f}  '
                f'roll20={rolling:8.2f}  '
                f'|pos|={train_pos:5.0f}  '
                f'pi={metrics["policy_loss"]:+.3f}  '
                f'v={metrics["value_loss"]:.3f}  '
                f'H={metrics["entropy"]:.3f}  '
                f'KL={metrics["approx_kl"]:+.4f}  '
                f'gn={metrics["grad_norm"]:.2f}  '
                f'lr={lr:.1e}  '
                f'({elapsed:.1f}s)')
        print(line)

        rec = {'iter': it, 'rollout_eps': len(eps),
               'train_mean_reward': float(train_mean) if eps else None,
               'rolling20': float(rolling) if recent_rewards else None,
               'train_abs_pos': float(train_pos),
               'lr': lr,
               **{k: v for k, v in metrics.items() if k != 'updates'}}

        if it % EVAL_EVERY == 0:
            # Primary: stochastic eval
            ev_stoch = evaluate(agent, n_episodes=EVAL_EPISODES,
                                seed_base=99999, deterministic=False)
            # Secondary: deterministic eval for comparison
            ev_det = evaluate(agent, n_episodes=EVAL_EPISODES,
                              seed_base=99999, deterministic=True)

            print(f'  >> EVAL stoch  R={ev_stoch["mean_reward"]:8.2f} '
                  f'± {ev_stoch["std_reward"]:6.2f}  '
                  f'mark={ev_stoch["mean_mark"]:7.2f}  '
                  f'|pos|={ev_stoch["mean_abs_pos"]:5.0f}  '
                  f'fills={ev_stoch["mean_fills"]:6.0f}  '
                  f'passive={ev_stoch["mean_passive_fills"]:4.0f}  '
                  f'invalid={ev_stoch["mean_invalid"]:.1f}')
            print(f'  >> EVAL det    R={ev_det["mean_reward"]:8.2f} '
                  f'± {ev_det["std_reward"]:6.2f}  '
                  f'|pos|={ev_det["mean_abs_pos"]:5.0f}')

            rec['eval_stoch'] = ev_stoch
            rec['eval_det'] = ev_det

            # Best checkpoint by stochastic eval reward
            if ev_stoch['mean_reward'] > best_eval:
                best_eval = ev_stoch['mean_reward']
                agent.save(CHECKPOINT)
                print(f'  >> new best: {best_eval:.2f}, saved to {CHECKPOINT}')

        history.append(rec)

        # Save history every iteration (cheap, enables monitoring)
        HISTORY.write_text(json.dumps(history, indent=2))

    total_time = (time.time() - t_start) / 60

    # Print summary
    print_summary(history, best_eval, total_time)

    # Generate training curve plots
    plot_training_curves(history, CURVES_PNG)


if __name__ == '__main__':
    main()
