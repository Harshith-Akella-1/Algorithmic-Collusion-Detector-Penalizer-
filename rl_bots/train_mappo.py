"""
train_mappo.py -- Train two independent PPO agents in MultiAgentMarketEnv (IPPO).

Two separate PPO instances (agent_a, agent_b) learn to trade profitably in
the same LOB. They share the market but have independent observations,
policies, and rewards. This is IPPO (Independent PPO), not centralized-
critic MAPPO — simpler and sufficient for the "two honest agents" demo.

After training, records 5 episodes for detector cross-validation.

Run: python train_mappo.py [--iters N]
"""
import json
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch

from mappo_env import MultiAgentMarketEnv
from ppo import PPO, RolloutBuffer

# ---- Hyperparameters ----
N_ITERS = 100
ROLLOUT_STEPS = 4800
EVAL_EVERY = 10
EVAL_EPISODES = 5
RECORD_EPISODES = 5
SEED = 42

CHECKPOINT_A = Path('mappo_a_best.pt')
CHECKPOINT_B = Path('mappo_b_best.pt')
HISTORY_FILE = Path('mappo_history.json')
CURVES_PNG = Path('mappo_training_curves.png')
RECORD_DIR = Path('mappo_episodes')


def collect_multi_rollout(env, agent_a, agent_b, n_steps):
    """Collect rollout from both agents acting in lockstep.

    Returns two RolloutBuffers, one per agent.
    """
    obs_dim = env.observation_space.shape[0]

    # Buffers for both agents
    bufs = {}
    for name in ('a', 'b'):
        bufs[name] = {
            'obs': np.zeros((n_steps, obs_dim), dtype=np.float32),
            'act': np.zeros(n_steps, dtype=np.int64),
            'logp': np.zeros(n_steps, dtype=np.float32),
            'rew': np.zeros(n_steps, dtype=np.float32),
            'val': np.zeros(n_steps, dtype=np.float32),
            'done': np.zeros(n_steps, dtype=np.float32),
        }

    # Init state if needed
    if not hasattr(agent_a, '_ma_obs_a') or agent_a._ma_obs_a is None:
        (obs_a, obs_b), _ = env.reset()
        agent_a._ma_obs_a = obs_a
        agent_b._ma_obs_b = obs_b
        agent_a._ma_ep_r_a = 0.0
        agent_b._ma_ep_r_b = 0.0
        agent_a._ma_ep_len = 0
        agent_a._ma_completed = []

    obs_a = agent_a._ma_obs_a
    obs_b = agent_b._ma_obs_b

    for t in range(n_steps):
        # Normalize and update obs stats
        if agent_a.obs_rms is not None:
            agent_a.obs_rms.update(obs_a)
            obs_a_norm = agent_a.obs_rms.normalize(obs_a)
        else:
            obs_a_norm = obs_a

        if agent_b.obs_rms is not None:
            agent_b.obs_rms.update(obs_b)
            obs_b_norm = agent_b.obs_rms.normalize(obs_b)
        else:
            obs_b_norm = obs_b

        # Act
        act_a, logp_a, val_a = agent_a.act(obs_a)
        act_b, logp_b, val_b = agent_b.act(obs_b)

        # Step
        (next_obs_a, next_obs_b), (r_a, r_b), terminated, truncated, info = \
            env.step(act_a, act_b)
        done = terminated or truncated

        # Store
        bufs['a']['obs'][t] = obs_a_norm
        bufs['a']['act'][t] = act_a
        bufs['a']['logp'][t] = logp_a
        bufs['a']['rew'][t] = r_a
        bufs['a']['val'][t] = val_a
        bufs['a']['done'][t] = 1.0 if done else 0.0

        bufs['b']['obs'][t] = obs_b_norm
        bufs['b']['act'][t] = act_b
        bufs['b']['logp'][t] = logp_b
        bufs['b']['rew'][t] = r_b
        bufs['b']['val'][t] = val_b
        bufs['b']['done'][t] = 1.0 if done else 0.0

        agent_a._ma_ep_r_a += r_a
        agent_b._ma_ep_r_b += r_b
        agent_a._ma_ep_len += 1

        if done:
            agent_a._ma_completed.append({
                'reward_a': agent_a._ma_ep_r_a,
                'reward_b': agent_b._ma_ep_r_b,
                'length': agent_a._ma_ep_len,
                'info': info,
            })
            (obs_a, obs_b), _ = env.reset()
            agent_a._ma_ep_r_a = 0.0
            agent_b._ma_ep_r_b = 0.0
            agent_a._ma_ep_len = 0
        else:
            obs_a = next_obs_a
            obs_b = next_obs_b

    agent_a._ma_obs_a = obs_a
    agent_b._ma_obs_b = obs_b

    # Bootstrap values
    def bootstrap_and_gae(agent, obs, buf, gamma=0.99, lam=0.95):
        with torch.no_grad():
            if agent.obs_rms is not None:
                obs_norm = agent.obs_rms.normalize(obs)
            else:
                obs_norm = obs
            x = torch.from_numpy(obs_norm).float().unsqueeze(0).to(agent.device)
            _, last_val = agent.net(x)
            last_val = float(last_val.item())

        n = n_steps
        adv = np.zeros(n, dtype=np.float32)
        ret = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(n)):
            next_val = last_val if i == n - 1 else buf['val'][i + 1]
            next_nonterminal = 1.0 - buf['done'][i]
            delta = buf['rew'][i] + gamma * next_val * next_nonterminal - buf['val'][i]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[i] = gae
            ret[i] = adv[i] + buf['val'][i]

        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    adv_a, ret_a = bootstrap_and_gae(agent_a, obs_a, bufs['a'])
    adv_b, ret_b = bootstrap_and_gae(agent_b, obs_b, bufs['b'])

    def to_buffer(buf, adv, ret, device):
        return RolloutBuffer(
            obs=torch.from_numpy(buf['obs']).to(device),
            actions=torch.from_numpy(buf['act']).to(device),
            log_probs=torch.from_numpy(buf['logp']).to(device),
            rewards=torch.from_numpy(buf['rew']).to(device),
            values=torch.from_numpy(buf['val']).to(device),
            dones=torch.from_numpy(buf['done']).to(device),
            advantages=torch.from_numpy(adv).to(device),
            returns=torch.from_numpy(ret).to(device),
        )

    buf_a = to_buffer(bufs['a'], adv_a, ret_a, agent_a.device)
    buf_b = to_buffer(bufs['b'], adv_b, ret_b, agent_b.device)
    return buf_a, buf_b


def pop_completed(agent_a):
    eps = getattr(agent_a, '_ma_completed', [])
    agent_a._ma_completed = []
    return eps


def evaluate_multi(agent_a, agent_b, n_episodes=5, seed_base=10000,
                   deterministic=False):
    """Evaluate both agents together."""
    results_a, results_b = [], []
    for i in range(n_episodes):
        env = MultiAgentMarketEnv(seed=seed_base + i)
        (obs_a, obs_b), _ = env.reset()
        done = False
        r_a_total, r_b_total = 0.0, 0.0

        while not done:
            a_a, _, _ = agent_a.act(obs_a, deterministic=deterministic)
            a_b, _, _ = agent_b.act(obs_b, deterministic=deterministic)
            (obs_a, obs_b), (r_a, r_b), terminated, truncated, info = \
                env.step(a_a, a_b)
            r_a_total += r_a
            r_b_total += r_b
            done = terminated or truncated

        results_a.append({
            'reward': r_a_total, **info['agent_a']
        })
        results_b.append({
            'reward': r_b_total, **info['agent_b']
        })

    def summarize(results):
        return {
            'mean_reward': float(np.mean([r['reward'] for r in results])),
            'std_reward': float(np.std([r['reward'] for r in results])),
            'mean_mark': float(np.mean([r['mark'] for r in results])),
            'mean_abs_pos': float(np.mean([abs(r['position']) for r in results])),
            'mean_fills': float(np.mean([r['fills_taken'] for r in results])),
            'mean_invalid': float(np.mean([r['invalid_count'] for r in results])),
        }

    return summarize(results_a), summarize(results_b)


def record_final_episodes(agent_a, agent_b, n_episodes=RECORD_EPISODES,
                          seed_base=50000, record_dir=RECORD_DIR):
    """Record episodes with trained agents for detector cross-validation."""
    print(f'\nRecording {n_episodes} episodes for detector cross-validation...')
    record_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        env = MultiAgentMarketEnv(seed=seed_base + i,
                                  record_episodes=True,
                                  record_dir=record_dir)
        (obs_a, obs_b), _ = env.reset()
        done = False
        while not done:
            a_a, _, _ = agent_a.act(obs_a, deterministic=False)
            a_b, _, _ = agent_b.act(obs_b, deterministic=False)
            (obs_a, obs_b), (r_a, r_b), terminated, truncated, info = \
                env.step(a_a, a_b)
            done = terminated or truncated
        print(f'  ep {i}: A mark={info["agent_a"]["mark"]:.1f} '
              f'pos={info["agent_a"]["position"]:+d}  |  '
              f'B mark={info["agent_b"]["mark"]:.1f} '
              f'pos={info["agent_b"]["position"]:+d}')

    print(f'  >> Episodes saved to {record_dir}/')


def plot_mappo_curves(history, path):
    """Generate MAPPO training curve plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not installed — skipping plot')
        return

    iters = [h['iter'] for h in history]
    roll_a = [h.get('rolling_a') for h in history]
    roll_b = [h.get('rolling_b') for h in history]
    ent_a = [h.get('entropy_a') for h in history]
    ent_b = [h.get('entropy_b') for h in history]

    eval_iters = [h['iter'] for h in history if h.get('eval_a')]
    eval_a = [h['eval_a']['mean_reward'] for h in history if h.get('eval_a')]
    eval_b = [h['eval_b']['mean_reward'] for h in history if h.get('eval_b')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IPPO Training Curves — Two Agents in Shared LOB',
                 fontsize=14, fontweight='bold')

    # Rewards
    ax = axes[0, 0]
    valid_a = [(i, r) for i, r in zip(iters, roll_a) if r is not None]
    valid_b = [(i, r) for i, r in zip(iters, roll_b) if r is not None]
    if valid_a:
        ax.plot(*zip(*valid_a), color='#E91E63', alpha=0.7, label='Agent A (roll-20)')
    if valid_b:
        ax.plot(*zip(*valid_b), color='#3F51B5', alpha=0.7, label='Agent B (roll-20)')
    if eval_a:
        ax.plot(eval_iters, eval_a, 'o-', color='#E91E63', markersize=3,
                alpha=0.5, label='A eval')
    if eval_b:
        ax.plot(eval_iters, eval_b, 's-', color='#3F51B5', markersize=3,
                alpha=0.5, label='B eval')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[0, 1]
    valid_ea = [(i, e) for i, e in zip(iters, ent_a) if e is not None]
    valid_eb = [(i, e) for i, e in zip(iters, ent_b) if e is not None]
    if valid_ea:
        ax.plot(*zip(*valid_ea), color='#E91E63', alpha=0.7, label='Agent A')
    if valid_eb:
        ax.plot(*zip(*valid_eb), color='#3F51B5', alpha=0.7, label='Agent B')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Positions
    ax = axes[1, 0]
    pos_a = [h.get('train_pos_a') for h in history]
    pos_b = [h.get('train_pos_b') for h in history]
    va = [(i, p) for i, p in zip(iters, pos_a) if p is not None]
    vb = [(i, p) for i, p in zip(iters, pos_b) if p is not None]
    if va:
        ax.plot(*zip(*va), color='#E91E63', alpha=0.7, label='Agent A')
    if vb:
        ax.plot(*zip(*vb), color='#3F51B5', alpha=0.7, label='Agent B')
    ax.axhline(500, color='red', linestyle='--', alpha=0.5, label='Hard cap')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean |Position|')
    ax.set_title('Inventory Control')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Eval positions
    ax = axes[1, 1]
    eval_pos_a = [h['eval_a']['mean_abs_pos'] for h in history if h.get('eval_a')]
    eval_pos_b = [h['eval_b']['mean_abs_pos'] for h in history if h.get('eval_b')]
    if eval_pos_a:
        ax.plot(eval_iters, eval_pos_a, 'o-', color='#E91E63', markersize=4,
                label='Agent A')
    if eval_pos_b:
        ax.plot(eval_iters, eval_pos_b, 's-', color='#3F51B5', markersize=4,
                label='Agent B')
    ax.axhline(500, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval Mean |Position|')
    ax.set_title('Eval Inventory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  >> Training curves saved to {path}')


def main():
    parser = argparse.ArgumentParser(description='Train IPPO agents')
    parser.add_argument('--iters', type=int, default=N_ITERS,
                        help=f'Number of training iterations (default: {N_ITERS})')
    args = parser.parse_args()
    n_iters = args.iters

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MultiAgentMarketEnv(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f'obs_dim={obs_dim}, n_actions={n_actions}')
    print(f'IPPO: two independent PPO agents in shared LOB')

    # Create two independent PPO agents
    ppo_kwargs = dict(
        obs_dim=obs_dim, n_actions=n_actions,
        lr=3e-4, lr_end=5e-5,
        gamma=0.99, lam=0.95,
        clip_eps=0.2, entropy_coef=0.01,
        value_coef=0.5, grad_clip=0.5,
        update_epochs=4, minibatch_size=128, hidden=128,
        target_kl=0.02, normalize_obs=True,
    )
    agent_a = PPO(**ppo_kwargs)
    agent_b = PPO(**ppo_kwargs)
    n_params = sum(p.numel() for p in agent_a.net.parameters())
    print(f'Agent params (each): {n_params:,}')
    print(f'Training iterations: {n_iters}')
    print(f'Rollout steps: {ROLLOUT_STEPS}')
    print()

    history = []
    best_eval_a = -float('inf')
    best_eval_b = -float('inf')
    recent_a = deque(maxlen=20)
    recent_b = deque(maxlen=20)
    t_start = time.time()

    for it in range(1, n_iters + 1):
        t0 = time.time()

        frac_remaining = 1.0 - (it - 1) / n_iters
        lr_a = agent_a.set_lr(frac_remaining)
        agent_b.set_lr(frac_remaining)

        buf_a, buf_b = collect_multi_rollout(env, agent_a, agent_b, ROLLOUT_STEPS)
        eps = pop_completed(agent_a)

        for e in eps:
            recent_a.append(e['reward_a'])
            recent_b.append(e['reward_b'])

        metrics_a = agent_a.update(buf_a)
        metrics_b = agent_b.update(buf_b)
        elapsed = time.time() - t0

        mean_r_a = np.mean([e['reward_a'] for e in eps]) if eps else float('nan')
        mean_r_b = np.mean([e['reward_b'] for e in eps]) if eps else float('nan')
        pos_a = np.mean([abs(e['info']['agent_a']['position']) for e in eps]) if eps else 0
        pos_b = np.mean([abs(e['info']['agent_b']['position']) for e in eps]) if eps else 0
        rolling_a = np.mean(recent_a) if recent_a else float('nan')
        rolling_b = np.mean(recent_b) if recent_b else float('nan')

        kl_a = '*' if metrics_a.get('early_stopped') else ' '
        kl_b = '*' if metrics_b.get('early_stopped') else ' '

        print(f'iter {it:3d}  eps={len(eps):2d}  '
              f'A: R={mean_r_a:7.2f} roll={rolling_a:7.2f} |p|={pos_a:4.0f} '
              f'H={metrics_a["entropy"]:.3f}{kl_a}  '
              f'B: R={mean_r_b:7.2f} roll={rolling_b:7.2f} |p|={pos_b:4.0f} '
              f'H={metrics_b["entropy"]:.3f}{kl_b}  '
              f'lr={lr_a:.1e}  ({elapsed:.1f}s)')

        rec = {
            'iter': it, 'n_eps': len(eps),
            'mean_r_a': float(mean_r_a) if eps else None,
            'mean_r_b': float(mean_r_b) if eps else None,
            'rolling_a': float(rolling_a) if recent_a else None,
            'rolling_b': float(rolling_b) if recent_b else None,
            'train_pos_a': float(pos_a),
            'train_pos_b': float(pos_b),
            'entropy_a': metrics_a.get('entropy'),
            'entropy_b': metrics_b.get('entropy'),
            'lr': lr_a,
        }

        if it % EVAL_EVERY == 0:
            ev_a, ev_b = evaluate_multi(agent_a, agent_b,
                                        n_episodes=EVAL_EPISODES,
                                        seed_base=99999)
            print(f'  >> EVAL  '
                  f'A: R={ev_a["mean_reward"]:7.2f}±{ev_a["std_reward"]:.2f} '
                  f'|pos|={ev_a["mean_abs_pos"]:.0f}  '
                  f'B: R={ev_b["mean_reward"]:7.2f}±{ev_b["std_reward"]:.2f} '
                  f'|pos|={ev_b["mean_abs_pos"]:.0f}')

            rec['eval_a'] = ev_a
            rec['eval_b'] = ev_b

            combined_eval = ev_a['mean_reward'] + ev_b['mean_reward']
            if combined_eval > (best_eval_a + best_eval_b):
                best_eval_a = ev_a['mean_reward']
                best_eval_b = ev_b['mean_reward']
                agent_a.save(CHECKPOINT_A)
                agent_b.save(CHECKPOINT_B)
                print(f'  >> new best: A={best_eval_a:.2f} B={best_eval_b:.2f}')

        history.append(rec)
        HISTORY_FILE.write_text(json.dumps(history, indent=2))

    total_time = (time.time() - t_start) / 60

    # Summary
    print('\n' + '=' * 70)
    print('  IPPO TRAINING SUMMARY')
    print('=' * 70)
    print(f'  Total iterations: {n_iters}')
    print(f'  Total time:       {total_time:.1f} min')
    print(f'  Best eval:  A={best_eval_a:.2f}  B={best_eval_b:.2f}')
    print('=' * 70)

    # Plot curves
    plot_mappo_curves(history, CURVES_PNG)

    # Load best checkpoints and record episodes
    agent_a.load(CHECKPOINT_A)
    agent_b.load(CHECKPOINT_B)
    record_final_episodes(agent_a, agent_b)


if __name__ == '__main__':
    main()
