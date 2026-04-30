"""
ppo.py -- Custom PyTorch PPO for discrete action spaces.

Standard clipped-surrogate PPO with Phase 2 improvements:
  - Shared trunk (2 hidden layers), separate actor and critic heads
  - Generalized Advantage Estimation (GAE), lambda=0.95
  - Clipped objective, eps=0.2
  - Clipped value function (prevents value overshooting)
  - Entropy bonus to encourage exploration
  - Gradient clipping at norm=0.5
  - Multiple SGD passes per rollout (4 epochs default)
  - Running observation normalization (online mean/var)
  - Linear LR schedule support
  - KL-based early stopping per update iteration
  - Gradient norm logging for diagnostics

Public API:
    PPO(obs_dim, n_actions, **kwargs)
    .collect_rollout(env, n_steps) -> RolloutBuffer
    .update(buffer)               -> dict of training metrics
    .act(obs)                     -> action (deterministic=False uses sampling)
    .set_lr(frac_remaining)       -> adjust LR linearly
    .save(path) / .load(path)
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Running mean/std for observation normalization
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self, shape, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # avoid div-by-zero
        self.clip = clip

    def update(self, x):
        """x: (batch, shape) or (shape,)"""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        new_var = m2 / total
        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x):
        """Normalize observation, return as float32."""
        normed = (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + 1e-8
        )
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


# ---------------------------------------------------------------------------
# Network: shared trunk, actor + critic heads
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    obs: torch.Tensor          # (T, obs_dim)
    actions: torch.Tensor      # (T,)
    log_probs: torch.Tensor    # (T,) old log_probs
    rewards: torch.Tensor      # (T,)
    values: torch.Tensor       # (T,) old value estimates
    dones: torch.Tensor        # (T,) 1.0 if episode terminated AT step t
    advantages: torch.Tensor   # (T,) computed in-place after rollout
    returns: torch.Tensor      # (T,) advantages + values

    def __len__(self):
        return self.obs.shape[0]


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------
class PPO:
    def __init__(self, obs_dim, n_actions,
                 lr=3e-4,
                 lr_end=5e-5,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 grad_clip=0.5,
                 update_epochs=4,
                 minibatch_size=64,
                 hidden=128,
                 target_kl=0.02,
                 normalize_obs=True,
                 device='cpu'):
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, n_actions, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.lr_start = lr
        self.lr_end = lr_end
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.n_actions = n_actions
        self.target_kl = target_kl

        # Observation normalization
        self.normalize_obs = normalize_obs
        self.obs_rms = RunningMeanStd(shape=(obs_dim,)) if normalize_obs else None

    # ------------------------------------------------------------------
    def set_lr(self, frac_remaining):
        """Set learning rate based on linear schedule. frac_remaining goes 1 -> 0."""
        lr = self.lr_end + frac_remaining * (self.lr_start - self.lr_end)
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr

    def current_lr(self):
        return self.opt.param_groups[0]['lr']

    # ------------------------------------------------------------------
    def _normalize_obs(self, obs_np):
        """Normalize a single observation (numpy). Updates running stats."""
        if self.obs_rms is not None:
            self.obs_rms.update(obs_np)
            return self.obs_rms.normalize(obs_np)
        return obs_np

    def _normalize_obs_batch(self, obs_np, update=False):
        """Normalize a batch of observations (numpy). Optionally update stats."""
        if self.obs_rms is not None:
            if update:
                self.obs_rms.update(obs_np)
            return self.obs_rms.normalize(obs_np)
        return obs_np

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """obs: np.ndarray (obs_dim,). Returns (action_int, log_prob, value)."""
        # Normalize but don't update stats during act (already updated in collect)
        if self.obs_rms is not None:
            obs_normed = self.obs_rms.normalize(obs)
        else:
            obs_normed = obs
        x = torch.from_numpy(obs_normed).float().unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        dist = Categorical(logits=logits)
        if deterministic:
            a = logits.argmax(-1)
        else:
            a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    # ------------------------------------------------------------------
    def collect_rollout(self, env, n_steps):
        """Run env for n_steps, accumulating transitions. Resets when episode ends."""
        obs_dim = env.observation_space.shape[0]
        obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros(n_steps, dtype=np.int64)
        logp_buf = np.zeros(n_steps, dtype=np.float32)
        rew_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)

        if not hasattr(self, '_env_obs') or self._env_obs is None:
            self._env_obs, _ = env.reset()
            self._ep_reward = 0.0
            self._ep_len = 0
            self._completed_episodes = []

        obs = self._env_obs

        # Collect raw obs for batch normalization update
        raw_obs_batch = np.zeros((n_steps, obs_dim), dtype=np.float32)

        for t in range(n_steps):
            raw_obs_batch[t] = obs

            # Update running stats and normalize
            if self.obs_rms is not None:
                self.obs_rms.update(obs)
                obs_normed = self.obs_rms.normalize(obs)
            else:
                obs_normed = obs

            action, logp, value = self.act(obs)  # act uses obs_rms.normalize internally
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_buf[t] = obs_normed
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = reward
            val_buf[t] = value
            done_buf[t] = 1.0 if done else 0.0

            self._ep_reward += reward
            self._ep_len += 1

            if done:
                self._completed_episodes.append({
                    'reward': self._ep_reward,
                    'length': self._ep_len,
                    'mark': info.get('mark', 0.0),
                    'position': info.get('position', 0),
                    'realized_pnl': info.get('realized_pnl', 0.0),
                    'invalid_count': info.get('invalid_count', 0),
                    'fills_taken': info.get('fills_taken', 0),
                    'passive_fills': info.get('passive_fills', 0),
                })
                obs, _ = env.reset()
                self._ep_reward = 0.0
                self._ep_len = 0
            else:
                obs = next_obs
        self._env_obs = obs

        # Bootstrap value for final state
        with torch.no_grad():
            if self.obs_rms is not None:
                obs_normed = self.obs_rms.normalize(obs)
            else:
                obs_normed = obs
            x = torch.from_numpy(obs_normed).float().unsqueeze(0).to(self.device)
            _, last_value = self.net(x)
            last_value = float(last_value.item())

        # GAE
        adv = np.zeros(n_steps, dtype=np.float32)
        ret = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n_steps)):
            next_val = last_value if t == n_steps - 1 else val_buf[t + 1]
            next_non_terminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.gamma * next_val * next_non_terminal - val_buf[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            adv[t] = gae
            ret[t] = adv[t] + val_buf[t]

        # Normalize advantages
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return RolloutBuffer(
            obs=torch.from_numpy(obs_buf).to(self.device),
            actions=torch.from_numpy(act_buf).to(self.device),
            log_probs=torch.from_numpy(logp_buf).to(self.device),
            rewards=torch.from_numpy(rew_buf).to(self.device),
            values=torch.from_numpy(val_buf).to(self.device),
            dones=torch.from_numpy(done_buf).to(self.device),
            advantages=torch.from_numpy(adv).to(self.device),
            returns=torch.from_numpy(ret).to(self.device),
        )

    def pop_episodes(self):
        """Return and clear the list of completed episodes."""
        eps = getattr(self, '_completed_episodes', [])
        self._completed_episodes = []
        return eps

    # ------------------------------------------------------------------
    def update(self, buf):
        n = len(buf)
        idx = np.arange(n)

        metrics = {
            'policy_loss': 0.0, 'value_loss': 0.0,
            'entropy': 0.0, 'approx_kl': 0.0,
            'clip_frac': 0.0, 'grad_norm': 0.0,
            'updates': 0, 'early_stopped': False,
        }

        for epoch in range(self.update_epochs):
            np.random.shuffle(idx)
            epoch_kl_accum = 0.0
            epoch_updates = 0

            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                if len(mb) < 4:
                    continue
                mb_t = torch.from_numpy(mb).long().to(self.device)
                obs = buf.obs[mb_t]
                actions = buf.actions[mb_t]
                old_logp = buf.log_probs[mb_t]
                advantages = buf.advantages[mb_t]
                returns = buf.returns[mb_t]
                old_values = buf.values[mb_t]

                logits, values = self.net(obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp)

                # Clipped policy loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss: prevent value function from overshooting
                values_clipped = old_values + torch.clamp(
                    values - old_values, -self.clip_eps, self.clip_eps
                )
                value_loss_unclipped = (values - returns) ** 2
                value_loss_clipped = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

                with torch.no_grad():
                    log_ratio = new_logp - old_logp
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl
                metrics['clip_frac'] += clip_frac
                metrics['grad_norm'] += float(grad_norm)
                metrics['updates'] += 1

                epoch_kl_accum += approx_kl
                epoch_updates += 1

            # KL-based early stopping: if mean KL this epoch > target, stop
            if epoch_updates > 0:
                epoch_mean_kl = epoch_kl_accum / epoch_updates
                if epoch_mean_kl > self.target_kl:
                    metrics['early_stopped'] = True
                    break

        if metrics['updates'] > 0:
            for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl',
                       'clip_frac', 'grad_norm'):
                metrics[k] /= metrics['updates']
        return metrics

    # ------------------------------------------------------------------
    def save(self, path):
        state = {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
        }
        if self.obs_rms is not None:
            state['obs_rms'] = self.obs_rms.state_dict()
        torch.save(state, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck['net'])
        self.opt.load_state_dict(ck['opt'])
        if self.obs_rms is not None and 'obs_rms' in ck:
            self.obs_rms.load_state_dict(ck['obs_rms'])
