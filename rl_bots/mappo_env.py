"""
mappo_env.py -- Multi-agent Gymnasium environment for two RL traders.

Two independent RL agents (RL_A and RL_B) trade in the same LOB alongside
13 noise traders and 1 market maker. Each agent has its own observation
(position, cash, mid, spread, depth, returns, time, resting order, PnL)
and does NOT see the other agent's state.

Design:
  - Agents act simultaneously at each step
  - Same action space as single-agent MarketEnv (7 discrete actions)
  - Same observation space (12 floats per agent)
  - Independent rewards (PnL-based, no collusion incentive)
  - Optional episode recording for detector cross-validation

Usage:
    env = MultiAgentMarketEnv(seed=42)
    (obs_a, obs_b), info = env.reset()
    (obs_a, obs_b), (r_a, r_b), done, trunc, info = env.step(action_a, action_b)
"""
import math
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Fix import: simulator.py lives in Simulator/
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_simulator_dir = os.path.join(_project_root, 'Simulator')
if _simulator_dir not in sys.path:
    sys.path.insert(0, _simulator_dir)

from simulator import LOB, NoiseTrader, MarketMaker, DT, DURATION

# Import constants from single-agent env
from market_env import (
    ACT_INTERVAL, LOT_SIZE, MAX_POSITION_HARD, MAX_POSITION_NORM,
    CASH_SCALE, PRICE_SCALE, INV_PENALTY, INVALID_PENALTY,
    EOD_PENALTY, PASSIVE_BONUS, REWARD_SCALE, ANCHOR_INIT,
    URGENCY_START, URGENCY_MAX_MULT, _inventory_penalty
)

N_NOISE = 13  # 2 agents take 2 slots, down from 14+1


class AgentState:
    """Per-agent state tracking."""
    def __init__(self, name):
        self.name = name
        self.position = 0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.avg_entry = 0.0
        self.resting_oid = None
        self.resting_side = None
        self.last_mark = 0.0
        self.invalid_count = 0
        self.fills_taken = 0
        self.passive_fills = 0


class MultiAgentMarketEnv(gym.Env):
    """Two-agent LOB environment for IPPO training."""

    metadata = {'render_modes': []}

    def __init__(self, seed=None, record_episodes=False, record_dir=None):
        super().__init__()
        self._seed = seed
        self.record_episodes = record_episodes
        self.record_dir = Path(record_dir) if record_dir else None

        # Gym spaces (per agent)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(12,), dtype=np.float32
        )

        self._episode_count = 0
        self._reset_state()

    def _reset_state(self):
        seed = self._seed if self._seed is not None else np.random.randint(1 << 31)
        self.rng = np.random.default_rng(seed)

        self.lob = LOB()
        self.lob.submit('seed', 'buy', 'limit', 100, 0.0, 99.95)
        self.lob.submit('seed', 'sell', 'limit', 100, 0.0, 100.05)

        self.noise = [NoiseTrader(f'N{i}') for i in range(N_NOISE)]
        self.mm = MarketMaker()

        self.t = 0.0
        self.anchor = ANCHOR_INIT
        self.history = [ANCHOR_INIT]

        # Two agents
        self.agent_a = AgentState('RL_A')
        self.agent_b = AgentState('RL_B')
        self.agents = [self.agent_a, self.agent_b]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._reset_state()
        # Warmup
        self._advance(50)
        for ag in self.agents:
            ag.last_mark = self._mark_to_market(ag)
        self._episode_count += 1
        obs_a = self._observation(self.agent_a)
        obs_b = self._observation(self.agent_b)
        return (obs_a, obs_b), {}

    def _advance(self, n_ticks):
        """Advance LOB + noise + MM by n_ticks."""
        for _ in range(n_ticks):
            if self.t >= DURATION:
                break
            self.anchor += self.rng.normal(0, 0.02)
            self.history.append(self.anchor)
            drift = self.anchor - self.history[max(0, len(self.history) - 50)]
            for n in self.noise:
                n.act(self.lob, self.t, self.anchor, drift, self.rng)
            self.mm.act(self.lob, self.t, self.anchor)
            # Record fills for both agents during advance
            for ag in self.agents:
                self._record_fills_for(ag)
            self.t = round(self.t + DT, 6)

    def _record_fills_for(self, agent):
        """Record fills for a specific agent."""
        for tr in self.lob.trades:
            if tr.get('_seen_' + agent.name):
                continue
            if tr['buyer'] == agent.name or tr['seller'] == agent.name:
                qty = tr['qty']
                px = tr['price']
                if tr['buyer'] == agent.name:
                    new_pos = agent.position + qty
                    if agent.position >= 0:
                        if new_pos > 0:
                            agent.avg_entry = (agent.avg_entry * agent.position
                                               + px * qty) / new_pos
                    else:
                        closing = min(qty, -agent.position)
                        agent.realized_pnl += (agent.avg_entry - px) * closing
                        if new_pos > 0:
                            agent.avg_entry = px
                    agent.cash -= qty * px
                    agent.position = new_pos
                else:
                    new_pos = agent.position - qty
                    if agent.position <= 0:
                        if new_pos < 0:
                            denom = -new_pos
                            agent.avg_entry = (agent.avg_entry * (-agent.position)
                                               + px * qty) / denom
                    else:
                        closing = min(qty, agent.position)
                        agent.realized_pnl += (px - agent.avg_entry) * closing
                        if new_pos < 0:
                            agent.avg_entry = px
                    agent.cash += qty * px
                    agent.position = new_pos
                agent.fills_taken += qty

                # Passive fill tracking
                if tr.get('aggressor') == 'buy' and tr['seller'] == agent.name:
                    agent.passive_fills += qty
                elif tr.get('aggressor') == 'sell' and tr['buyer'] == agent.name:
                    agent.passive_fills += qty

            tr['_seen_' + agent.name] = True

    def _mid(self):
        m = self.lob.mid()
        return m if m is not None else self.anchor

    def _mark_to_market(self, agent):
        return agent.cash + agent.position * self._mid()

    def _depth_top5(self, side):
        book = self.lob.bids if side == 'buy' else self.lob.asks
        if not book:
            return 0
        levels_seen = []
        total = 0
        for o in book:
            if o.price not in levels_seen:
                if len(levels_seen) >= 5:
                    break
                levels_seen.append(o.price)
            total += o.remaining
        return total

    def _observation(self, agent):
        """Build per-agent observation (12 floats). Agent doesn't see the other."""
        mid = self._mid()
        bb = self.lob.best_bid()
        ba = self.lob.best_ask()
        spread = (ba - bb) if (bb is not None and ba is not None) else 0.10

        if len(self.history) >= 11:
            ret_1s = math.log(self.history[-1] / self.history[-11])
        else:
            ret_1s = 0.0
        if len(self.history) >= 101:
            ret_10s = math.log(self.history[-1] / self.history[-101])
        else:
            ret_10s = 0.0

        bid_d = self._depth_top5('buy')
        ask_d = self._depth_top5('sell')

        unrealized = (agent.position * (mid - agent.avg_entry)
                      if agent.position != 0 else 0.0)

        obs = np.array([
            agent.position / MAX_POSITION_NORM,
            agent.cash / CASH_SCALE,
            (mid - ANCHOR_INIT) / PRICE_SCALE,
            spread / PRICE_SCALE,
            math.log1p(bid_d) / 8.0,
            math.log1p(ask_d) / 8.0,
            ret_1s * 100.0,
            ret_10s * 100.0,
            (DURATION - self.t) / DURATION,
            1.0 if (agent.resting_oid is not None and
                    agent.resting_oid in self.lob.by_id) else 0.0,
            unrealized / CASH_SCALE,
            agent.realized_pnl / CASH_SCALE,
        ], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    def _cancel_resting(self, agent):
        if agent.resting_oid is not None and agent.resting_oid in self.lob.by_id:
            self.lob.cancel(agent.resting_oid, self.t)
        agent.resting_oid = None
        agent.resting_side = None

    def _would_breach_position(self, agent, side):
        if side == 'buy':
            return (agent.position + LOT_SIZE) > MAX_POSITION_HARD
        else:
            return (agent.position - LOT_SIZE) < -MAX_POSITION_HARD

    def _apply_action(self, agent, action):
        """Apply discrete action for one agent. Returns (valid, passive_rested)."""
        bb = self.lob.best_bid()
        ba = self.lob.best_ask()

        if action == 0:  # HOLD
            return True, False

        self._cancel_resting(agent)

        if action == 1:  # BUY_MARKET
            if ba is None or self._would_breach_position(agent, 'buy'):
                return False, False
            self.lob.submit(agent.name, 'buy', 'market', LOT_SIZE, self.t)
            return True, False

        if action == 2:  # SELL_MARKET
            if bb is None or self._would_breach_position(agent, 'sell'):
                return False, False
            self.lob.submit(agent.name, 'sell', 'market', LOT_SIZE, self.t)
            return True, False

        if action == 3:  # BUY_PASSIVE
            if bb is None or self._would_breach_position(agent, 'buy'):
                return False, False
            oid = self.lob.submit(agent.name, 'buy', 'limit', LOT_SIZE, self.t, bb)
            rested = oid in self.lob.by_id
            if rested:
                agent.resting_oid = oid
                agent.resting_side = 'buy'
            return True, rested

        if action == 4:  # SELL_PASSIVE
            if ba is None or self._would_breach_position(agent, 'sell'):
                return False, False
            oid = self.lob.submit(agent.name, 'sell', 'limit', LOT_SIZE, self.t, ba)
            rested = oid in self.lob.by_id
            if rested:
                agent.resting_oid = oid
                agent.resting_side = 'sell'
            return True, rested

        if action == 5:  # BUY_AGGRESSIVE
            if ba is None or self._would_breach_position(agent, 'buy'):
                return False, False
            self.lob.submit(agent.name, 'buy', 'limit', LOT_SIZE, self.t, ba)
            return True, False

        if action == 6:  # SELL_AGGRESSIVE
            if bb is None or self._would_breach_position(agent, 'sell'):
                return False, False
            self.lob.submit(agent.name, 'sell', 'limit', LOT_SIZE, self.t, bb)
            return True, False

        return False, False

    def step(self, action_a, action_b):
        """Step both agents simultaneously.

        Returns:
            (obs_a, obs_b), (reward_a, reward_b), terminated, truncated, info
        """
        # Apply both actions
        valid_a, passive_a = self._apply_action(self.agent_a, int(action_a))
        valid_b, passive_b = self._apply_action(self.agent_b, int(action_b))

        if not valid_a:
            self.agent_a.invalid_count += 1
        if not valid_b:
            self.agent_b.invalid_count += 1

        # Process immediate fills
        for ag in self.agents:
            self._record_fills_for(ag)

        # Advance time
        self._advance(ACT_INTERVAL)

        time_remaining = (DURATION - self.t) / DURATION
        terminated = self.t >= DURATION

        # Compute rewards independently
        rewards = []
        for ag, valid, passive in [(self.agent_a, valid_a, passive_a),
                                   (self.agent_b, valid_b, passive_b)]:
            mark = self._mark_to_market(ag)
            delta = mark - ag.last_mark
            ag.last_mark = mark

            reward = delta
            reward -= _inventory_penalty(ag.position, time_remaining)
            if not valid:
                reward -= INVALID_PENALTY
            if passive:
                reward += PASSIVE_BONUS
            if terminated:
                reward -= EOD_PENALTY * abs(ag.position)

            rewards.append(float(reward * REWARD_SCALE))

        # Record episode data if requested
        if terminated and self.record_episodes and self.record_dir:
            self._save_episode()

        obs_a = self._observation(self.agent_a)
        obs_b = self._observation(self.agent_b)

        info = {
            'agent_a': {
                'mark': self._mark_to_market(self.agent_a),
                'position': self.agent_a.position,
                'cash': self.agent_a.cash,
                'realized_pnl': self.agent_a.realized_pnl,
                'invalid_count': self.agent_a.invalid_count,
                'fills_taken': self.agent_a.fills_taken,
                'passive_fills': self.agent_a.passive_fills,
            },
            'agent_b': {
                'mark': self._mark_to_market(self.agent_b),
                'position': self.agent_b.position,
                'cash': self.agent_b.cash,
                'realized_pnl': self.agent_b.realized_pnl,
                'invalid_count': self.agent_b.invalid_count,
                'fills_taken': self.agent_b.fills_taken,
                'passive_fills': self.agent_b.passive_fills,
            },
        }

        return ((obs_a, obs_b), tuple(rewards),
                bool(terminated), False, info)

    def _save_episode(self):
        """Save episode orders and trades to parquet for detector inference."""
        ep_dir = self.record_dir / f'ep_{self._episode_count:03d}'
        ep_dir.mkdir(parents=True, exist_ok=True)

        orders_df = pd.DataFrame(self.lob.order_log)
        trades_list = []
        for tr in self.lob.trades:
            trades_list.append({
                'ts': tr['ts'], 'buyer': tr['buyer'], 'seller': tr['seller'],
                'price': tr['price'], 'qty': tr['qty'],
                'aggressor': tr['aggressor'],
                'mid_at_trade': tr.get('mid_at_trade'),
            })
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        orders_df.to_parquet(ep_dir / 'orders.parquet', index=False)
        if len(trades_df) > 0:
            trades_df.to_parquet(ep_dir / 'trades.parquet', index=False)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def random_baseline(n_episodes=3, seed_base=0):
    """Run random-action baseline for both agents."""
    rng = np.random.default_rng(seed_base)
    print('Multi-Agent Random Baseline:')
    print(f'  N_NOISE={N_NOISE}, agents=RL_A+RL_B')
    print()

    for ep in range(n_episodes):
        env = MultiAgentMarketEnv(seed=seed_base + ep)
        (obs_a, obs_b), _ = env.reset()
        done = False
        ep_r_a, ep_r_b = 0.0, 0.0
        n_steps = 0

        while not done:
            a_a = rng.integers(0, 7)
            a_b = rng.integers(0, 7)
            (obs_a, obs_b), (r_a, r_b), terminated, truncated, info = env.step(a_a, a_b)
            ep_r_a += r_a
            ep_r_b += r_b
            n_steps += 1
            done = terminated or truncated

        ia = info['agent_a']
        ib = info['agent_b']
        print(f'  ep {ep}: steps={n_steps}')
        print(f'    A: R={ep_r_a:8.2f}  mark={ia["mark"]:8.2f}  '
              f'pos={ia["position"]:+5d}  fills={ia["fills_taken"]:4d}  '
              f'invalid={ia["invalid_count"]:3d}')
        print(f'    B: R={ep_r_b:8.2f}  mark={ib["mark"]:8.2f}  '
              f'pos={ib["position"]:+5d}  fills={ib["fills_taken"]:4d}  '
              f'invalid={ib["invalid_count"]:3d}')


if __name__ == '__main__':
    random_baseline()
