"""
market_env.py -- Gymnasium environment for one RL trader in the LOB simulator.

The agent acts every ACT_INTERVAL ticks (default 10 ticks = 1 second of sim
time). Between actions, the LOB and noise traders run normally. The agent's
only goal is to maximize PnL — there is no collusion signal in the reward.

Observation (12 floats, all bounded roughly to [-5, 5] after normalization):
    0  position_norm        position / MAX_POSITION_NORM
    1  cash_norm             cash / CASH_SCALE
    2  mid_norm              (mid - anchor_init) / PRICE_SCALE
    3  spread_norm           spread / PRICE_SCALE
    4  bid_depth_norm        log1p(top-5 bid qty) / 8
    5  ask_depth_norm        log1p(top-5 ask qty) / 8
    6  ret_1s                mid log-return over last 10 ticks
    7  ret_10s               mid log-return over last 100 ticks
    8  time_remaining        (DURATION - t) / DURATION  in [0, 1]
    9  has_resting_limit     0 or 1
    10 unrealized_pnl_norm   position * (mid - avg_entry) / CASH_SCALE
    11 realized_pnl_norm     realized PnL / CASH_SCALE

Action (7 discrete):
    0  HOLD
    1  BUY_MARKET             market buy LOT_SIZE
    2  SELL_MARKET             market sell LOT_SIZE
    3  BUY_PASSIVE            limit buy at best_bid (joins the queue)
    4  SELL_PASSIVE            limit sell at best_ask
    5  BUY_AGGRESSIVE         limit buy at best_ask (crosses)
    6  SELL_AGGRESSIVE        limit sell at best_bid

A new order from the agent first cancels its prior resting limit (if any).
This keeps the agent's book state tractable for the policy network.

Reward (per agent step):
    mark_t = cash + position * mid
    reward = (mark_t - mark_{t-1})
             - inventory_penalty(position, time_remaining)
             - INVALID_PENALTY  (if action invalid)
             + PASSIVE_BONUS    (if passive limit rests in book)
             + at episode end: -EOD_PENALTY * |position|

Phase 2 improvements over smoke-test version:
  1. Hard position cap at ±MAX_POSITION_HARD — rejects orders that breach
  2. Passive limit bonus — biases exploration toward spread capture
  3. Time-decay inventory urgency — ramps penalty in last 20% of episode
  4. Asymmetric inventory penalty — cubic ramp for extreme positions

Reset: spawns LOB, 14 noise traders, 1 market maker, 1 RL agent slot.
Episode ends when the simulation clock reaches DURATION.
"""
import math
import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Fix import: simulator.py lives in Simulator/ one level up from rl_bots/
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_simulator_dir = os.path.join(_project_root, 'Simulator')
if _simulator_dir not in sys.path:
    sys.path.insert(0, _simulator_dir)

from simulator import LOB, NoiseTrader, MarketMaker, DT, DURATION

# ---- Constants ----
N_NOISE = 14                    # one fewer than baseline; agent takes the slot
ACT_INTERVAL = 10               # ticks between agent actions (1 second)
LOT_SIZE = 50

# Position limits
MAX_POSITION_HARD = 500         # hard cap: orders that would breach are rejected
MAX_POSITION_NORM = 1000        # for observation normalization only

# Reward / observation scaling
CASH_SCALE = 10_000.0
PRICE_SCALE = 1.0               # anchor moves on the order of ±1 over an episode
INV_PENALTY = 5e-6              # per step, base penalty coefficient
INVALID_PENALTY = 0.05          # invalid / no-op-able action
EOD_PENALTY = 0.002             # at episode end, * |position|
PASSIVE_BONUS = 0.003           # reward for placing a passive limit that rests
REWARD_SCALE = 0.01             # rewards are dollar-PnL; scale down for stable value-fn

# Time-decay settings
URGENCY_START = 0.80            # urgency kicks in when time_remaining < this
URGENCY_MAX_MULT = 4.0          # max multiplier on inventory penalty at EOD

ANCHOR_INIT = 100.0


def _inventory_penalty(position, time_remaining):
    """
    Asymmetric inventory penalty that ramps faster at extreme positions
    and increases urgency near end-of-day.

    Base: INV_PENALTY * |pos|^2.5 / sqrt(MAX_POSITION_HARD)
    Time decay: multiplied by up to URGENCY_MAX_MULT in the last 20%
    """
    base = INV_PENALTY * (abs(position) ** 2.5) / math.sqrt(MAX_POSITION_HARD)

    # Time-decay urgency: ramp penalty in last 20% of episode
    if time_remaining < URGENCY_START:
        progress = 1.0 - (time_remaining / URGENCY_START)  # 0 -> 1 as time runs out
        multiplier = 1.0 + (URGENCY_MAX_MULT - 1.0) * progress
        base *= multiplier

    return base


class MarketEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None, agent_name='RL', verbose=False):
        super().__init__()
        self.agent_name = agent_name
        self.verbose = verbose
        self._seed = seed

        # Gym spaces
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(12,), dtype=np.float32
        )

        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self):
        seed = self._seed if self._seed is not None else np.random.randint(1 << 31)
        self.rng = np.random.default_rng(seed)

        self.lob = LOB()
        # Seed two-sided book
        self.lob.submit('seed', 'buy', 'limit', 100, 0.0, 99.95)
        self.lob.submit('seed', 'sell', 'limit', 100, 0.0, 100.05)

        self.noise = [NoiseTrader(f'N{i}') for i in range(N_NOISE)]
        self.mm = MarketMaker()

        self.t = 0.0
        self.anchor = ANCHOR_INIT
        self.history = [ANCHOR_INIT]

        # Agent state
        self.position = 0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.avg_entry = 0.0       # for unrealized PnL display
        self.resting_oid = None
        self.resting_side = None   # track side of resting order
        self.last_mark = 0.0
        self.steps_taken = 0
        self.invalid_count = 0
        self.fills_taken = 0       # diagnostics
        self.passive_fills = 0     # fills from resting limits

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._reset_state()
        # Run a brief warmup so the book has depth before the agent acts
        self._advance(50)
        self.last_mark = self._mark_to_market()
        return self._observation(), {}

    # ------------------------------------------------------------------
    def _advance(self, n_ticks):
        """Advance LOB + noise + MM by n_ticks, no agent action."""
        for _ in range(n_ticks):
            if self.t >= DURATION:
                break
            self.anchor += self.rng.normal(0, 0.02)
            self.history.append(self.anchor)
            drift = self.anchor - self.history[max(0, len(self.history) - 50)]
            for n in self.noise:
                n.act(self.lob, self.t, self.anchor, drift, self.rng)
            self.mm.act(self.lob, self.t, self.anchor)
            self._record_fills_for_agent()
            self.t = round(self.t + DT, 6)

    # ------------------------------------------------------------------
    def _record_fills_for_agent(self):
        """Drain new trades from lob.trades involving the agent.

        Updates cash, position, realized PnL.
        We process trades not yet seen using a marker on the trade dict.
        """
        for tr in self.lob.trades:
            if tr.get('_seen'):
                continue
            if tr['buyer'] == self.agent_name or tr['seller'] == self.agent_name:
                qty = tr['qty']
                px = tr['price']
                if tr['buyer'] == self.agent_name:
                    # Agent bought: cash decreases, position increases
                    new_pos = self.position + qty
                    if self.position >= 0:
                        # extending long: blend avg entry
                        if new_pos > 0:
                            self.avg_entry = (self.avg_entry * self.position
                                              + px * qty) / new_pos
                    else:
                        # closing short
                        closing = min(qty, -self.position)
                        self.realized_pnl += (self.avg_entry - px) * closing
                        if new_pos > 0:
                            self.avg_entry = px
                    self.cash -= qty * px
                    self.position = new_pos
                else:
                    # Agent sold
                    new_pos = self.position - qty
                    if self.position <= 0:
                        # extending short
                        if new_pos < 0:
                            denom = -new_pos
                            self.avg_entry = (self.avg_entry * (-self.position)
                                              + px * qty) / denom
                    else:
                        # closing long
                        closing = min(qty, self.position)
                        self.realized_pnl += (px - self.avg_entry) * closing
                        if new_pos < 0:
                            self.avg_entry = px
                    self.cash += qty * px
                    self.position = new_pos
                self.fills_taken += qty

                # Track if this was a passive fill (resting limit got hit)
                # Passive = agent was the resting side, not the aggressor
                if tr.get('aggressor') == 'buy' and tr['seller'] == self.agent_name:
                    self.passive_fills += qty
                elif tr.get('aggressor') == 'sell' and tr['buyer'] == self.agent_name:
                    self.passive_fills += qty

            tr['_seen'] = True

    # ------------------------------------------------------------------
    def _mid(self):
        m = self.lob.mid()
        return m if m is not None else self.anchor

    def _mark_to_market(self):
        return self.cash + self.position * self._mid()

    # ------------------------------------------------------------------
    def _depth_top5(self, side):
        # OrderBook stores bids/asks as list[Order] sorted (see simulator.py)
        # Each entry is an Order with `remaining` attribute.
        book = self.lob.bids if side == 'buy' else self.lob.asks
        # Build per-level totals walking from best price out
        if not book:
            return 0
        # The orders are already sorted by (price-priority, ts)
        # Take top-5 distinct price levels
        levels_seen = []
        total = 0
        for o in book:
            if o.price not in levels_seen:
                if len(levels_seen) >= 5:
                    break
                levels_seen.append(o.price)
            total += o.remaining
        return total

    # ------------------------------------------------------------------
    def _observation(self):
        mid = self._mid()
        bb = self.lob.best_bid()
        ba = self.lob.best_ask()
        spread = (ba - bb) if (bb is not None and ba is not None) else 0.10

        # returns
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

        unrealized = self.position * (mid - self.avg_entry) if self.position != 0 else 0.0

        obs = np.array([
            self.position / MAX_POSITION_NORM,
            self.cash / CASH_SCALE,
            (mid - ANCHOR_INIT) / PRICE_SCALE,
            spread / PRICE_SCALE,
            math.log1p(bid_d) / 8.0,
            math.log1p(ask_d) / 8.0,
            ret_1s * 100.0,           # scale up because returns are tiny
            ret_10s * 100.0,
            (DURATION - self.t) / DURATION,
            1.0 if self.resting_oid in self.lob.by_id else 0.0,
            unrealized / CASH_SCALE,
            self.realized_pnl / CASH_SCALE,
        ], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    # ------------------------------------------------------------------
    def _cancel_resting(self):
        if self.resting_oid is not None and self.resting_oid in self.lob.by_id:
            self.lob.cancel(self.resting_oid, self.t)
        self.resting_oid = None
        self.resting_side = None

    def _would_breach_position(self, side):
        """Check if a trade on `side` would breach the hard position cap."""
        if side == 'buy':
            return (self.position + LOT_SIZE) > MAX_POSITION_HARD
        else:  # sell
            return (self.position - LOT_SIZE) < -MAX_POSITION_HARD

    def _apply_action(self, action):
        """Apply discrete action. Returns (valid: bool, passive_rested: bool)."""
        bb = self.lob.best_bid()
        ba = self.lob.best_ask()

        if action == 0:                                  # HOLD
            return True, False

        # All trade actions cancel any prior resting limit first
        self._cancel_resting()

        if action == 1:                                  # BUY_MARKET
            if ba is None:
                return False, False
            if self._would_breach_position('buy'):
                return False, False
            self.lob.submit(self.agent_name, 'buy', 'market', LOT_SIZE, self.t)
            return True, False

        if action == 2:                                  # SELL_MARKET
            if bb is None:
                return False, False
            if self._would_breach_position('sell'):
                return False, False
            self.lob.submit(self.agent_name, 'sell', 'market', LOT_SIZE, self.t)
            return True, False

        if action == 3:                                  # BUY_PASSIVE
            if bb is None:
                return False, False
            if self._would_breach_position('buy'):
                return False, False
            oid = self.lob.submit(self.agent_name, 'buy', 'limit',
                                  LOT_SIZE, self.t, bb)
            rested = oid in self.lob.by_id  # True if it didn't immediately fill
            if rested:
                self.resting_oid = oid
                self.resting_side = 'buy'
            return True, rested

        if action == 4:                                  # SELL_PASSIVE
            if ba is None:
                return False, False
            if self._would_breach_position('sell'):
                return False, False
            oid = self.lob.submit(self.agent_name, 'sell', 'limit',
                                  LOT_SIZE, self.t, ba)
            rested = oid in self.lob.by_id
            if rested:
                self.resting_oid = oid
                self.resting_side = 'sell'
            return True, rested

        if action == 5:                                  # BUY_AGGRESSIVE (cross)
            if ba is None:
                return False, False
            if self._would_breach_position('buy'):
                return False, False
            self.lob.submit(self.agent_name, 'buy', 'limit',
                            LOT_SIZE, self.t, ba)
            return True, False

        if action == 6:                                  # SELL_AGGRESSIVE (cross)
            if bb is None:
                return False, False
            if self._would_breach_position('sell'):
                return False, False
            self.lob.submit(self.agent_name, 'sell', 'limit',
                            LOT_SIZE, self.t, bb)
            return True, False

        return False, False

    # ------------------------------------------------------------------
    def step(self, action):
        valid, passive_rested = self._apply_action(int(action))
        if not valid:
            self.invalid_count += 1
        # Process any fills from the agent's order itself (immediate-cross)
        self._record_fills_for_agent()

        # Advance time
        self._advance(ACT_INTERVAL)
        self.steps_taken += 1

        # Compute reward
        mark = self._mark_to_market()
        delta = mark - self.last_mark
        self.last_mark = mark

        time_remaining = (DURATION - self.t) / DURATION

        reward = delta
        # Inventory penalty (asymmetric + time-decay)
        reward -= _inventory_penalty(self.position, time_remaining)
        # Invalid action penalty
        if not valid:
            reward -= INVALID_PENALTY
        # Passive limit bonus: encourage resting orders (spread capture)
        if passive_rested:
            reward += PASSIVE_BONUS

        terminated = self.t >= DURATION
        if terminated:
            reward -= EOD_PENALTY * abs(self.position)

        truncated = False
        info = {
            'mark': mark,
            'position': self.position,
            'cash': self.cash,
            'realized_pnl': self.realized_pnl,
            'invalid_count': self.invalid_count,
            'fills_taken': self.fills_taken,
            'passive_fills': self.passive_fills,
        }
        return self._observation(), float(reward * REWARD_SCALE), bool(terminated), truncated, info


# ---------------------------------------------------------------------------
# Smoke test: random-action baseline
# ---------------------------------------------------------------------------
def random_baseline(n_episodes=5, seed_base=0):
    rng = np.random.default_rng(seed_base)
    results = []
    for ep in range(n_episodes):
        env = MarketEnv(seed=seed_base + ep)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        n_steps = 0
        while not done:
            action = rng.integers(0, 7)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            n_steps += 1
            done = terminated or truncated
        results.append({
            'episode': ep,
            'reward': ep_reward,
            'steps': n_steps,
            'final_mark': info['mark'],
            'final_position': info['position'],
            'realized_pnl': info['realized_pnl'],
            'invalid': info['invalid_count'],
            'fills': info['fills_taken'],
            'passive_fills': info['passive_fills'],
        })
        print(f"  ep {ep}: reward={ep_reward:8.2f}  "
              f"mark={info['mark']:8.2f}  "
              f"pos={info['position']:+5d}  "
              f"realized={info['realized_pnl']:7.2f}  "
              f"invalid={info['invalid_count']}  "
              f"fills={info['fills_taken']}  "
              f"passive_fills={info['passive_fills']}  "
              f"steps={n_steps}")
    return results


if __name__ == '__main__':
    print('Random-action baseline (sanity check):')
    print(f'  MAX_POSITION_HARD = {MAX_POSITION_HARD}')
    print(f'  LOT_SIZE = {LOT_SIZE}')
    print(f'  PASSIVE_BONUS = {PASSIVE_BONUS}')
    print(f'  URGENCY_START = {URGENCY_START}')
    print()
    random_baseline(n_episodes=5)
