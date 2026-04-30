"""
simulator.py -- unified collusion simulator.

Provides:
    LOB           : limit order book with price-time priority
    NoiseTrader   : random trader (drift-biased, partial cancel)
    MarketMaker   : passive symmetric quoter
    ColluderPair  : pair of traders with one of 4 collusion modes:
                    {wash, paint, spoof, mirror, none}
    run_episode() : runs one 600s episode, returns (orders, trades, label)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass

DT = 0.1
DURATION = 600


# =========================================================================
# LOB
# =========================================================================
@dataclass
class Order:
    oid: int; trader: str; side: str
    price: float; qty: int; ts: float; remaining: int


class LOB:
    def __init__(self):
        self.bids: list[Order] = []
        self.asks: list[Order] = []
        self.by_id: dict[int, Order] = {}
        self.next_oid = 1
        self.trades: list[dict] = []
        self.order_log: list[dict] = []

    def best_bid(self): return self.bids[0].price if self.bids else None
    def best_ask(self): return self.asks[0].price if self.asks else None

    def mid(self):
        b, a = self.best_bid(), self.best_ask()
        return (b + a) / 2 if b is not None and a is not None else None

    def submit(self, trader, side, otype, qty, ts, price=None):
        oid = self.next_oid; self.next_oid += 1
        o = Order(oid, trader, side,
                  price if price is not None else float('nan'),
                  qty, ts, qty)
        self.order_log.append({
            'ts': ts, 'oid': oid, 'trader': trader, 'side': side,
            'type': otype, 'price': o.price, 'qty': qty,
        })
        self._match(o, ts, is_market=(otype == 'market'))
        if otype == 'limit' and o.remaining > 0:
            book = self.bids if side == 'buy' else self.asks
            book.append(o)
            book.sort(key=(lambda x: (-x.price, x.ts)) if side == 'buy'
                      else (lambda x: (x.price, x.ts)))
            self.by_id[oid] = o
        return oid

    def cancel(self, oid, ts):
        o = self.by_id.pop(oid, None)
        if o is None:
            return False
        (self.bids if o.side == 'buy' else self.asks).remove(o)
        self.order_log.append({
            'ts': ts, 'oid': o.oid, 'trader': o.trader,
            'side': o.side, 'type': 'cancel',
            'price': o.price, 'qty': o.remaining,
        })
        return True

    def _match(self, inc, ts, is_market):
        opp = self.asks if inc.side == 'buy' else self.bids
        while opp and inc.remaining > 0:
            top = opp[0]
            if not is_market:
                if inc.side == 'buy'  and top.price > inc.price: break
                if inc.side == 'sell' and top.price < inc.price: break
            qty = min(inc.remaining, top.remaining)
            buyer, seller = ((inc.trader, top.trader) if inc.side == 'buy'
                             else (top.trader, inc.trader))
            self.trades.append({
                'ts': ts, 'buyer': buyer, 'seller': seller,
                'price': top.price, 'qty': qty,
                'aggressor': inc.side,
                'mid_at_trade': self.mid(),
            })
            inc.remaining -= qty; top.remaining -= qty
            if top.remaining == 0:
                self.by_id.pop(top.oid, None); opp.pop(0)


# =========================================================================
# Bots
# =========================================================================
class NoiseTrader:
    def __init__(self, name, rate=0.6, p_aggressive=0.3,
                 spread_dev=0.10, p_cancel=0.45):
        self.name = name
        self.rate = rate
        self.p_aggressive = p_aggressive
        self.spread_dev = spread_dev
        self.p_cancel = p_cancel
        self.active_orders: list[int] = []

    def act(self, lob, t, anchor, drift, rng):
        if rng.random() > self.rate * DT:
            return
        # Lazy prune stale OIDs, then maybe cancel one
        if self.active_orders and rng.random() < self.p_cancel:
            self.active_orders = [o for o in self.active_orders if o in lob.by_id]
            if self.active_orders:
                idx = int(rng.integers(0, len(self.active_orders)))
                lob.cancel(self.active_orders.pop(idx), t)
        bias = 0.5 + 0.2 * np.sign(drift)
        side = 'buy' if rng.random() < bias else 'sell'
        qty = int(rng.integers(10, 100))
        if rng.random() < self.p_aggressive:
            lob.submit(self.name, side, 'market', qty, t)
        else:
            off = rng.exponential(self.spread_dev)
            price = round(anchor - off if side == 'buy' else anchor + off, 2)
            oid = lob.submit(self.name, side, 'limit', qty, t, price)
            self.active_orders.append(oid)


class MarketMaker:
    def __init__(self, name='MM', hs=0.05, qty=80, refresh=3.0):
        self.name = name; self.hs = hs; self.qty = qty; self.refresh = refresh
        self.last = -1e9; self.bid_id = self.ask_id = None

    def act(self, lob, t, anchor):
        if t - self.last < self.refresh:
            return
        for attr in ('bid_id', 'ask_id'):
            oid = getattr(self, attr)
            if oid is not None:
                lob.cancel(oid, t); setattr(self, attr, None)
        self.bid_id = lob.submit(self.name, 'buy', 'limit',
                                 self.qty, t, round(anchor - self.hs, 2))
        self.ask_id = lob.submit(self.name, 'sell', 'limit',
                                 self.qty, t, round(anchor + self.hs, 2))
        self.last = t


# =========================================================================
# ColluderPair: 4 modes (wash, paint, spoof, mirror) + none
# =========================================================================
class ColluderPair:
    """
    A and B always run as NoiseTrader instances (cover trades).
    Inside the scheduled window, mode-specific scheme actions run on top.
    Scheme OIDs live in self.state and never enter active_orders, so
    noise-cancel logic cannot accidentally touch them.
    """
    def __init__(self, A, B, mode='none'):
        self.A = NoiseTrader(A, rate=0.4, p_aggressive=0.25)
        self.B = NoiseTrader(B, rate=0.4, p_aggressive=0.25)
        self.mode = mode
        self.window = None
        self.state = {}

    def schedule(self, t0, t1, rng):
        self.window = (t0, t1)
        if self.mode == 'wash':
            self.state = {'next_wash': t0}
        elif self.mode == 'paint':
            self.state = {'phase': 'idle', 'next_t': t0,
                          'post_side': 'buy', 'post_oid': None,
                          'post_qty': 0}
        elif self.mode == 'spoof':
            self.state = {'phase': 'idle', 'next_t': t0,
                          'spoof_oids': [], 'spoof_side': 'buy'}
        elif self.mode == 'mirror':
            self.state = {'phase': 'idle', 'next_t': t0,
                          'mirror_side': 'buy', 'A_oids': [], 'B_oids': [],
                          'sync_lag': 0.0, 'B_build_delay': 0.0}

    def step(self, lob, t, anchor, drift, rng):
        # Always cover trades, regardless of mode
        self.A.act(lob, t, anchor, drift, rng)
        self.B.act(lob, t, anchor, drift, rng)
        if self.mode == 'none' or self.window is None:
            return
        if not (self.window[0] <= t <= self.window[1]):
            return
        if self.mode == 'wash':
            self._wash(lob, t, anchor, rng)
        elif self.mode == 'paint':
            self._paint(lob, t, anchor, rng)
        elif self.mode == 'spoof':
            self._spoof(lob, t, anchor, rng)
        elif self.mode == 'mirror':
            self._mirror(lob, t, anchor, rng)

    # ----- wash -----
    def _wash(self, lob, t, anchor, rng):
        nxt = self.state.get('next_wash')
        if nxt is not None and t >= nxt:
            p = round(anchor + rng.normal(0, 0.01), 2)
            q = int(rng.integers(50, 150))
            lob.submit(self.A.name, 'sell', 'limit', q, t, p)
            lob.submit(self.B.name, 'buy', 'market', q, t + 0.05)
            self.state['next_wash'] = t + rng.uniform(8, 20)

    # ----- paint -----
    def _paint(self, lob, t, anchor, rng):
        phase = self.state['phase']; next_t = self.state['next_t']
        if phase == 'idle' and t >= next_t:
            self.state.update(phase='post', next_t=t)
        elif phase == 'post' and t >= next_t:
            ps = self.state['post_side']
            qty = int(rng.integers(10, 99))
            bb, ba = lob.best_bid(), lob.best_ask()
            mid = lob.mid() or anchor
            if bb is not None and ba is not None and (ba - bb) > 0.02:
                price = round(bb + 0.01 if ps == 'buy' else ba - 0.01, 2)
            else:
                jit = abs(rng.normal(0, 0.005))
                price = round(mid - jit if ps == 'buy' else mid + jit, 2)
            oid = lob.submit(self.A.name, ps, 'limit', qty, t, price)
            self.state.update(phase='cross', post_oid=oid, post_qty=qty,
                              next_t=t + float(rng.uniform(0.05, 0.20)))
        elif phase == 'cross' and t >= next_t:
            ps = self.state['post_side']
            cs = 'sell' if ps == 'buy' else 'buy'
            cb_qty = self.state.get('post_qty', int(rng.integers(10, 99)))
            lob.submit(self.B.name, cs, 'market', cb_qty, t)
            oid = self.state.get('post_oid')
            if oid is not None and oid in lob.by_id:
                lob.cancel(oid, t)
            self.state.update(phase='idle',
                              post_side='sell' if ps == 'buy' else 'buy',
                              post_oid=None, post_qty=0,
                              next_t=t + float(rng.uniform(15, 35)))

    # ----- spoof -----
    def _spoof(self, lob, t, anchor, rng):
        phase = self.state['phase']; next_t = self.state['next_t']
        if phase == 'idle' and t >= next_t:
            spoof_side = self.state['spoof_side']
            n_layers = int(rng.integers(2, 6))
            spoof_oids = []
            for i in range(n_layers):
                qty = int(rng.integers(20, 120))
                lvl = rng.uniform(0.01, 0.05) * (i + 1) + rng.normal(0, 0.004)
                price = (round(anchor - abs(lvl), 2) if spoof_side == 'buy'
                         else round(anchor + abs(lvl), 2))
                oid = lob.submit(self.A.name, spoof_side, 'limit', qty, t, price)
                spoof_oids.append(oid)
            self.state.update(phase='spoofed', spoof_oids=spoof_oids,
                              next_t=t + float(rng.uniform(0.05, 0.25)))
        elif phase == 'spoofed' and t >= next_t:
            exec_side = 'sell' if self.state['spoof_side'] == 'buy' else 'buy'
            exec_qty = int(rng.integers(30, 150))
            lob.submit(self.B.name, exec_side, 'market', exec_qty, t)
            self.state.update(phase='executing',
                              next_t=t + float(rng.uniform(0.05, 0.35)))
        elif phase == 'executing' and t >= next_t:
            live = [o for o in self.state['spoof_oids'] if o in lob.by_id]
            cancel_frac = float(rng.uniform(0.60, 1.00))
            n_to_cancel = max(1, round(cancel_frac * len(live)))
            rng.shuffle(live)
            for oid in live[:n_to_cancel]:
                lob.cancel(oid, t)
            next_side = 'sell' if self.state['spoof_side'] == 'buy' else 'buy'
            self.state.update(phase='idle', spoof_oids=[],
                              spoof_side=next_side,
                              next_t=t + float(rng.uniform(10, 30)))

    # ----- mirror -----
    def _mirror(self, lob, t, anchor, rng):
        phase = self.state['phase']; next_t = self.state['next_t']
        TICK = 0.02
        if phase == 'idle' and t >= next_t:
            self.state['B_build_delay'] = float(rng.uniform(0.0, 0.5))
            self.state['sync_lag'] = float(rng.uniform(0.1, 1.0))
            self.state['phase'] = 'build_A'
            self.state['next_t'] = t
        elif phase == 'build_A' and t >= next_t:
            side = self.state['mirror_side']
            n_layers = int(rng.integers(3, 6))
            A_oids = []
            for i in range(n_layers):
                qty = int(rng.integers(30, 80))
                off = TICK * (i + 1) + rng.uniform(-TICK / 2, TICK / 2)
                price = (round(anchor - off, 2) if side == 'buy'
                         else round(anchor + off, 2))
                A_oids.append(lob.submit(self.A.name, side, 'limit', qty, t, price))
            self.state.update(A_oids=A_oids, phase='build_B',
                              next_t=t + self.state['B_build_delay'])
        elif phase == 'build_B' and t >= next_t:
            side = self.state['mirror_side']
            n_layers = int(rng.integers(3, 6))
            B_oids = []
            for i in range(n_layers):
                qty = int(rng.integers(30, 80))
                off = TICK * (i + 1) + TICK / 2 + rng.uniform(-TICK / 2, TICK / 2)
                price = (round(anchor - off, 2) if side == 'buy'
                         else round(anchor + off, 2))
                B_oids.append(lob.submit(self.B.name, side, 'limit', qty, t, price))
            self.state.update(B_oids=B_oids, phase='hold',
                              next_t=t + float(rng.uniform(5, 15)))
        elif phase == 'hold' and t >= next_t:
            self.state.update(phase='cancel_A', next_t=t)
        elif phase == 'cancel_A' and t >= next_t:
            for oid in [o for o in self.state['A_oids'] if o in lob.by_id]:
                lob.cancel(oid, t)
            self.state.update(A_oids=[], phase='cancel_B',
                              next_t=t + self.state['sync_lag'])
        elif phase == 'cancel_B' and t >= next_t:
            for oid in [o for o in self.state['B_oids'] if o in lob.by_id]:
                lob.cancel(oid, t)
            next_side = 'sell' if self.state['mirror_side'] == 'buy' else 'buy'
            self.state.update(B_oids=[], mirror_side=next_side,
                              phase='idle', next_t=t + float(rng.uniform(20, 40)))


# =========================================================================
# Episode runner
# =========================================================================
COLLUSION_TYPES = ['none', 'wash', 'paint', 'spoof', 'mirror']

def run_episode(seed, collusion_type='none', n_noise=15):
    rng = np.random.default_rng(seed)
    lob = LOB()
    lob.submit('seed', 'buy', 'limit', 100, 0.0, 99.95)
    lob.submit('seed', 'sell', 'limit', 100, 0.0, 100.05)

    noise = [NoiseTrader(f'N{i}') for i in range(n_noise)]
    mm = MarketMaker()
    pair = ColluderPair('CA', 'CB', mode=collusion_type)

    label = {'collusion_type': collusion_type, 'pair': None,
             't_start': None, 't_end': None}
    if collusion_type != 'none':
        # Per-mode duration ranges
        if collusion_type == 'wash':
            scheme_dur = float(rng.uniform(60, 180))
        else:
            scheme_dur = float(rng.uniform(120, 300))
        max_t0 = DURATION - scheme_dur - 10
        t0 = float(rng.uniform(0.1 * DURATION, max_t0))
        t1 = t0 + scheme_dur
        pair.schedule(t0, t1, rng)
        label = {'collusion_type': collusion_type, 'pair': ['CA', 'CB'],
                 't_start': t0, 't_end': t1}

    anchor = 100.0; history = [100.0]; LOOKBACK = 50
    t = 0.0
    while t < DURATION:
        anchor += rng.normal(0, 0.02)
        history.append(anchor)
        drift = anchor - history[max(0, len(history) - LOOKBACK)]
        for n in noise:
            n.act(lob, t, anchor, drift, rng)
        mm.act(lob, t, anchor)
        pair.step(lob, t, anchor, drift, rng)
        t = round(t + DT, 6)
    return (pd.DataFrame(lob.order_log),
            pd.DataFrame(lob.trades), label)


if __name__ == '__main__':
    print('Smoke test: one episode of each type')
    for ct in COLLUSION_TYPES:
        orders, trades, label = run_episode(seed=42, collusion_type=ct)
        print(f'  {ct:8s} | orders={len(orders):5d}  trades={len(trades):5d}  '
              f'window={label["t_start"]} -> {label["t_end"]}')
