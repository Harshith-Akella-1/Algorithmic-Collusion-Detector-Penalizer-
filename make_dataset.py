"""
End-to-end starter for collusion dataset generation.
Run: python make_dataset.py
Outputs: data/ep_XXXX/{orders.parquet, trades.parquet, label.json}
"""
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

rng = np.random.default_rng()
DT = 0.1                # tick size (seconds)
DURATION = 600          # episode length; bump to 3600 for real runs

# =================== LOB ===================
@dataclass
class Order:
    oid: int; trader: str; side: str
    price: float; qty: int; ts: float; remaining: int

class LOB:
    def __init__(self):
        self.bids: list[Order] = []     # desc price, asc ts
        self.asks: list[Order] = []     # asc price, asc ts
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
        o = Order(oid, trader, side, price if price is not None else float('nan'),
                  qty, ts, qty)
        self.order_log.append({'ts': ts, 'oid': oid, 'trader': trader, 'side': side,
                               'type': otype, 'price': o.price, 'qty': qty})
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
        if o is None: return False
        (self.bids if o.side == 'buy' else self.asks).remove(o)
        self.order_log.append({'ts': ts, 'oid': oid, 'trader': o.trader,
            'side': o.side, 'type': 'cancel', 'price': o.price, 'qty': o.remaining})
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
            self.trades.append({'ts': ts, 'buyer': buyer, 'seller': seller,
                'price': top.price, 'qty': qty, 'aggressor': inc.side})
            inc.remaining -= qty; top.remaining -= qty
            if top.remaining == 0:
                self.by_id.pop(top.oid, None); opp.pop(0)

# =================== Bots ===================
class NoiseTrader:
    """Random limit orders around mid. Same logic the colluders use as cover."""
    def __init__(self, name, rate=2.5):
        self.name = name; self.rate = rate
    def act(self, lob, t):
        if rng.random() > self.rate * DT: return
        mid = lob.mid() or 100.0
        side = rng.choice(['buy', 'sell'])
        off = rng.exponential(0.05)
        price = round(mid - off if side == 'buy' else mid + off, 2)
        lob.submit(self.name, side, 'limit', int(rng.integers(10, 100)), t, price)

class MarketMaker:
    def __init__(self, name, hs=0.05, qty=80, refresh=3.0):
        self.name = name; self.hs = hs; self.qty = qty; self.refresh = refresh
        self.last = -1e9; self.bid_id = self.ask_id = None
    def act(self, lob, t):
        if t - self.last < self.refresh: return
        for attr in ('bid_id', 'ask_id'):
            oid = getattr(self, attr)
            if oid is not None: lob.cancel(oid, t); setattr(self, attr, None)
        mid = lob.mid() or 100.0
        self.bid_id = lob.submit(self.name, 'buy',  'limit', self.qty, t, round(mid - self.hs, 2))
        self.ask_id = lob.submit(self.name, 'sell', 'limit', self.qty, t, round(mid + self.hs, 2))
        self.last = t

class WashColluderPair:
    """A and B act as noise traders. Inside a scheduled window they also wash trade."""
    def __init__(self, A, B):
        self.A = NoiseTrader(A, rate=0.3)
        self.B = NoiseTrader(B, rate=0.3)
        self.window = None; self.next_wash = None
    def schedule(self, t0, t1):
        self.window = (t0, t1); self.next_wash = t0
    def step(self, lob, t):
        self.A.act(lob, t); self.B.act(lob, t)        # cover trades
        if self.window and self.window[0] <= t <= self.window[1]:
            if self.next_wash is not None and t >= self.next_wash:
                mid = lob.mid() or 100.0
                p = round(mid + rng.normal(0, 0.01), 2)
                q = int(rng.integers(50, 150))
                lob.submit(self.A.name, 'sell', 'limit', q, t, p)
                lob.submit(self.B.name, 'buy',  'market', q, t + 0.05)
                self.next_wash = t + rng.uniform(8, 20)

# =================== Episode runner ===================
def run_episode(seed, collusion_type=None):
    global rng
    rng = np.random.default_rng(seed)
    lob = LOB()
    # seed the book so MM has a mid to quote around
    lob.submit('seed', 'buy',  'limit', 100, 0.0, 99.95)
    lob.submit('seed', 'sell', 'limit', 100, 0.0, 100.05)

    noise = [NoiseTrader(f'N{i}') for i in range(15)]
    mm = MarketMaker('MM')
    pair = WashColluderPair('CA', 'CB')

    label = {'collusion_type': 'none', 'pair': None, 't_start': None, 't_end': None}
    if collusion_type == 'wash':
        t0 = float(rng.uniform(0.2 * DURATION, 0.6 * DURATION))
        t1 = t0 + float(rng.uniform(60, 180))
        pair.schedule(t0, t1)
        label = {'collusion_type': 'wash', 'pair': ['CA', 'CB'],
                 't_start': t0, 't_end': t1}

    t = 0.0
    while t < DURATION:
        for n in noise: n.act(lob, t)
        mm.act(lob, t)
        pair.step(lob, t)
        t += DT
    return pd.DataFrame(lob.order_log), pd.DataFrame(lob.trades), label

def generate_dataset(out_dir='data', n_per_class=10, seed0=0):
    out = Path(out_dir); out.mkdir(exist_ok=True)
    types = [None, 'wash']
    i = 0
    for ct in types:
        for _ in range(n_per_class):
            orders, trades, label = run_episode(seed0 + i, collusion_type=ct)
            ed = out / f'ep_{i:04d}'; ed.mkdir(exist_ok=True)
            orders.to_parquet(ed / 'orders.parquet')
            trades.to_parquet(ed / 'trades.parquet')
            (ed / 'label.json').write_text(json.dumps(label, indent=2))
            i += 1
    print(f'Generated {i} episodes in {out}/')

# =================== Smoke test ===================
if __name__ == '__main__':
    orders, trades, label = run_episode(seed=42, collusion_type='wash')
    print(f'Orders logged : {len(orders)}')
    print(f'Trades logged : {len(trades)}')
    print(f'Label         : {label}')
    if len(trades) and label['t_start'] is not None:
        in_win = trades[(trades['ts'] >= label['t_start']) &
                        (trades['ts'] <= label['t_end'])]
        ca_cb = in_win[((in_win['buyer'] == 'CA') & (in_win['seller'] == 'CB')) |
                       ((in_win['buyer'] == 'CB') & (in_win['seller'] == 'CA'))]
        print(f'Trades in collusion window      : {len(in_win)}')
        print(f'CA<->CB trades in window        : {len(ca_cb)}')
        print(f'CA<->CB trades outside window   : '
              f'{((((trades["buyer"]=="CA")&(trades["seller"]=="CB"))|((trades["buyer"]=="CB")&(trades["seller"]=="CA"))).sum()) - len(ca_cb)}')
    print('\nGenerating mini dataset (20 episodes)...')
    generate_dataset(n_per_class=10)

