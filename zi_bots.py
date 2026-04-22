import numpy as np

class ZeroIntelligenceMarket:
    def __init__(self, num_makers=40, num_takers=10, initial_price=100.0, tick_size=0.1):
        self.num_makers = num_makers
        self.num_takers = num_takers
        self.fundamental_price = initial_price
        self.tick_size = tick_size
        
        # Volatility of the random walk
        self.volatility = 0.5 
        # How far from the fundamental price the makers place their limit orders
        self.spread_half_width = 0.5 
        
        # Reserve IDs 1 and 2 for the RL Bots (the colluders)
        # Give ZI bots high IDs so we can easily filter their trades later
        self.maker_ids = np.arange(10, 10 + num_makers)
        self.taker_ids = np.arange(100, 100 + num_takers)

    def generate_orders(self):
        """
        Generates a batch of limit and market orders for the current timestep.
        Returns a list of tuples: (agent_id, price, quantity, is_buy)
        """
        # 1. Update the fundamental value using a Gaussian random walk
        self.fundamental_price += np.random.normal(0, self.volatility)
        
        # Prevent the asset from going to zero or negative
        self.fundamental_price = max(10.0, self.fundamental_price)
        
        orders = []
        
        # 2. Vectorized Maker Orders (Liquidity Providers)
        half_m = self.num_makers // 2
        
        # Bids (Buys): Distributed below fundamental
        bids = np.random.normal(self.fundamental_price - self.spread_half_width, 1.0, half_m)
        bids = np.round(bids / self.tick_size) * self.tick_size
        
        # Asks (Sells): Distributed above fundamental
        asks = np.random.normal(self.fundamental_price + self.spread_half_width, 1.0, half_m)
        asks = np.round(asks / self.tick_size) * self.tick_size
        
        # Standard quantity for background liquidity
        maker_qty = 10 
        
        for i in range(half_m):
            orders.append((int(self.maker_ids[i]), float(bids[i]), maker_qty, True))
            orders.append((int(self.maker_ids[half_m + i]), float(asks[i]), maker_qty, False))
            
        # 3. Vectorized Taker Orders (Impatient Retail Volume)
        # Takers submit aggressive limit orders designed to cross the spread instantly
        # We simulate a "market order" by setting an artificially high bid or low ask
        half_t = self.num_takers // 2
        taker_qty = 5
        
        for i in range(half_t):
            # Aggressive Buy
            high_bid = self.fundamental_price * 1.5
            orders.append((int(self.taker_ids[i]), float(high_bid), taker_qty, True))
            
            # Aggressive Sell
            low_ask = self.fundamental_price * 0.5
            orders.append((int(self.taker_ids[half_t + i]), float(low_ask), taker_qty, False))
            
        # 4. Shuffle the order flow so makers and takers hit the engine randomly
        np.random.shuffle(orders)
        
        return orders
    
    