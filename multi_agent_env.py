import numpy as np
from pettingzoo import ParallelEnv
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matching_engine_cpp as trading_engine
from zi_bots import ZeroIntelligenceMarket

class CollusionSandbox(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "collusion_sandbox_v0"}

    def __init__(self, render_mode=None, tick_size=0.1):
        self.render_mode = render_mode  # <-- FIX: Add this line
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents[:]
        self.tick_size = tick_size
        self.agent_name_mapping = {"agent_1": 1, "agent_2": 2}
        
        self.action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32) 
            for agent in self.possible_agents
        }

    # FIX: Add these helper methods to satisfy PettingZoo 1.24+
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.engine = trading_engine.MatchingEngine()
        self.zi_market = ZeroIntelligenceMarket(tick_size=self.tick_size)
        
        self.portfolios = {
            "agent_1": {"cash": 1000.0, "inventory": 100},
            "agent_2": {"cash": 1000.0, "inventory": 100}
        }
        
        # Track previous value for the reward function
        self.prev_values = {agent: 1000.0 + (100 * 100.0) for agent in self.agents}
        
        self.timestep = 0
        observations = {
            agent: np.array([self.zi_market.fundamental_price, self.portfolios[agent]["cash"], self.portfolios[agent]["inventory"]], dtype=np.float32)
            for agent in self.agents
        }
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        self.timestep += 1
        
        # 1. Background Market Generates Orders
        zi_orders = self.zi_market.generate_orders()
        for order in zi_orders:
            # order tuple: (agent_id, price, quantity, is_buy)
            self.engine.process_order(order[0], order[1], order[2], order[3])
            
        mid_price = self.zi_market.fundamental_price

        # 2. RL Agents submit their orders based on actions
        action_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2} # Map 0-4 to tick offsets
        
        for agent_name, action in actions.items():
            agent_id = self.agent_name_mapping[agent_name]
            tick_offset = action_map[action]
            
            if tick_offset == 0:
                continue # Hold (No order)
                
            price = mid_price + (tick_offset * self.tick_size)
            
            # Simplified logic: If offsetting down, buy. If offsetting up, sell.
            is_buy = tick_offset < 0 
            qty = 10 
            
            # Process through the C++ engine and get execution reports
            fills = self.engine.process_order(agent_id, price, qty, is_buy)
            
            # Update portfolios based strictly on actual executions
            for fill in fills:
                if fill.buyer_id == agent_id:
                    self.portfolios[agent_name]["cash"] -= (fill.price * fill.quantity)
                    self.portfolios[agent_name]["inventory"] += fill.quantity
                elif fill.seller_id == agent_id:
                    self.portfolios[agent_name]["cash"] += (fill.price * fill.quantity)
                    self.portfolios[agent_name]["inventory"] -= fill.quantity

        # 3. Calculate Rewards (Change in Portfolio Value)
        # Portfolio Value = Cash + (Inventory * Mid Price)
        # Inside the step() function, replace your reward calculation:
        rewards = {}
        for agent in self.agents:
            current_value = self.portfolios[agent]["cash"] + (self.portfolios[agent]["inventory"] * mid_price)
            
            # Calculate the CHANGE in value (Profit/Loss)
            rewards[agent] = float(current_value - self.prev_values[agent])
            
            # Update prev_value for the next step
            self.prev_values[agent] = current_value

        # 4. Generate next observations
        observations = {
            agent: np.array([mid_price, self.portfolios[agent]["cash"], self.portfolios[agent]["inventory"]], dtype=np.float32)
            for agent in self.agents
        }

        # 5. Check termination
        terminations = {agent: self.timestep >= 1000 for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos
    