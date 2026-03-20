import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BertrandCollusionEnv(gym.Env):
    def __init__(self, max_price=100.0, marginal_cost=20.0):
        super().__init__()
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        
        # FIX 1: Change shape to (2,) so the model outputs two prices
        self.action_space = spaces.Box(low=marginal_cost, high=max_price, shape=(2,), dtype=np.float32)
        
        # Observation remains the same
        self.observation_space = spaces.Box(low=marginal_cost, high=max_price, shape=(2,), dtype=np.float32)
        self.state = np.array([max_price, max_price], dtype=np.float32)

    def step(self, actions):
        # Now actions[0] and actions[1] will both exist
        p1, p2 = float(actions[0]), float(actions[1])
        c = self.marginal_cost
        
        if p1 < p2:
            reward1, reward2 = (p1 - c), 0.0
        elif p2 < p1:
            reward1, reward2 = 0.0, (p2 - c)
        else:
            reward1 = reward2 = (p1 - c) * 0.5
            
        self.state = np.array([p1, p2], dtype=np.float32)
        
        # Keep individual logging for your Detector ("The Sheriff")
        with open("market_tape.csv", "a") as f:
            f.write(f"{p1},{p2},{reward1},{reward2}\n")
            
        # FIX 2: Return the SUM of rewards so SB3 doesn't crash
        total_reward = float(reward1 + reward2)
            
        return self.state, total_reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.max_price, self.max_price], dtype=np.float32)
        return self.state, {}