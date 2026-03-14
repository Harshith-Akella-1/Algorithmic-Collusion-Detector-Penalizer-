import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

# 1. THE ENVIRONMENT
class BertrandCollusionEnv(gym.Env):
    def __init__(self, max_price=100, marginal_cost=20):
        super(BertrandCollusionEnv, self).__init__()
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        self.action_space = spaces.Box(low=marginal_cost, high=max_price, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=marginal_cost, high=max_price, shape=(2,), dtype=np.float32)
        self.state = np.array([max_price, max_price], dtype=np.float32)

    def step(self, actions):
        p1, p2 = actions[0], actions[1]
        if p1 < p2:
            reward1, reward2 = (p1 - self.marginal_cost), 0
        elif p2 < p1:
            reward1, reward2 = 0, (p2 - self.marginal_cost)
        else:
            reward1 = reward2 = (p1 - self.marginal_cost) * 0.5
            
        self.state = np.array([p1, p2], dtype=np.float32)
        return self.state, (reward1, reward2), False, False, {}

    def reset(self, seed=None, options=None):
        self.state = np.array([self.max_price, self.max_price], dtype=np.float32)
        return self.state, {}

# 2. THE SIMULATION LOGIC
def run_simulation(agent_type="competitive", steps=100):
    env = BertrandCollusionEnv()
    obs, _ = env.reset()
    history = []

    for _ in range(steps):
        if agent_type == "collusive":
            actions = [80.0, 80.0] # High fixed price
        else:
            p1, p2 = obs
            actions = [max(21, p1 - 1), max(21, p2 - 1)] # Undercutting logic
        
        obs, rewards, _, _, _ = env.step(actions)
        history.append((actions[0], actions[1], rewards[0], rewards[1]))
    
    return np.array(history)

# 3. THE VISUALIZATION
if __name__ == "__main__":
    print("Running simulations...")
    comp_history = run_simulation("competitive")
    coll_history = run_simulation("collusive")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(comp_history[:, 0], label="Competitive Price", color='blue', linestyle='--')
    plt.plot(coll_history[:, 0], label="Collusive Price", color='red')
    plt.axhline(y=20, color='black', label="Marginal Cost")
    plt.title("Price Evolution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(comp_history[:, 2].cumsum(), label="Comp Profit", color='blue', linestyle='--')
    plt.plot(coll_history[:, 2].cumsum(), label="Coll Profit", color='red')
    plt.title("Cumulative Profit")
    plt.legend()

    plt.tight_layout()
    plt.savefig("collusion_evidence.png")
    print("Success. 'collusion_evidence.png' generated.")