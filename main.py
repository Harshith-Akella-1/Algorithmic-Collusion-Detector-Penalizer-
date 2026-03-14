import numpy as np
from environment import BertrandCollusionEnv

def run_simulation(agent_type="competitive", steps=50):
    env = BertrandCollusionEnv()
    obs, _ = env.reset()
    history = []

    for _ in range(steps):
        if agent_type == "collusive":
            # Both agents agree to stay at a high price
            actions = [80.0, 80.0]
        else:
            # Simple Rule: Undercut the opponent by 1 unit until Marginal Cost (20)
            p1, p2 = obs
            actions = [max(20, p1 - 1), max(20, p2 - 1)]
        
        obs, rewards, _, _, _ = env.step(actions)
        history.append((actions[0], actions[1], rewards[0], rewards[1]))
    
    return np.array(history)

if __name__ == "__main__":
    # This part runs if you execute main.py directly
    print("Running Competitive Simulation...")
    comp_data = run_simulation("competitive")
    print(f"Final Competitive Prices: {comp_data[-1, :2]}")