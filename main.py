from market_env import BertrandCollusionEnv
from stable_baselines3 import PPO
import os

def train_and_simulate(timesteps=10000):
    # Clear old logs
    if os.path.exists("market_tape.csv"):
        os.remove("market_tape.csv")
        
    env = BertrandCollusionEnv()
    
    # We use a single policy to control both agents for this prototype
    # In a real MARL setup, you'd use a multi-agent wrapper.
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Training bots to explore the market...")
    model.learn(total_timesteps=timesteps)
    model.save("pricing_bot_model")
    
    print("Simulation complete. 'market_tape.csv' generated for the Sheriff.")

if __name__ == "__main__":
    train_and_simulate()