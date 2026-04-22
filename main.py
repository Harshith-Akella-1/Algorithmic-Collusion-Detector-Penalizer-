import supersuit as ss
from stable_baselines3 import PPO
from multi_agent_env import CollusionSandbox

def train_colluders():
    # 1. Initialize the Environment
    env = CollusionSandbox(tick_size=0.1)

    # 2. Add "Memory" via Frame Stacking (Crucial for detecting Tit-for-Tat)
    # This stacks the last 4 observations so agents see price trends
    env = ss.frame_stack_v1(env, 4)

    # 3. Convert PettingZoo Parallel Env to a Vectorized Gym Env
    # This allows SB3 to treat the multi-agent env as a batch of single-agent envs
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    # 4. Initialize PPO with Collusion-friendly Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,          # High gamma = agents care about future long-term profit
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # Encourage exploration to find collusive states
        tensorboard_log="./collusion_logs/"
    )

    # 5. Training Loop
    print("Starting training... Aiming for 1 million steps.")
    model.learn(total_timesteps=1_000_000)

    # 6. Save the trained agents
    model.save("collusion_ppo_model")
    print("Model saved. Ready for 'Sheriff' testing.")

if __name__ == "__main__":
    train_colluders()