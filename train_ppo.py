
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Create the environment
env = gym.make("LunarLander-v3")
env = Monitor(env)

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    seed=42
)

# Train the agent
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_lunarlander")

env.close()

print("Training completed.")
print("Model saved as ppo_lunarlander.zip")
