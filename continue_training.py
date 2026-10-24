
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

PROJECT_DIR = (
    r"C:\Users\amitp\Documents\AI project Bot"
    r"\lunar lander type-a"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "ppo_lunlarlander_v2.zip"
)

OUTPUT_PATH = os.path.join(
    PROJECT_DIR, "ppo_lunlarlander_v3"
)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Existing model not found: {MODEL_PATH}"
    )

# Create the same environment
env = Monitor(gym.make("LunarLander-v3"))

try:
    # Load existing trained model
    model = PPO.load(MODEL_PATH, env=env)

    print("Existing model loaded.")
    print("Continuing training for 200,000 timesteps...")

    model.learn(
        total_timesteps=200_000,
        reset_num_timesteps=False
    )

    # Save to a new file; preserve the previous model
    model.save(OUTPUT_PATH)

    print("Additional training completed.")
    print(f"Updated model saved to: {OUTPUT_PATH}.zip")

finally:
    env.close()
