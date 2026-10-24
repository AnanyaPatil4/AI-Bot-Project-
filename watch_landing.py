
from pathlib import Path
import time
import gymnasium as gym
from stable_baselines3 import PPO

# ==========================================
# 1. FIND THE PROJECT FOLDER
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent

print("Project folder:", PROJECT_DIR)

# Look for the updated model first
possible_models = [
    PROJECT_DIR / "ppo_lunlarlander_v2.zip",
    PROJECT_DIR / "ppo_lunarlander_v2.zip",
    PROJECT_DIR / "ppo_lunlarlander_v2",
    PROJECT_DIR / "ppo_lunarlander_v2",
    PROJECT_DIR / "ppo_lunlarlander.zip",
    PROJECT_DIR / "ppo_lunarlander.zip",
]

MODEL_PATH = next(
    (path for path in possible_models if path.is_file()),
    None
)

# ==========================================
# 2. CHECK THE MODEL FILE
# ==========================================
print("\nZIP/model files found:")

for file in PROJECT_DIR.glob("*.zip"):
    print(" -", file.name)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "\nNo recognized PPO model was found in:\n"
        f"{PROJECT_DIR}\n\n"
        "Check the printed filenames and update "
        "possible_models with the exact model name."
    )

print("\nLoading model:", MODEL_PATH.name)

model = PPO.load(str(MODEL_PATH))

print("PPO model loaded successfully!")

# ==========================================
# 3. OPEN THE VISIBLE LUNARLANDER GAME
# ==========================================
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

N_EPISODES = 5

try:
    # ======================================
    # 4. WATCH THE TRAINED AGENT
    # ======================================
    for episode in range(1, N_EPISODES + 1):

        obs, info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        print(f"\n{'=' * 40}")
        print(f"Episode {episode}/{N_EPISODES}")
        print("AI is attempting to land...")

        while not (terminated or truncated):

            # Use the trained policy
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)
            steps += 1

        # ==================================
        # 5. REPORT THE FINAL RESULT
        # ==================================
        left_contact = bool(obs[6])
        right_contact = bool(obs[7])

        print(f"Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        print(f"Final X position: {obs[0]:.3f}")
        print(f"Final Y position: {obs[1]:.3f}")
        print(f"Horizontal velocity: {obs[2]:.3f}")
        print(f"Vertical velocity: {obs[3]:.3f}")
        print(f"Final angle: {obs[4]:.3f} radians")
        print(f"Left leg contact: {left_contact}")
        print(f"Right leg contact: {right_contact}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        if truncated:
            print("Result: Episode timed out; landing not confirmed.")

        elif terminated and total_reward >= 200:
            print("Result: Reward benchmark reached.")

        elif terminated:
            print("Result: Episode ended; inspect reward and contacts.")

        # Pause before starting the next attempt
        time.sleep(3)

finally:
    env.close()
    print("\nVisual evaluation finished.")
