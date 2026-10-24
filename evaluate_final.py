
from pathlib import Path
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ==========================================
# 1. LOAD THE TRAINED MODEL
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent

possible_models = [
    PROJECT_DIR / "ppo_lunlarlander_v2.zip",
    PROJECT_DIR / "ppo_lunarlander_v2.zip",
    PROJECT_DIR / "ppo_lunlarlander.zip",
    PROJECT_DIR / "ppo_lunarlander.zip",
]

MODEL_PATH = next(
    (p for p in possible_models if p.is_file()), None
)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Trained PPO model not found. Check the ZIP filenames."
    )

model = PPO.load(str(MODEL_PATH))
print(f"Loaded model: {MODEL_PATH.name}")

# ==========================================
# 2. EVALUATE 100 EPISODES
# ==========================================
N_EPISODES = 100
env = gym.make("LunarLander-v3")
results = []

try:
    for episode in range(1, N_EPISODES + 1):
        obs, info = env.reset(seed=2000 + episode)

        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(
                obs, deterministic=True
            )
            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)
            steps += 1

        # Final state
        left_contact = bool(obs[6])
        right_contact = bool(obs[7])
        x_velocity = float(obs[2])
        y_velocity = float(obs[3])
        angle = float(obs[4])

        # Strict, transparent operational success criterion.
        # This is a project-defined criterion, not an official
        # Gymnasium success flag.
        successful_landing = (
            terminated
            and not truncated
            and left_contact
            and right_contact
            and abs(x_velocity) < 0.1
            and abs(y_velocity) < 0.1
            and abs(angle) < 0.2
        )

        results.append({
            "episode": episode,
            "total_reward": round(total_reward, 3),
            "steps": steps,
            "final_x": round(float(obs[0]), 4),
            "final_y": round(float(obs[1]), 4),
            "final_x_velocity": round(x_velocity, 4),
            "final_y_velocity": round(y_velocity, 4),
            "final_angle_rad": round(angle, 4),
            "left_leg_contact": left_contact,
            "right_leg_contact": right_contact,
            "terminated": terminated,
            "truncated": truncated,
            "successful_landing": successful_landing
        })

        print(
            f"Episode {episode:3d}/{N_EPISODES} | "
            f"Reward: {total_reward:8.2f} | "
            f"Success: {successful_landing}"
        )

finally:
    env.close()

# ==========================================
# 3. SAVE RESULTS AND SUMMARY
# ==========================================
df = pd.DataFrame(results)
df.to_csv(PROJECT_DIR / "ppo_final_evaluation.csv", index=False)

success_count = int(df["successful_landing"].sum())
success_rate = 100 * df["successful_landing"].mean()

summary = pd.DataFrame([{
    "episodes": len(df),
    "mean_reward": df["total_reward"].mean(),
    "median_reward": df["total_reward"].median(),
    "reward_std": df["total_reward"].std(),
    "best_reward": df["total_reward"].max(),
    "worst_reward": df["total_reward"].min(),
    "mean_steps": df["steps"].mean(),
    "successful_landings": success_count,
    "landing_success_rate_percent": success_rate
}])

summary.to_csv(
    PROJECT_DIR / "ppo_final_summary.csv", index=False
)

print("\n========== FINAL PPO RESULTS ==========")
print(f"Episodes evaluated: {len(df)}")
print(f"Mean reward: {df['total_reward'].mean():.2f}")
print(f"Reward standard deviation: {df['total_reward'].std():.2f}")
print(f"Mean episode length: {df['steps'].mean():.2f}")
print(f"Successful landings: {success_count}/{len(df)}")
print(f"Landing success rate: {success_rate:.1f}%")

# ==========================================
# 4. PLOT PPO REWARDS
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"])
plt.axhline(200, linestyle="--", label="Reward 200 benchmark")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO Agent: Reward Across 100 Episodes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    PROJECT_DIR / "ppo_final_rewards.png", dpi=150
)
plt.show()

# ==========================================
# 5. COMPARE AGAINST RANDOM BASELINE
# ==========================================
baseline_path = PROJECT_DIR / "baseline_readings.csv"

if baseline_path.is_file():
    baseline = pd.read_csv(baseline_path)

    comparison = pd.DataFrame({
        "controller": ["Random baseline", "Trained PPO"],
        "mean_reward": [
            baseline["total_reward"].mean(),
            df["total_reward"].mean()
        ]
    })

    comparison.to_csv(
        PROJECT_DIR / "baseline_vs_ppo.csv", index=False
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        comparison["controller"],
        comparison["mean_reward"]
    )
    plt.ylabel("Mean Episode Reward")
    plt.title("Random Baseline vs Trained PPO")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(
        PROJECT_DIR / "baseline_vs_ppo.png", dpi=150
    )
    plt.show()

    print("\nBaseline mean reward:",
          f"{baseline['total_reward'].mean():.2f}")
    print("PPO mean reward:",
          f"{df['total_reward'].mean():.2f}")
    print("Comparison saved to baseline_vs_ppo.csv/png")

print("\nSaved evaluation CSV, summary, and graphs.")
