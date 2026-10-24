
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Evaluation settings
N_EPISODES = 100
MODEL_PATH = "ppo_lunarlander.zip"

# Load the trained PPO model
model = PPO.load(MODEL_PATH)

# Create a fresh environment (no rendering for faster evaluation)
env = gym.make("LunarLander-v3")

results = []

for episode in range(1, N_EPISODES + 1):
    obs, info = env.reset(seed=1000 + episode)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Deterministic actions for consistent evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    results.append({
        "episode": episode,
        "total_reward": round(total_reward, 2),
        "steps": steps,
        "final_x_velocity": round(float(obs[2]), 4),
        "final_y_velocity": round(float(obs[3]), 4),
        "final_angle": round(float(obs[4]), 4),
        "left_leg_contact": int(obs[6]),
        "right_leg_contact": int(obs[7]),
        "terminated": terminated,
        "truncated": truncated,
        # Conventional reward-based benchmark, not a direct
        # physical landing detector
        "reward_200_threshold": total_reward >= 200
    })

    print(
        f"Episode {episode}/{N_EPISODES} | "
        f"Reward: {total_reward:.2f} | Steps: {steps}"
    )

env.close()

# Save episode-by-episode results
df = pd.DataFrame(results)
df.to_csv("ppo_evaluation.csv", index=False)

# Print summary
print("\n========== PPO EVALUATION ==========")
print(f"Episodes evaluated: {len(df)}")
print(f"Mean reward: {df['total_reward'].mean():.2f}")
print(f"Reward standard deviation: {df['total_reward'].std():.2f}")
print(f"Best reward: {df['total_reward'].max():.2f}")
print(f"Worst reward: {df['total_reward'].min():.2f}")
print(f"Mean episode length: {df['steps'].mean():.2f}")
print(
    "Episodes with reward >= 200: "
    f"{df['reward_200_threshold'].sum()}/{len(df)} "
    f"({100 * df['reward_200_threshold'].mean():.1f}%)"
)

# Save a summary file
summary = pd.DataFrame([{
    "episodes": len(df),
    "mean_reward": df["total_reward"].mean(),
    "reward_std": df["total_reward"].std(),
    "best_reward": df["total_reward"].max(),
    "worst_reward": df["total_reward"].min(),
    "mean_steps": df["steps"].mean(),
    "episodes_reward_ge_200": int(df["reward_200_threshold"].sum()),
    "reward_ge_200_percent": 100 * df["reward_200_threshold"].mean()
}])
summary.to_csv("ppo_evaluation_summary.csv", index=False)

# Plot reward per episode
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"])
plt.axhline(200, linestyle="--", label="Reward = 200 benchmark")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO: Total Reward per Evaluation Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ppo_evaluation_rewards.png", dpi=150)
plt.show()

print("\nSaved ppo_evaluation.csv")
print("Saved ppo_evaluation_summary.csv")
print("Saved ppo_evaluation_rewards.png")
