
import pandas as pd
import matplotlib.pyplot as plt

# Load baseline readings
df = pd.read_csv("baseline_readings.csv")

# Calculate baseline statistics
rewards = df["total_reward"]
steps = df["steps"]

print("\n========== BASELINE ANALYSIS ==========")
print(f"Number of recorded episodes: {len(df)}")
print(f"Mean reward:                 {rewards.mean():.2f}")
print(f"Best reward:                 {rewards.max():.2f}")
print(f"Worst reward:                {rewards.min():.2f}")
print(f"Reward standard deviation:   {rewards.std():.2f}")
print(f"Mean episode length:         {steps.mean():.2f} steps")
print(f"Longest episode:             {steps.max()} steps")
print(f"Shortest episode:            {steps.min()} steps")

# Optional: final-state summary
print("\nMean final vertical velocity:",
      f"{df['final_y_velocity'].mean():.4f}")

print("Mean final horizontal velocity:",
      f"{df['final_x_velocity'].mean():.4f}")

# Plot reward distribution
plt.figure(figsize=(8, 5))
plt.hist(rewards, bins=20, edgecolor="black")
plt.xlabel("Total Reward")
plt.ylabel("Number of Episodes")
plt.title("Baseline Reward Distribution")
plt.tight_layout()
plt.savefig("baseline_reward_distribution.png", dpi=150)
plt.show()

# Plot rolling mean to visualize reward variability
window = min(20, len(df))
if window > 0:
    rolling_reward = rewards.rolling(window=window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], rolling_reward)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling Mean Reward ({window} episodes)")
    plt.title("Baseline Rolling Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("baseline_rolling_reward.png", dpi=150)
    plt.show()

# Export summary statistics
summary = pd.DataFrame({
    "metric": [
        "episode_count",
        "mean_reward",
        "best_reward",
        "worst_reward",
        "reward_std",
        "mean_steps",
        "min_steps",
        "max_steps"
    ],
    "value": [
        len(df),
        rewards.mean(),
        rewards.max(),
        rewards.min(),
        rewards.std(),
        steps.mean(),
        steps.min(),
        steps.max()
    ]
})

summary.to_csv("baseline_summary.csv", index=False)
print("\nSaved baseline_summary.csv")
print("Saved baseline_reward_distribution.png")
print("Saved baseline_rolling_reward.png")

#negative mean reward (-182.44):the random controller performs 
# poorly on average under LunarLander's built-in reward fucntion 
# High variability (110.23 standard deviation):performance 
# differs substantially between episodes. #Best reqard(+46.34):Random actions
# occaisonally produce a better outcome, but one good episode does not demonstrate reliable landing. 
#variable episode lengths: attempts end after different number of steps, consistent with inconsistent trajectories. 
