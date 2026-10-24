
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load baseline data
df = pd.read_csv("baseline_readings.csv")

print("\nFirst 5 episodes:")
print(df.head())

print("\nBaseline statistics:")
print(df[["total_reward", "steps"]].describe())

# Plot 1: Reward per episode
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"])
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Baseline: Total Reward per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_reward.png")
plt.show()

# Plot 2: Episode length
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["steps"])
plt.xlabel("Episode")
plt.ylabel("Number of Steps")
plt.title("Baseline: Episode Length")
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_steps.png")
plt.show()

# Plot 3: Final vertical velocity
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["final_y_velocity"])
plt.xlabel("Episode")
plt.ylabel("Final Vertical Velocity")
plt.title("Baseline: Final Vertical Velocity per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_vertical_velocity.png")
plt.show()

print("\nPlots saved:")
print(os.path.abspath("baseline_reward.png"))
print(os.path.abspath("baseline_steps.png"))
print(os.path.abspath("baseline_vertical_velocity.png"))

#Your graphs show that the random LunarLander controller performs poorly and inconsistently. This is exactly what we expect from a baseline experiment without AI training. These results will serve as the reference for comparison with your PPO-trained agent.
#1. Interpretation of each graph
#Graph 1: Total Reward per Episode
#What it measures: Overall performance during one landing attempt.
#- Most rewards are negative, generally between approximately −50 and −300.
#- Some episodes perform extremely poorly, with rewards below −400.
#- A few episodes achieve rewards near or above zero, but there is no consistent improvement across episodes.
#Inference: The random controller does not reliably learn how to land the spacecraft. Its performance varies considerably because it selects actions randomly.
#Graph 2: Episode Length
#What it measures: Number of simulation steps before an episode ends.
#- Most episodes last approximately 60–120 steps.
#- Some last longer, reaching around 160 steps.
#3- Episode length fluctuates considerably across the 300 attempts.
#Inference: The lander follows inconsistent trajectories. However, a longer episode does not necessarily mean a better landing; it could simply mean that the lander takes longer to crash or reach a terminal condition.
#Graph 3: Final Vertical Velocity
#What it measures: Vertical speed at the end of each episode.
#- Most final vertical velocities are negative.
#- Negative vertical velocity means the lander is moving downward.
#- Several episodes end with substantial downward speeds, reaching approximately −1.5 in the environment's velocity units.
#Inference: Many episodes end while the lander is descending relatively quickly, which can indicate a hard landing or crash. However, vertical velocity alone cannot confirm whether a landing was successful.
#2. Overall conclusion
#Baseline established

#The 300 episodes provide a reference for evaluating the trained agent.



#Performance is inconsistent

#Rewards and episode lengths fluctuate without a clear learning trend.



#PPO should be evaluated against this baseline

#We will check whether training improves rewards, landing success, and touchdown stability.



#3. What these graphs cannot tell us yet
#The graphs do not establish the exact landing success rate. To measure that properly, we should record a success indicator using the environment's termination information and landing conditions, rather than relying on reward alone.

#Also, the final vertical velocity is only one component of touchdown quality. Horizontal velocity, angle, leg contact, and whether the lander crashed also matter.
#4. Next step: calculate baseline statistics
#Before PPO training, calculate the following from your CSV:
#Metric	Purpose
#Mean total reward	Average baseline performance
#Mean episode length	Typical duration of an attempt
#Best and worst reward	Performance range
#Reward standard deviation	How inconsistent the controller is
#Landing success rate	Percentage of successful landings, once success is defined reliably