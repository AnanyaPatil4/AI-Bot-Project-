
import gymnasium as gym
import random
import time
import csv
import os

# Experiment settings
TIME_LIMIT = 25 * 60  # 25 minutes
MAX_EPISODES_TO_SAVE = 300
CSV_FILE = "baseline_readings.csv"

# Open LunarLander with a visible game window
env = gym.make("LunarLander-v3", render_mode="human")

records = []
start_time = time.time()
episode = 0

try:
    while time.time() - start_time < TIME_LIMIT:
        observation, info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Random action; no trained AI
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

        elapsed = time.time() - start_time
        episode += 1

        # Save only the first 300 completed episodes
        if len(records) < MAX_EPISODES_TO_SAVE:
            records.append({
                "episode": episode,
                "elapsed_seconds": round(elapsed, 2),
                "steps": steps,
                "total_reward": round(total_reward, 2),
                "final_x": round(float(observation[0]), 4),
                "final_y": round(float(observation[1]), 4),
                "final_x_velocity": round(float(observation[2]), 4),
                "final_y_velocity": round(float(observation[3]), 4),
                "final_angle": round(float(observation[4]), 4),
                "final_angular_velocity": round(float(observation[5]), 4),
                "left_leg_contact": int(observation[6]),
                "right_leg_contact": int(observation[7]),
                "terminated": terminated,
                "truncated": truncated
            })

        print(
            f"Episode {episode} | "
            f"Steps: {steps} | "
            f"Reward: {total_reward:.2f} | "
            f"Elapsed: {elapsed / 60:.1f} min"
        )

finally:
    env.close()

    if records:
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print(f"\nExperiment finished.")
    print(f"Completed episodes: {episode}")
    print(f"Rows saved: {len(records)}")
    print(f"CSV file: {os.path.abspath(CSV_FILE)}")

