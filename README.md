# AI-Bot-Project-
# LunarLander AI Using Reinforcement Learning (PPO)

## 1. Project Overview

This project trains an AI agent to land a spacecraft safely in the LunarLander simulation using Reinforcement Learning (RL).

The agent learns by interacting with the environment, receiving rewards or penalties, and improving its actions over multiple training episodes.

We use **Proximal Policy Optimization (PPO)**, an RL algorithm provided by Stable-Baselines3, and the LunarLander environment from Gymnasium.

**Main objective:** Develop an AI agent that can control the spacecraft, descend toward the landing pad, and achieve stable landings more consistently than a random controller.

## 2. How Does It Work?

The project has four main stages.

### Stage 1: Baseline Experiment

Before training the AI, the lander is controlled using random actions.

- The experiment records up to 300 episodes.
- Each episode represents one landing attempt.
- Rewards, episode lengths, and final-state measurements are saved to a CSV file.
- Graphs help us understand how the untrained random controller performs.

This baseline gives us a reference for measuring improvement after training.

### Stage 2: PPO Training

The PPO agent interacts with LunarLander and learns a policy for selecting actions.

The environment provides an eight-value observation containing the lander's position, velocity, angle, angular velocity, and leg-contact states.

The agent can choose among four discrete actions: do nothing, fire the left engine, fire the main engine, or fire the right engine.

During initial training, we use Gymnasium's built-in reward function rather than a custom reward function.

### Stage 3: Visual Testing

After training, the saved PPO model is loaded into a visible LunarLander window.

This allows us to watch the AI control the spacecraft and inspect its final position, velocity, angle, reward, and leg contacts.

### Stage 4: Evaluation and Comparison

The trained agent is evaluated over multiple episodes.

We compare its performance with the random baseline using:

- Mean episode reward
- Reward variability
- Episode length
- Landing success rate, using a clearly defined criterion
- Reward and performance graphs

A higher reward generally indicates better performance, but reward alone does not guarantee a safe landing.

## 3. Project Workflow

Random Controller → Baseline Data Collection → Baseline Analysis → PPO Training → Visual Testing → Final Evaluation → Baseline vs PPO Comparison

## 4. Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PPO reinforcement-learning algorithm
- Matplotlib
- Pandas
- Git and GitHub
- VS Code

## 5. Project Files

| File | Purpose |
|---|---|
| `baseline_lunarlander.py` | Runs the random-action baseline experiment |
| `plot_baseline.py` | Generates baseline graphs |
| `baseline_analysis.py` | Calculates baseline statistics |
| `train_ppo.py` | Trains the initial PPO agent |
| `continue_training.py` | Continues training a saved PPO model |
| `watch_landing.py` | Opens the game window to watch the trained agent |
| `evaluate_ppo.py` | Evaluates the trained model |
| `evaluate_final.py` | Runs the final evaluation and creates comparison plots |
| `baseline_readings.csv` | Stores baseline episode readings |
| `ppo_final_evaluation.csv` | Stores final PPO episode results, if generated |
| `ppo_final_summary.csv` | Stores final PPO summary statistics, if generated |
| `baseline_vs_ppo.png` | Compares baseline and PPO mean rewards, if generated |
| `requirements.txt` | Lists Python package dependencies |
| `ppo_lunlarlander_v2.zip` | Saved PPO model, if included in the repository |

Filenames may vary slightly depending on the model version and scripts included in the repository.

## 6. How to Set Up the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/AnanyaPatil4/AI-Bot-Project-.git
cd AI-Bot-Project-
```

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is missing, install the main dependencies manually:

```bash
python -m pip install "gymnasium[box2d]" "stable-baselines3[extra]" matplotlib pandas
```

### Step 3: Run the baseline experiment

```bash
python baseline_lunarlander.py
```

### Step 4: Train PPO

```bash
python train_ppo.py
```

To continue training a saved model, use `continue_training.py` after verifying the model filename it loads and the filename it saves.

### Step 5: Watch the trained agent

```bash
python watch_landing.py
```

Make sure the model filename used in the script matches the saved model in the project folder.

### Step 6: Evaluate the model

```bash
python evaluate_final.py
```

This produces evaluation results and graphs if the script and required input files are present.

**Note:** Running the training script again will start a new training run unless the script explicitly loads an existing model.

## 7. Current Results

The initial random baseline recorded 300 episodes, with a mean reward of approximately **−182.44**.

During an initial visual test of the continued PPO model, five episodes produced the following rewards:

- Episode 1: −37.01
- Episode 2: 179.99
- Episode 3: −16.06
- Episode 4: −20.09
- Episode 5: 214.81

Both landing legs registered contact in all five episodes. Two episodes reached a reward of at least 200.

These five episodes are preliminary observations, not a reliable estimate of the overall landing success rate. Use the final evaluation CSV and summary to report the larger test results when available.

## 8. Future Improvements

- Improve landing consistency and touchdown stability.
- Evaluate performance over more episodes.
- Compare different PPO hyperparameters.
- Investigate reward shaping if the built-in reward does not produce the desired behavior.
- Add trajectory visualizations and more detailed landing metrics.
- Compare different RL algorithms.

## 9. Team Contributions

Team members can improve the project by modifying the training setup, evaluating different policies, analysing results, and proposing better landing criteria.

Use separate Git branches for experimental changes and submit pull requests before merging into the main branch.

## 10. Conclusion

This project demonstrates the reinforcement-learning workflow from establishing a random baseline to training, visualizing, and evaluating a PPO agent in LunarLander. The objective is to measure and improve reliable landing behavior through repeatable experiments.
