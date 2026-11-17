# AI-Based Lunar Lander Control Using Reinforcement Learning, Machine Learning and Deep Learning

## 1. Project Overview

This project develops and evaluates an AI-based autonomous spacecraft landing system using Reinforcement Learning (RL), Classical Machine Learning (ML), and Deep Learning (DL) in the Gymnasium LunarLander-v3 environment.

The project initially trains an AI agent to land a spacecraft safely using Proximal Policy Optimization (PPO), an RL algorithm provided by Stable-Baselines3. The agent learns through repeated interactions with the environment, receiving rewards or penalties and improving its landing policy over multiple training episodes.

In addition to reinforcement learning, Classical Machine Learning and Deep Learning techniques are applied to analyse episode-level data, identify landing patterns, predict landing outcomes, and evaluate model performance.

**Main objective:** Develop an intelligent autonomous landing system, evaluate its performance against a random controller, and investigate how machine learning models can analyse and predict successful spacecraft landing outcomes.

## 2. How Does It Work?

The project consists of six main stages.

### Stage 1: Baseline Experiment

Before training the AI, the spacecraft is controlled using random actions.

- The experiment records 300 episodes.
- Each episode represents one landing attempt.
- Rewards, episode lengths, and final-state measurements are saved to CSV files.
- Graphs are generated to understand the performance of the untrained random controller.

This baseline provides a reference for measuring improvements after training.

### Stage 2: PPO Training

The PPO agent interacts with the LunarLander environment and learns an optimal policy for selecting actions.

- The environment provides an eight-dimensional observation containing position, velocity, angle, angular velocity, and leg-contact states.
- The agent selects among four discrete actions: do nothing, fire the left engine, fire the main engine, or fire the right engine.
- PPO uses the environment's built-in reward function during training.
- Initial training is followed by additional training to improve landing performance.

### Stage 3: Visual Testing

After training, the saved PPO model is loaded into a visible LunarLander simulation.

This allows us to observe the spacecraft's behaviour and inspect:
- Landing trajectory.
- Episode reward.
- Final position and velocity.
- Spacecraft angle.
- Left and right leg contacts.

### Stage 4: Evaluation and Comparison

The trained PPO models are evaluated over multiple episodes and compared against the random baseline.

Performance is measured using:
- Mean episode reward.
- Reward variability.
- Episode length.
- Landing success rate.
- Final position and velocity.
- Landing stability and reward distributions.

A higher reward generally indicates better performance, but reward alone does not guarantee a safe landing.

### Stage 5: Classical Machine Learning Analysis

Classical Machine Learning techniques are applied to the recorded episode datasets to understand landing behaviour and predict outcomes.

The following algorithms are implemented:

- **Linear Regression:** Predicts episode reward using final-state features.
- **Logistic Regression:** Predicts binary landing outcomes.
- **K-Means Clustering:** Groups episodes based on reward and episode length.
- **DBSCAN:** Identifies episode clusters and unusual landing behaviours.
- **Support Vector Machine (SVM):** Classifies successful and unsuccessful landing episodes.
- **Decision Tree:** Analyses classification patterns and decision rules.
- **Random Forest:** Predicts landing outcomes using multiple decision trees.

These models help identify relationships between episode characteristics and landing performance.

### Stage 6: Deep Learning and Performance Analysis

A Neural Network is implemented using NumPy to predict landing outcomes through supervised learning.

The network uses a 3–8–1 architecture:
- Input layer: 3 neurons.
- Hidden layer: 8 neurons with tanh activation.
- Output layer: 1 neuron with sigmoid activation.

The network is trained using:
- Forward propagation.
- Backpropagation.
- Gradient descent.
- Binary cross-entropy loss.

The machine learning models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

Five-fold stratified cross-validation is used for comparing the supervised classification models.

## 3. Project Workflow

Random Controller → Baseline Data Collection → Baseline Analysis → PPO Training → Visual Testing → Final PPO Evaluation → Baseline vs PPO Comparison → Data Preprocessing → Regression and Clustering → Classification → Neural Network Training → Model Performance Analysis.

## 4. Technologies Used

- Python
- Gymnasium (LunarLander-v3)
- Stable-Baselines3
- Proximal Policy Optimization (PPO)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Git and GitHub
- VS Code

## 5. Project Files

### Reinforcement Learning

| File | Purpose |
|---|---|
| `baseline_lunarlander.py` | Runs the random-action baseline experiment |
| `plot_baseline.py` | Generates baseline performance graphs |
| `baseline_analysis.py` | Calculates baseline statistics |
| `train_ppo.py` | Trains the initial PPO agent |
| `continue_training.py` | Continues training a saved PPO model |
| `watch_landing.py` | Visually tests the trained agent |
| `evaluate_ppo.py` | Evaluates the trained PPO model |
| `evaluate_final.py` | Performs final evaluation and generates comparison results |

### Classical Machine Learning and Deep Learning

| File | Purpose |
|---|---|
| `linear_regression.py` | Implements NumPy linear regression for reward prediction |
| `logistic_regression.py` | Implements NumPy logistic regression for landing prediction |
| `kmeans_clustering.py` | Implements K-Means clustering from scratch |
| `dbscan_clustering.py` | Performs density-based episode clustering |
| `neural_network_numpy.py` | Implements a 3–8–1 neural network using backpropagation |
| `classification_models.py` | Compares SVM, Decision Tree, and Random Forest models |
| `landing_analysis.py` | Compares different landing success criteria |
| `common.py` | Provides shared data-loading and preprocessing functions |

### Dataset and Results

| File | Purpose |
|---|---|
| `baseline_readings.csv` | Stores random-controller episode readings |
| `ppo_evaluation.csv` | Stores initial PPO evaluation results |
| `ppo_final_evaluation.csv` | Stores final PPO evaluation results |
| `ppo_final_summary.csv` | Stores final PPO summary statistics |
| `classification_5fold_results.csv` | Stores classification model comparison results (generated) |
| `kmeans_episode_clusters.csv` | Stores K-Means cluster assignments (generated) |
| `dbscan_episode_clusters.csv` | Stores DBSCAN cluster assignments (generated) |
| `linear_regression_actual_vs_predicted.png` | Visualises reward prediction performance (generated) |
| `classification_model_comparison.png` | Compares classification model performance (generated) |

The repository also contains saved PPO models, evaluation plots, and the project research paper.

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

Install additional machine learning dependencies if required:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

For the LunarLander environment, ensure Box2D dependencies are installed:

```bash
python -m pip install "gymnasium[box2d]" "stable-baselines3[extra]"
```

### Step 3: Run the baseline experiment

```bash
python baseline_lunarlander.py
```

### Step 4: Train PPO

```bash
python train_ppo.py
```

To continue training an existing model, use:

```bash
python continue_training.py
```

Verify the model filename before running.

### Step 5: Watch the trained agent

```bash
python watch_landing.py
```

### Step 6: Evaluate the PPO model

```bash
python evaluate_final.py
```

### Step 7: Run Classical Machine Learning models

```bash
python machine_learning/linear_regression.py
python machine_learning/logistic_regression.py
python machine_learning/kmeans_clustering.py
python machine_learning/dbscan_clustering.py
python machine_learning/classification_models.py
```

### Step 8: Run the Deep Learning model

```bash
python machine_learning/neural_network_numpy.py
```

### Step 9: Analyse landing criteria

```bash
python machine_learning/landing_analysis.py
```

**Note:** The machine learning scripts use the existing episode-level CSV files. PPO training does not need to be repeated to perform the ML analysis. Run the required baseline and PPO evaluation scripts first if the corresponding datasets are unavailable.

## 7. Current Results

The project evaluates three controller stages: Random, PPO-v1, and PPO-v2.

### Reinforcement Learning Results

| Evaluation Stage | Episodes | Mean Reward | Landing Rate |
|---|---:|---:|---:|
| Random Controller | 300 | -182.4 | 6% |
| PPO-v1 | 100 | -120.3 | 0% |
| PPO-v2 | 100 | 133.7 | 52% |

The PPO-v2 model demonstrates a substantial improvement over the random controller and the initial PPO-v1 model.

### Landing Success Criterion Analysis

The PPO-v2 evaluation produced the following results:

- 52 episodes satisfied the project-defined landing criterion.
- 45 episodes achieved a reward of at least 200.
- 38 episodes satisfied both criteria.
- 25 episodes satisfied the stricter combined criterion, including the project landing rule, reward of at least 200, and horizontal position within 0.2 units of the landing pad centre.

This demonstrates that reward alone is not sufficient to establish safe landing performance.

### Classical Machine Learning Results

The research evaluates multiple classification algorithms using five-fold stratified cross-validation.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.61 | 0.70 |
| SVM Linear | 0.69 | 0.70 |
| SVM RBF | 0.71 | 0.79 |
| Decision Tree | 0.74 | 0.74 |
| Decision Tree (Depth 3) | 0.78 | 0.76 |
| Random Forest | 0.80 | 0.86 |

Random Forest achieves the highest reported classification accuracy and ROC-AUC among these models.

### Deep Learning Results

The NumPy-based 3–8–1 neural network achieves:

- Accuracy: 0.84
- F1-score: 0.86
- ROC-AUC: 0.89

These results are based on the reported 75/25 train-test experiment and should be distinguished from the five-fold classification results above.

## 8. Future Improvements

- Improve landing consistency and touchdown stability.
- Evaluate PPO performance across multiple independent training seeds.
- Increase the number of evaluation episodes.
- Investigate reward shaping and position-aware landing objectives.
- Experiment with different PPO hyperparameters.
- Implement advanced neural networks using PyTorch or TensorFlow.
- Explore additional deep learning architectures.
- Compare reinforcement learning algorithms such as DQN and SAC where appropriate.
- Improve trajectory visualisation and landing performance analysis.
- Investigate model generalisation and prediction reliability.

## 9. Team Contributions

Team members contribute to the development and evaluation of the autonomous landing system through:

- Reinforcement learning model development and training.
- Baseline experiments and PPO performance evaluation.
- Classical machine learning implementation and analysis.
- Neural network development and performance evaluation.
- Data preprocessing, visualisation, and result interpretation.
- Research documentation and experimental improvements.

Separate Git branches can be used for experimental changes, with pull requests submitted before merging into the main branch.

## 10. Conclusion

This project demonstrates an integrated approach to autonomous spacecraft landing using Reinforcement Learning, Classical Machine Learning, and Deep Learning.

The PPO agent learns to control the spacecraft through repeated interaction with the LunarLander environment and demonstrates improved performance compared with the random controller.

Classical machine learning models provide additional insights into episode behaviour, clustering patterns, and landing outcome prediction, while the NumPy-based neural network demonstrates the application of backpropagation for supervised learning.

The research also highlights the importance of carefully defining landing success, as higher episode rewards do not necessarily guarantee safe and stable touchdown.

Overall, the project combines intelligent control, predictive modelling, and data-driven performance evaluation to investigate autonomous spacecraft landing.
