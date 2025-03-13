import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import itertools

# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter search space
param_space = {
    "n_estimators": [10, 50, 100, 150, 200],  # Number of trees
    "max_depth": [2, 5, 10, 20, 30],  # Maximum tree depth
    "min_samples_split": [2, 5, 10],  # Min samples for a split
    "min_samples_leaf": [1, 3, 5, 7]  # Min samples per leaf
}

# Convert parameter space to a list of possible actions
actions = list(itertools.product(*param_space.values()))
n_actions = len(actions)

# Initialize Q-table (state-action mapping)
q_table = np.zeros(n_actions)

# RL Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation tradeoff
epsilon_decay = 0.99  # Decay rate for exploration
n_episodes = 50  # Number of training episodes

best_score = 0  # Track the best score found

# Reinforcement learning loop
for episode in range(n_episodes):
    # Choose action using ε-greedy strategy
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(n_actions)  # Explore
    else:
        action_idx = np.argmax(q_table)  # Exploit best known action

    # Get hyperparameters from action index
    params = dict(zip(param_space.keys(), actions[action_idx]))

    # Train and evaluate model
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
    
    # Reward is the accuracy
    reward = score

    # Update Q-table using Q-learning update rule
    best_future_q = np.max(q_table)  # Future reward estimate
    q_table[action_idx] = (1 - alpha) * q_table[action_idx] + alpha * (reward + gamma * best_future_q)

    # Update best score
    if score > best_score:
        best_score = score

    # Decay exploration rate
    epsilon *= epsilon_decay

    print(f"Episode {episode+1}: Params {params}, Score: {score:.4f}")

# Get best parameters
best_action_idx = np.argmax(q_table)
best_params = dict(zip(param_space.keys(), actions[best_action_idx]))
print("\nBest parameters:", best_params)
print("Best cross-validated accuracy:", best_score)

# Train the best model on full training set
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_accuracy)
