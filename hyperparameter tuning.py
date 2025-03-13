import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 10, 200)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 2, 32)  # Maximum tree depth
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)  # Min samples for a split
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)  # Min samples per leaf

    # Create the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    # Cross-validation to evaluate model performance
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
    return score

# Create and run the study
study = optuna.create_study(direction="maximise")
study.optimize(objective, n_trials=50)

# Best hyperparameters
print("Best parameters:", study.best_params)
print("Best cross-validated accuracy:", study.best_value)

# Train the best model
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

# Test the model
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_accuracy)
