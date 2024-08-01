# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, make_scorer
import joblib

# Load your dataset
data = pd.read_csv("dataset.csv")

# Example preprocessing (handle missing values, feature engineering, etc.)
data["user_growth_rate"] = (data["new_users"] - data["users_left"]) / data[
    "existing_users_before"
]

# Define features and target
X = data[
    [
        "new_users",
        "users_left",
        "existing_users_before",
        "existing_users_after",
        "user_growth_rate",
    ]
]
y = data[
    "strategy_effectiveness"
]  # Ensure this column has the values: 'positive', 'negative', 'average'

# Encode target variable if necessary
y = y.map({"positive": 1, "negative": 0, "average": 2})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define the scoring metric as recall
scorer = make_scorer(recall_score, average="macro")

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scorer
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, "best_model.pkl")

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["negative", "positive", "average"]
    )
)

# Recall score
recall = recall_score(y_test, y_pred, average="macro")
print(f"Recall Score: {recall:.4f}")
