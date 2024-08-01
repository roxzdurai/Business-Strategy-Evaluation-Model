import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
new_users = np.random.randint(50, 200, n_samples)
users_left = np.random.randint(10, 100, n_samples)
existing_users_before = np.random.randint(100, 500, n_samples)
existing_users_after = existing_users_before + new_users - users_left
user_growth_rate = (new_users - users_left) / existing_users_before


# Define strategy effectiveness based on some rules for demonstration
def determine_effectiveness(new_users, users_left):
    if new_users > users_left * 2:
        return "positive"
    elif new_users < users_left:
        return "negative"
    else:
        return "average"


strategy_effectiveness = [
    determine_effectiveness(nu, ul) for nu, ul in zip(new_users, users_left)
]

# Create the DataFrame
data = pd.DataFrame(
    {
        "new_users": new_users,
        "users_left": users_left,
        "existing_users_before": existing_users_before,
        "existing_users_after": existing_users_after,
        "user_growth_rate": user_growth_rate,
        "strategy_effectiveness": strategy_effectiveness,
    }
)

# Save to CSV for later use
data.to_csv("dataset.csv", index=False)
