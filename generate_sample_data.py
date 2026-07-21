"""Generate a sample binary classification dataset for testing."""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

data = {
    "age": np.random.randint(18, 80, n),
    "income": np.random.lognormal(10.5, 0.5, n).astype(int),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 40, n),
    "loan_amount": np.random.randint(1000, 100000, n),
    "debt_ratio": np.round(np.random.uniform(0, 0.8, n), 3),
    "num_accounts": np.random.randint(1, 15, n),
    "gender": np.random.choice(["M", "F"], n),
    "education": np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], n, p=[0.3, 0.4, 0.2, 0.1]
    ),
    "city_tier": np.random.choice(["Tier1", "Tier2", "Tier3"], n),
}

df = pd.DataFrame(data)

# Introduce missing values and outliers
for col in ["income", "credit_score", "debt_ratio"]:
    mask = np.random.random(n) < 0.05
    df.loc[mask, col] = np.nan

df.loc[np.random.choice(n, 10, replace=False), "income"] = df["income"].max() * 3

# Create balanced binary target with realistic relationship
score = (
    0.02 * (df["age"] - 40)
    + 0.00002 * (df["income"].fillna(df["income"].median()) - 40000)
    + 0.004 * (df["credit_score"].fillna(650) - 650)
    - 3 * (df["debt_ratio"].fillna(0.3) - 0.3)
    + 0.03 * (df["employment_years"] - 10)
    + np.random.normal(0, 1.2, n)
)
df["default"] = (score < 0).astype(int)

# Add some duplicate rows
df = pd.concat([df, df.sample(20)], ignore_index=True)

df.to_csv("data/sample_loan_default.csv", index=False)
print(f"Generated {len(df)} rows -> data/sample_loan_default.csv")
print(f"Target distribution:\n{df['default'].value_counts()}")
