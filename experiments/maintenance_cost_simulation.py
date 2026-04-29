import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("models/rf_model.pkl")

# Load data
DATA_PATH = "CMAPSSData/train_FD001.txt"

columns = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
df.columns = columns

# Compute RUL
max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]

df = df.merge(max_cycle, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]
df = df.drop(columns=["max_cycle"])

# --- SAME FEATURE ENGINEERING AS MODEL ---
df = df.sort_values(["engine_id", "cycle"])

for sensor in [f"sensor_{i}" for i in range(1, 22)]:
    df[f"{sensor}_roll_mean"] = (
        df.groupby("engine_id")[sensor]
        .rolling(window=5)
        .mean()
        .reset_index(0, drop=True)
    )
    df[f"{sensor}_roll_std"] = (
        df.groupby("engine_id")[sensor]
        .rolling(window=5)
        .std()
        .reset_index(0, drop=True)
    )
    df[f"{sensor}_trend"] = df.groupby("engine_id")[sensor].diff()

df = df.fillna(0)

# Features
features = list(model.feature_names_in_)

X = df[features]
y_true = df["RUL"]

# Predict
y_pred = model.predict(X)

# ---------------------------
# COST MODEL
# ---------------------------

FAILURE_COST = 100000
MAINTENANCE_COST = 10000
thresholds = [10, 20, 30, 40, 50]

print("\n=== THRESHOLD OPTIMIZATION ===")

for THRESHOLD in thresholds:
    
    maintenance_actions = (y_pred < THRESHOLD).sum()
    predicted_failures = ((y_pred >= THRESHOLD) & (y_true == 0)).sum()

    cost_with = (
        maintenance_actions * MAINTENANCE_COST
        + predicted_failures * FAILURE_COST
    )

    print(f"\nThreshold: {THRESHOLD}")
    print(f"Maintenance actions: {maintenance_actions}")
    print(f"Missed failures: {predicted_failures}")
    print(f"Total cost: {cost_with:,} €")

# Without model (reactive)
failures = (y_true == 0).sum()
cost_without = failures * FAILURE_COST

# With model (predictive)
maintenance_actions = (y_pred < THRESHOLD).sum()
predicted_failures = ((y_pred >= THRESHOLD) & (y_true == 0)).sum()

cost_with = (
    maintenance_actions * MAINTENANCE_COST
    + predicted_failures * FAILURE_COST
)

# ---------------------------
# RESULTS
# ---------------------------

print("\n=== COST SIMULATION ===")
print(f"Failures (no model): {failures}")
print(f"Cost without predictive maintenance: {cost_without:,} €")

print("\nWith predictive maintenance:")
print(f"Maintenance actions: {maintenance_actions}")
print(f"Missed failures: {predicted_failures}")
print(f"Total cost: {cost_with:,} €")

savings = cost_without - cost_with
print(f"\nEstimated savings: {savings:,} €")
