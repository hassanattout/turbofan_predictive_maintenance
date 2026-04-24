import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CONFIGURATION
# ---------------------------
DATA_FOLDER = "CMAPSSData"
MODEL_FOLDER = "models"
FIGURE_FOLDER = "figures"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)

MODEL_FILENAME = os.path.join(MODEL_FOLDER, "rf_model.pkl")
FIGURE_FILENAME = os.path.join(FIGURE_FOLDER, "predicted_vs_actual_RUL.png")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def compute_rul(df):
    """Compute Remaining Useful Life for each engine."""
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycles, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)

    return df


def load_data(train_file):
    columns = (
        ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(train_file, sep=r"\s+", header=None)
    df.columns = columns
    df = compute_rul(df)

    return df


# ---------------------------
# LOAD DATA
# ---------------------------
train_file = os.path.join(DATA_FOLDER, "train_FD001.txt")
df_train = load_data(train_file)

features = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]

X = df_train[features]
y = df_train["RUL"]

# ---------------------------
# SPLIT DATA
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# MODEL TRAINING
# ---------------------------
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------
# VALIDATION
# ---------------------------
y_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)

print(f"Validation RMSE: {rmse:.2f} cycles")

# ---------------------------
# SAVE MODEL
# ---------------------------
joblib.dump(model, MODEL_FILENAME)
print(f"Model saved: {MODEL_FILENAME}")

# ---------------------------
# PLOT PREDICTED VS ACTUAL RUL
# ---------------------------
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_val, y=y_pred, alpha=0.5)

max_value = max(y_val.max(), y_pred.max())
plt.plot([0, max_value], [0, max_value], linestyle="--")

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs Actual RUL")
plt.text(10, max_value * 0.9, f"RMSE: {rmse:.2f} cycles")

plt.tight_layout()
plt.savefig(FIGURE_FILENAME)
plt.savefig("predicted_vs_actual_RUL.png")
plt.show()

print(f"Figure saved: {FIGURE_FILENAME}")
print("Root figure saved: predicted_vs_actual_RUL.png")
