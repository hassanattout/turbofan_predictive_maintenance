import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATA_PATH = "CMAPSSData/train_FD001.txt"

columns = (
    ["engine_id", "cycle"]
    + ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data():
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)

    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])

    return df


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {
        "Model": name,
        "RMSE": round(rmse, 2)
    }


def main():
    df = load_data()

    features = (
        ["setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    X = df[features]
    y = df["RUL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    results.append(
        evaluate_model(
            "Linear Regression",
            LinearRegression(),
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    if XGBOOST_AVAILABLE:
        results.append(
            evaluate_model(
                "XGBoost",
                XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                X_train,
                X_test,
                y_train,
                y_test
            )
        )
    else:
        print("XGBoost is not installed. Run: python3 -m pip install xgboost")

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison_results.csv", index=False)

    print("\nModel Comparison Results:")
    print(results_df)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.title("Model Comparison - RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()