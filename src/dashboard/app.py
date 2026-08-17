from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.features import (
    BASE_FEATURES,
    MODEL_FEATURES,
    ROLLING_WINDOW,
    add_time_series_features,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "rf_model.pkl"
THRESHOLD = 50

st.set_page_config(page_title="Turbofan Predictive Maintenance", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Run python src/training/train_model.py first.")
    st.stop()

model = load_model()

st.title("Turbofan Engine Predictive Maintenance")
st.caption(
    "NASA C-MAPSS educational demo using ordered engine histories. "
    "This is not an aviation-certified maintenance system."
)
st.write(
    "Upload a CSV containing engine_id, cycle, three operating settings "
    "and 21 sensor columns."
)

st.sidebar.title("Method")
st.sidebar.write("Model: Random Forest Regressor")
st.sidebar.write(f"Rolling window: {ROLLING_WINDOW} cycles")
st.sidebar.write("Validation: complete held-out engines")
st.sidebar.write(f"Alert threshold: {THRESHOLD} predicted cycles")

uploaded_file = st.file_uploader("Upload engine-history CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    required = ["engine_id", "cycle", *BASE_FEATURES]
    missing = [column for column in required if column not in data.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    counts = data.groupby("engine_id").size()
    short_engines = counts[counts < ROLLING_WINDOW].index.tolist()
    if short_engines:
        st.warning(
            "Some engines contain fewer than five cycles. Early-cycle "
            "predictions use partial histories with zero-filled trend/std "
            f"values: {short_engines[:10]}"
        )

    data = add_time_series_features(data)
    data["Predicted_RUL"] = model.predict(data[MODEL_FEATURES])

    st.subheader("Predictions")
    st.dataframe(
        data[["engine_id", "cycle", "Predicted_RUL"]].head(100),
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum predicted RUL", f"{data['Predicted_RUL'].min():.1f}")
    col2.metric("Average predicted RUL", f"{data['Predicted_RUL'].mean():.1f}")
    col3.metric("Maximum predicted RUL", f"{data['Predicted_RUL'].max():.1f}")

    critical = data[data["Predicted_RUL"] < THRESHOLD]
    if critical.empty:
        st.success("No predictions are below the demonstration threshold.")
    else:
        st.error(
            f"{len(critical)} observations are below the "
            f"{THRESHOLD}-cycle demonstration threshold."
        )

    engine_ids = sorted(data["engine_id"].unique())
    selected_engine = st.selectbox("Select engine", engine_ids)
    engine_data = data[data["engine_id"] == selected_engine]
    st.line_chart(
        engine_data.set_index("cycle")["Predicted_RUL"],
        use_container_width=True,
    )

    if "RUL" in data.columns:
        st.subheader("Predicted vs actual RUL")
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            x=data["RUL"],
            y=data["Predicted_RUL"],
            alpha=0.5,
            ax=axis,
        )
        maximum = max(data["RUL"].max(), data["Predicted_RUL"].max())
        axis.plot([0, maximum], [0, maximum], linestyle="--")
        axis.set_xlabel("Actual RUL")
        axis.set_ylabel("Predicted RUL")
        st.pyplot(figure)
        plt.close(figure)
