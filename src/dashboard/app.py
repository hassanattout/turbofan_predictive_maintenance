import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL_FOLDER = "models"
MODEL_FILENAME = os.path.join(MODEL_FOLDER, "rf_model.pkl")
THRESHOLD = 50

if not os.path.exists(MODEL_FILENAME):
    st.error(
        "Model file not found. Please run `python3 turbofan.py` first "
        "to generate `models/rf_model.pkl`."
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILENAME)

model = load_model()

st.title("Turbofan Engine Predictive Maintenance Dashboard")
st.caption(
    "Predict Remaining Useful Life from turbofan engine sensor data "
    "and flag critical degradation risk."
)
st.write("Upload your engine CSV and get predicted Remaining Useful Life.")

st.sidebar.title("Project Info")
st.sidebar.write("NASA C-MAPSS predictive maintenance demo")
st.sidebar.write(f"Critical RUL threshold: {THRESHOLD} cycles")
st.sidebar.write("Model: Random Forest Regressor")

uploaded_file = st.file_uploader("Upload engine CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    features = ["setting_1", "setting_2", "setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)
    ]

    missing_cols = [col for col in features if col not in data.columns]

    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        st.stop()

    predictions = model.predict(data[features])
    data["Predicted_RUL"] = predictions

    st.subheader("Predictions")

    display_cols = ["Predicted_RUL"]
    if "engine_id" in data.columns:
        display_cols.insert(0, "engine_id")
    if "cycle" in data.columns:
        display_cols.insert(1, "cycle")

    st.dataframe(data[display_cols].head(100))

    with st.expander("View full uploaded dataset"):
        st.dataframe(data)

    st.subheader("Predicted RUL Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Min RUL", f"{data['Predicted_RUL'].min():.1f}")
    col2.metric("Avg RUL", f"{data['Predicted_RUL'].mean():.1f}")
    col3.metric("Max RUL", f"{data['Predicted_RUL'].max():.1f}")

    low_rul = data[data["Predicted_RUL"] < THRESHOLD]

    if not low_rul.empty:
        st.error(
            f"{len(low_rul)} critical predictions: "
            f"RUL below {THRESHOLD} cycles"
        )
    else:
        st.success("No critical low-RUL predictions detected.")

    st.subheader("RUL Trend Over Time")

    if "engine_id" in data.columns and "cycle" in data.columns:
        engine_ids = sorted(data["engine_id"].unique())
        selected_engine = st.selectbox("Select Engine ID", engine_ids)

        engine_data = data[data["engine_id"] == selected_engine]
        st.line_chart(engine_data.set_index("cycle")["Predicted_RUL"])

        st.caption(f"Displaying RUL trend for Engine {selected_engine}")
    else:
        st.line_chart(data["Predicted_RUL"])

    if "RUL" in data.columns:
        st.subheader("Predicted vs Actual RUL")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            x=data["RUL"],
            y=data["Predicted_RUL"],
            alpha=0.5,
            ax=ax,
        )

        max_val = max(data["RUL"].max(), data["Predicted_RUL"].max())
        ax.plot([0, max_val], [0, max_val], linestyle="--")

        ax.set_xlabel("Actual RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title("Predicted vs Actual RUL")

        st.pyplot(fig)
        plt.close(fig)

if os.path.exists("predicted_vs_actual_RUL.png"):
    st.subheader("Model Performance Overview")
    st.image(
        "predicted_vs_actual_RUL.png",
        caption="Predicted vs Actual RUL on validation data",
    )
