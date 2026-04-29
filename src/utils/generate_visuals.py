import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1. THRESHOLD OPTIMIZATION PLOT
# ---------------------------
thresholds = [10, 20, 30, 40, 50]
costs = [9690000, 19030000, 28150000, 36780000, 45800000]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, marker="o")
plt.axhline(y=10000000, linestyle="--", label="No model baseline")
plt.xlabel("RUL Threshold")
plt.ylabel("Total Cost (€)")
plt.title("Threshold Optimization - Maintenance Cost")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_optimization.png")
plt.close()


# ---------------------------
# 2. FEATURE IMPORTANCE PLOT
# ---------------------------
model = joblib.load("models/rf_model.pkl")

features = list(model.feature_names_in_)
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(9, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()


# ---------------------------
# 3. ARCHITECTURE DIAGRAM (FIXED SPACING)
# ---------------------------
steps = [
    "NASA C-MAPSS Data",
    "Feature Engineering",
    "ML Model",
    "FastAPI",
    "Streamlit Dashboard",
    "Maintenance Decision"
]

plt.figure(figsize=(18, 4))  # ← wider

ax = plt.gca()
ax.axis("off")

x_positions = range(len(steps))

for i, step in enumerate(steps):
    ax.text(
        i * 2, 0,  # ← spacing increased
        step,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="black")
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=((i + 1) * 2 - 0.3, 0),
            xytext=(i * 2 + 0.3, 0),
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

plt.xlim(-1, len(steps) * 2)
plt.ylim(-1, 1)

plt.title("Predictive Maintenance System Architecture")

plt.tight_layout()
plt.savefig("architecture.png")
plt.close()

print("Created:")
print("- threshold_optimization.png")
print("- feature_importance.png")
print("- architecture.png")
