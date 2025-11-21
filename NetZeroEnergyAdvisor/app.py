# app.py
# Streamlit dashboard for Net Zero Energy Advisor
# Dynamic predictions and recommendation box

import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Paths
# -----------------------
BASE = Path(__file__).resolve().parent
DATA_RAW = BASE / "data" / "raw"
MODELS = BASE / "models"

# -----------------------
# Helper Functions
# -----------------------
def load_and_unwrap(path):
    """Load joblib object and return (model, features)."""
    obj = joblib.load(path)
    if hasattr(obj, "predict"):
        return obj, getattr(obj, "feature_names_in_", None)
    if isinstance(obj, dict):
        if "model" in obj and hasattr(obj["model"], "predict"):
            return obj["model"], obj.get("features", None)
        for v in obj.values():
            if hasattr(v, "predict"):
                return v, obj.get("features", None)
    return None, None

def safe_read_energy(nrows=5000):
    path = DATA_RAW / "energydata_complete.csv"
    usecols = ["date", "Appliances", "T1", "RH_1", "T_out", "RH_out"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["date"], nrows=nrows, low_memory=True)
    return df

def safe_read_solar(nrows=5000):
    path = DATA_RAW / "Plant_1_Generation_Data.csv"
    usecols = ["DATE_TIME", "DAILY_YIELD", "AC_POWER", "DC_POWER"]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["DATE_TIME"], nrows=nrows, low_memory=True)
    return df
# -----------------------
# Build MLR input from user input
# -----------------------
def build_mlr_input_user(df, user_data):
    """
    Build input DataFrame for MLR model based on user-provided values.
    """
    # Simulate last 3 appliance values based on number of appliances input
    num_appliances = user_data["num_appliances"]
    lag_1 = num_appliances * 1.0
    lag_2 = num_appliances * 0.9
    lag_3 = num_appliances * 0.8

    if mlr_features is not None and len(mlr_features) > 0:
        cols = {}
        for f in mlr_features:
            if f == "lag_1":
                cols[f] = lag_1
            elif f == "lag_2":
                cols[f] = lag_2
            elif f == "lag_3":
                cols[f] = lag_3
            else:
                # For other features, use average from dataset
                cols[f] = df[f].mean() if f in df.columns else 0
        return pd.DataFrame([cols])
    else:
        return pd.DataFrame({
            "lag_1": [lag_1],
            "lag_2": [lag_2],
            "lag_3": [lag_3],
        })

# -----------------------
# Recommendation System
# -----------------------
def give_recommendation(consumption, solar, balance, user_data):
    home_size = user_data["home_size"]
    appliances = user_data["num_appliances"]
    panels = user_data["num_panels"]

    if balance > 50:
        status = "success"
        title = "✅ Great job!."
        rec = (
            "- Store extra energy in batteries or export to the grid.\n"
            "- Clean solar panels monthly for best efficiency.\n"
        )
    elif -50 <= balance <= 50:
        status = "warning"
        title = "⚖️ Almost Net-Zero! You're close to balancing consumption and generation."
        rec = (
            "- Use appliances during peak sunlight.\n"
            "- Slightly increase solar output or reduce load by 5–10%.\n"
        )
    else:
        status = "error"
        title = "⚠️ High Consumption Alert! You consume more energy than you generate."
        rec = (
            "- Add 2–3 more solar panels.\n"
            "- Use energy-efficient appliances.\n"
            "- Reduce evening power use or invest in battery backup.\n"
        )

    # Optional additional advice based on home/panels
    if home_size > 2000 or appliances > 15:
        rec += "\n🏠 Your home is large — consider smart thermostats or zoning HVAC."
    elif panels < 6:
        rec += "\n☀️ You have fewer panels — adding 2–3 more could help reach Net-Zero."

    return status, title, rec

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Net-Zero Energy Advisor", layout="wide")
st.title("🌞 Net-Zero Energy Advisor Dashboard")

# -----------------------
# Load Models
# -----------------------
mlr_model, mlr_features = load_and_unwrap(MODELS / "mlr_consumption.pkl")
svr_model, svr_features = load_and_unwrap(MODELS / "svr_solar.pkl")
xgb_model, xgb_features = load_and_unwrap(MODELS / "xgb_netbalance.pkl")

if any(m is None for m in [mlr_model, svr_model, xgb_model]):
    st.error("⚠️ Could not load one or more model files. Check your 'models/' folder.")
    st.stop()

# -----------------------
# Load Datasets
# -----------------------
df_energy = safe_read_energy()
df_solar = safe_read_solar()

if "DATE_TIME" in df_solar.columns:
    df_solar["DATE"] = pd.to_datetime(df_solar["DATE_TIME"]).dt.date
    daily = df_solar.groupby("DATE").agg({"DAILY_YIELD": "sum"}).reset_index()
else:
    daily = pd.DataFrame({"DATE": [], "DAILY_YIELD": []})

# -----------------------
# User Input Section
# -----------------------
st.sidebar.header("🏠 User Information")
user_data = {
    "home_size": st.sidebar.number_input("Home Size (sq. ft.)", 500, 5000, 1500),
    "num_appliances": st.sidebar.number_input("No. of Appliances", 5, 30, 12),
    "num_panels": st.sidebar.number_input("No. of Solar Panels", 2, 20, 6),
}

st.sidebar.write("Inputs used for personalized advice.")

# -----------------------
# Build MLR input (lag features)
# -----------------------
def build_mlr_input(df, idx):
    vals = df["Appliances"].astype(float).reset_index(drop=True)
    i = min(idx, len(vals) - 1)
    lag_vals = vals.iloc[max(0, i - 3):i].tolist()
    while len(lag_vals) < 3:
        lag_vals.insert(0, lag_vals[0] if lag_vals else 0)

    if mlr_features is not None and len(mlr_features) > 0:
        cols = {}
        for f in mlr_features:
            if f.startswith("lag_"):
                if f == "lag_1": cols[f] = lag_vals[-1]
                elif f == "lag_2": cols[f] = lag_vals[-2]
                elif f == "lag_3": cols[f] = lag_vals[-3]
            else:
                cols[f] = df[f].iloc[i] if f in df.columns else 0
        return pd.DataFrame([cols])
    else:
        return pd.DataFrame({
            "lag_1": [lag_vals[-1]],
            "lag_2": [lag_vals[-2]],
            "lag_3": [lag_vals[-3]],
        })

# Build MLR input using user input
X_mlr = build_mlr_input_user(df_energy, user_data)
y_mlr_pred = float(mlr_model.predict(X_mlr)[0])

# Build SVR input based on user input (approximate daily yield from number of panels)
if not daily.empty:
    panels = user_data["num_panels"]
    # Scale daily yield by number of panels
    last_daily_yield = daily["DAILY_YIELD"].iloc[-1]
    y_svr_pred = float(svr_model.predict(pd.DataFrame({"DAILY_YIELD": [last_daily_yield * panels / 6]}))[0])
else:
    y_svr_pred = 0

# XGBoost prediction
X_xgb = pd.DataFrame({"consumption": [y_mlr_pred], "solar": [y_svr_pred]})
y_xgb_pred = float(xgb_model.predict(X_xgb)[0])

# -----------------------
# Dynamic SVR prediction based on user input
# -----------------------
if not daily.empty:
    X_svr = pd.DataFrame({"DAILY_YIELD": [daily["DAILY_YIELD"].iloc[-1]]})
    base_solar = float(svr_model.predict(X_svr)[0])
    # Scale by number of panels
    y_svr_pred = base_solar * (user_data["num_panels"] / 6)
else:
    y_svr_pred = 0

# -----------------------
# Dynamic XGBoost prediction
# -----------------------
X_xgb = pd.DataFrame({"consumption": [y_mlr_pred], "solar": [y_svr_pred]})
y_xgb_pred = float(xgb_model.predict(X_xgb)[0])

# -----------------------
# Recommendations Box
# -----------------------
st.header("💡 Smart Energy Recommendation")

status, title, rec = give_recommendation(y_mlr_pred, y_svr_pred, y_xgb_pred, user_data)

if status == "success":
    bg_color = "#e8f5e9"
    border_color = "#2e7d32"
elif status == "warning":
    bg_color = "#fff8e1"
    border_color = "#f9a825"
else:
    bg_color = "#ffebee"
    border_color = "#c62828"

st.markdown(
    f"""
    <div style="
        background-color:{bg_color};
        border-left: 6px solid {border_color};
        padding: 1.2em;
        border-radius: 8px;
        margin-top: 10px;
        ">
        <h4 style="color:{border_color}; margin-bottom:0.5em;">{title}</h4>
        <p style="font-size: 16px; color:#333; white-space: pre-line;">{rec}</p>
        <b>Calculated Values:</b><br>
        - Predicted Energy Consumption: {y_mlr_pred:.2f} Wh<br>
        - Predicted Solar Output: {y_svr_pred:.2f} kWh<br>
        - Predicted Net Energy Balance: {y_xgb_pred:.2f} Wh
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Plots
# -----------------------
st.header("📊 Visual Insights")
colA, colB, colC = st.columns(3)

# --- PCA Scatter ---
with colA:
    st.subheader("PCA of Lag Features")
    def create_lag_features(df, n_lags=3):
        df_lag = pd.DataFrame()
        for lag in range(1, n_lags + 1):
            df_lag[f"lag_{lag}"] = df["Appliances"].shift(lag)
        df_lag = df_lag.dropna().reset_index(drop=True)
        return df_lag

    mlr_features_df = create_lag_features(df_energy)
    scaler = StandardScaler()
    scaled_lag = scaler.fit_transform(mlr_features_df)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_lag)
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

    fig1, ax1 = plt.subplots()
    ax1.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6, edgecolor='k')
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax1.set_title("PCA Scatter")
    st.pyplot(fig1, use_container_width=True)

# --- SVR vs Linear Regression Curve ---
with colB:
    st.subheader("SVR vs Linear Regression (DC → AC Power)")

    # Prepare dataset
    df_curve = df_solar.dropna().copy()
    df_curve = df_curve.iloc[:5000]  # first 5000 rows

    X = df_curve["DC_POWER"].values.reshape(-1, 1)
    y = df_curve["AC_POWER"].values

    # Linear Regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_lin_pred = lin_reg.predict(X)

    # SVR
    from sklearn.svm import SVR
    svr = SVR(kernel='rbf', C=100, gamma=0.01)
    svr.fit(X, y)
    y_svr_pred = svr.predict(X)

    # Sort for smooth curves
    sorted_idx = X.flatten().argsort()
    X_sorted = X.flatten()[sorted_idx]
    y_lin_sorted = y_lin_pred[sorted_idx]
    y_svr_sorted = y_svr_pred[sorted_idx]

    # Plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(X, y, s=8, alpha=0.4, label="Actual Data")
    ax2.plot(X_sorted, y_lin_sorted, linewidth=2, label="Linear Regression")
    ax2.plot(X_sorted, y_svr_sorted, linewidth=2, label="SVR (Non-Linear)")

    ax2.set_xlabel("DC Power")
    ax2.set_ylabel("AC Power")
    ax2.set_title("SVR vs Linear Regression Curve Fit")
    ax2.legend()
    ax2.grid(alpha=0.3)

    st.pyplot(fig2, use_container_width=True)


# --- Net Energy Balance Heatmap ---
with colC:
    st.subheader("Net Energy Balance Heatmap")
    N_days = 30
    N_days = min(N_days, len(df_energy), len(daily) if not daily.empty else len(df_energy))

    pred_consumption = []
    for i in range(len(df_energy) - N_days, len(df_energy)):
        X_mlr_i = build_mlr_input(df_energy, i)
        pred_consumption.append(float(mlr_model.predict(X_mlr_i)[0]))

    if not daily.empty:
        pred_solar = daily["DAILY_YIELD"].iloc[-N_days:].tolist()
        pred_solar = [x * (user_data["num_panels"] / 6) for x in pred_solar]
    else:
        pred_solar = [0] * N_days

    X_xgb_heat = pd.DataFrame({"consumption": pred_consumption, "solar": pred_solar})
    net_balance_pred = xgb_model.predict(X_xgb_heat)

    heat_df = pd.DataFrame(net_balance_pred, columns=["Net_Balance"]).T
    heat_df.columns = [f"Day {i+1}" for i in range(N_days)]

    fig3, ax3 = plt.subplots(figsize=(12, 2))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        cbar_kws={"label": "Net Balance (Wh)"},
        linewidths=0.5,
        ax=ax3
    )
    ax3.set_ylabel("")
    ax3.set_xlabel("Last N Days")
    ax3.set_title("Predicted Net Energy Balance Heatmap")
    st.pyplot(fig3, use_container_width=True)


