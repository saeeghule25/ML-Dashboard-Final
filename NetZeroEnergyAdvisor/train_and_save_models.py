
"""
train_and_save_models.py
-----------------------------------------
Trains and saves 3 ML models for Net-Zero Energy Advisor:
1. Multiple Linear Regression (Energy Consumption with lag features)
2. SVR (Solar Power Generation)
3. XGBoost Regressor (Net Energy Balance)
-----------------------------------------
Keep CSV files in: data/raw/
    - energydata_complete.csv
    - Plant_1_Generation_Data.csv
Run this script: python train_and_save_models.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
from pathlib import Path

# ---------------- PATH SETUP ----------------
BASE = Path(__file__).resolve().parent
DATA_RAW = BASE / "data" / "raw"
MODELS_DIR = BASE / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ===================================================
# PART 1 — ENERGY CONSUMPTION MODEL (MLR with lag)
# ===================================================
print("Training Multiple Linear Regression model...")

df_energy = pd.read_csv(DATA_RAW / "energydata_complete.csv")

# Ensure datetime type
df_energy["date"] = pd.to_datetime(df_energy["date"])
df_energy = df_energy.set_index("date")

# Target: Appliances energy consumption
target = "Appliances"

# Create lag features (previous 3 hours)
for lag in range(1, 4):
    df_energy[f"lag_{lag}"] = df_energy[target].shift(lag)

# Drop missing values due to lagging
df_energy = df_energy.dropna()

X_energy = df_energy[[f"lag_{i}" for i in range(1, 4)]]
y_energy = df_energy[target]

X_train, X_test, y_train, y_test = train_test_split(X_energy, y_energy, test_size=0.2, random_state=42)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

y_pred = mlr_model.predict(X_test)
print("MLR → MAE:", mean_absolute_error(y_test, y_pred))
print("MLR → RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MLR → R2:", r2_score(y_test, y_pred))

pickle.dump(mlr_model, open(MODELS_DIR / "mlr_consumption.pkl", "wb"))

# ===================================================
# PART 2 — SOLAR GENERATION MODEL (SVR)
# ===================================================
print("\nTraining SVR model for solar generation...")

df_solar = pd.read_csv(DATA_RAW / "Plant_1_Generation_Data.csv")
df_solar["DATE_TIME"] = pd.to_datetime(df_solar["DATE_TIME"])

# Aggregate to daily generation
df_daily = df_solar.groupby(df_solar["DATE_TIME"].dt.date).agg({
    "AC_POWER": "sum",
    "DC_POWER": "sum",
    "DAILY_YIELD": "max"
}).reset_index()

df_daily["DAY_INDEX"] = np.arange(len(df_daily))
X_solar = df_daily[["DAY_INDEX"]]
y_solar = df_daily["DAILY_YIELD"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_solar)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_solar, test_size=0.2, random_state=42)

svr_model = SVR(kernel="rbf")
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)

print("SVR → MAE:", mean_absolute_error(y_test, y_pred))
print("SVR → RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("SVR → R2:", r2_score(y_test, y_pred))

pickle.dump({"model": svr_model, "scaler": scaler}, open(MODELS_DIR / "svr_solar.pkl", "wb"))

# ===================================================
# PART 3 — NET-ZERO BALANCE MODEL (XGBoost)
# ===================================================
print("\nTraining XGBoost Regressor (Net-Balance)...")

# For simplicity, join solar + energy consumption on date index length
common_len = min(len(df_daily), len(df_energy))
df_joined = pd.DataFrame({
    "consumption": df_energy[target].iloc[:common_len].values,
    "solar": df_daily["DAILY_YIELD"].iloc[:common_len].values
})
df_joined["net_balance"] = df_joined["solar"] - df_joined["consumption"]

X = df_joined[["consumption", "solar"]]
y = df_joined["net_balance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("XGBoost → MAE:", mean_absolute_error(y_test, y_pred))
print("XGBoost → RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("XGBoost → R2:", r2_score(y_test, y_pred))

pickle.dump(xgb_model, open(MODELS_DIR / "xgb_netbalance.pkl", "wb"))

print("\n✅ All models trained and saved successfully in 'models/' folder.")
