# model_utils.py
# Utilities to build, save, load model and related encoders/pipeline

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gbr_pipeline.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
POLY_PATH = os.path.join(MODEL_DIR, "poly.joblib")

def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_df_for_model(df):
    """
    Input: raw dataframe (should contain columns region, mode, powertrain, category, year, parameter, value)
    Returns df_model (only EV sales rows) and X, y prepared for training/prediction
    """
    df_model = df.copy()
    df_model = df_model[df_model["parameter"] == "EV sales"].reset_index(drop=True)
    # Ensure columns exist
    required = ["region","mode","powertrain","category","year","value"]
    for c in required:
        if c not in df_model.columns:
            raise ValueError(f"Required column missing: {c}")
    return df_model

def train_and_save_model(df):
    """
    Trains a Gradient Boosting model and saves pipeline artifacts:
    - label encoders (dict)
    - polynomial transformer
    - scaler
    - trained model in a single joblib
    """
    ensure_model_dir()
    df_model = prepare_df_for_model(df)

    label_cols = ["region", "mode", "powertrain", "category"]
    le_dict = {}
    for col in label_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    # features and target (log-transform target)
    X_base = df_model[["region","mode","powertrain","category","year"]].copy()
    y = np.log1p(df_model["value"].values)

    # polynomial features for year (degree=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    year_poly = poly.fit_transform(X_base[["year"]])
    year_poly_df = pd.DataFrame(year_poly, columns=["year","year^2"])
    X = pd.concat([X_base[["region","mode","powertrain","category"]].reset_index(drop=True), year_poly_df], axis=1)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split (we keep full data to train final model but evaluate internally)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # model
    gbr = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=7, random_state=42)
    gbr.fit(X_train, y_train)

    # Save artifacts
    joblib.dump({
        "model": gbr,
        "label_encoders": le_dict,
        "scaler": scaler,
        "poly": poly
    }, MODEL_PATH)

    # Also save separate encoder dict for convenience (older code compatibility)
    joblib.dump(le_dict, ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(poly, POLY_PATH)

    return {"model": gbr, "label_encoders": le_dict, "scaler": scaler, "poly": poly}

def load_model_artifacts():
    ensure_model_dir()
    if not os.path.exists(MODEL_PATH):
        return None
    data = joblib.load(MODEL_PATH)
    return data  # dict with keys model, label_encoders, scaler, poly

def predict_sales(model_artifacts, input_dict):
    """
    input_dict should contain keys: 'region','mode','powertrain','category','year' (year numeric)
    Returns predicted sales in original scale (inverse log)
    """
    le_dict = model_artifacts["label_encoders"]
    poly = model_artifacts["poly"]
    scaler = model_artifacts["scaler"]
    model = model_artifacts["model"]

    # encode
    X_row = {}
    for c in ["region","mode","powertrain","category"]:
        val = str(input_dict.get(c, "")).strip()
        le = le_dict[c]
        # if unseen, add a fallback: map to most frequent label (0) or use try/except
        try:
            enc = le.transform([val])[0]
        except Exception:
            # unseen category -> attempt to find closest by string match else fallback 0
            enc = 0
        X_row[c] = enc

    year = float(input_dict.get("year", 2025))
    # polynomial for year
    year_poly = poly.transform([[year]])  # shape (1,2): year, year^2
    # combine
    import numpy as np
    X_arr = np.array([[X_row["region"], X_row["mode"], X_row["powertrain"], X_row["category"], year, year**2]])
    # but scaler expects same columns as training: region,mode,powertrain,category,year,year^2
    X_scaled = scaler.transform(X_arr)
    y_log = model.predict(X_scaled)
    y_pred = np.expm1(y_log)
    return float(y_pred[0])
