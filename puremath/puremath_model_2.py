import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

FEATURE_COLS = ["mean_L", "median_L", "min_L", "max_L", "std_L",
                "mean_f2_f1", "std_f2_f1",
                "mean_f1", "std_f1", "mean_f2", "std_f2"]


def predict_gender(summary_dict):
    model = xgb.XGBClassifier()
    model.load_model("XGBoost.model")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler = RobustScaler()

    df = pd.DataFrame([summary_dict])[FEATURE_COLS]

    X_poly = poly.fit_transform(df)
    X_scaled = scaler.fit_transform(X_poly)

    proba = model.predict_proba(X_scaled)[0]
    P_male, P_female = proba
    P_nonbinary = 0.0  # Not modeled in binary classifier

    return (P_male, P_female, P_nonbinary)
