'''
Uses the dill library to pickle Evelyn's pure math model
'''

import librosa
import numpy as np
import xgboost as xgb
import pandas as pd
from dill import dump
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures, RobustScaler


FEATURE_COLS = ["mean_L", "median_L", "min_L", "max_L", "std_L",
                "mean_f2_f1", "std_f2_f1",
                "mean_f1", "std_f1", "mean_f2", "std_f2"]
SPEED_OF_SOUND = 343  # m/s
FRAME_DURATION = 0.04
HOP_DURATION = 0.01
LPC_ORDER = 12

model = xgb.XGBClassifier()
model.load_model("XGBoost.model")


def predict_gender(summary_dict):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler = RobustScaler()

    df = pd.DataFrame([summary_dict])[FEATURE_COLS]

    X_poly = poly.fit_transform(df)
    X_scaled = scaler.fit_transform(X_poly)

    proba = model.predict_proba(X_scaled)[0]
    P_male, P_female = proba
    P_nonbinary = 0.0  # Not modeled in binary classifier

    return (P_male, P_female, P_nonbinary)


def summarize(L, f1, f2):
    def stats(arr):
        return {
            "mean": np.mean(arr) if len(arr) else np.nan,
            "median": np.median(arr) if len(arr) else np.nan,
            "std": np.std(arr) if len(arr) else np.nan,
            "min": np.min(arr) if len(arr) else np.nan,
            "max": np.max(arr) if len(arr) else np.nan
        }
    return {
        "mean_L": stats(L)["mean"],
        "median_L": stats(L)["median"],
        "std_L": stats(L)["std"],
        "min_L": stats(L)["min"],
        "max_L": stats(L)["max"],
        "mean_f1": stats(f1)["mean"],
        "std_f1": stats(f1)["std"],
        "mean_f2": stats(f2)["mean"],
        "std_f2": stats(f2)["std"],
        "mean_f2_f1": np.mean(np.array(f2) / np.array(f1)),
        "std_f2_f1": np.std(np.array(f2) / np.array(f1))
    }


def estimate_vocal_features(y, sr):
    frame_len = int(FRAME_DURATION * sr)
    hop_len = int(HOP_DURATION * sr)

    L_vals, f1_vals, f2_vals = [], [], []

    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i:i + frame_len]
        if np.allclose(frame, 0):
            continue
        frame *= np.hamming(len(frame))

        try:
            a = librosa.lpc(frame, order=LPC_ORDER)
            w = np.linspace(0, np.pi, 512)
            freqs = w * sr / (2 * np.pi)
            h = 1 / np.abs(np.polyval(a, np.exp(1j * w)))

            peaks, _ = find_peaks(h, distance=20)
            formants = freqs[peaks]

            if len(formants) > 1 and 80 < formants[0] < 1200:
                f1 = formants[0]
                f2 = formants[1]
                L = SPEED_OF_SOUND / (4 * f1)
                L_vals.append(L * 100)
                f1_vals.append(f1)
                f2_vals.append(f2)
        except np.linalg.LinAlgError:
            continue

    # L_vals = reject_outliers(L_vals)
    # f1_vals = reject_outliers(f1_vals)
    # f2_vals = reject_outliers(f2_vals)
    return summarize(L_vals, f1_vals, f2_vals)


def preproc_evelyn(x, sr):
    '''
    Given some sample data, return a 3-tuple prediction
    '''

    return predict_gender(
        estimate_vocal_features(x, sr))


if __name__ == '__main__':
    with open('pure_math_model.dill', 'wb') as f:
        dump(preproc_evelyn, f, recurse=True, byref=False)
