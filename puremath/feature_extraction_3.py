import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.stats import zscore, iqr

SPEED_OF_SOUND = 343  # m/s
FRAME_DURATION = 0.04
HOP_DURATION = 0.01
LPC_ORDER = 12


def preprocess(filepath):
    y, sr = load_audio(filepath)
    return estimate_vocal_features(y, sr)


def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr


def estimate_vocal_features(y, sr):
    frame_len = int(FRAME_DURATION * sr)
    hop_len = int(HOP_DURATION * sr)

    L_vals, f1_vals, f2_vals = [], [], []

    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i:i + frame_len]
        if np.allclose(frame, 0): continue
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
    """
    L_vals = reject_outliers(L_vals)
    f1_vals = reject_outliers(f1_vals)
    f2_vals = reject_outliers(f2_vals)
    """
    return summarize(L_vals, f1_vals, f2_vals)


def reject_outliers(arr, method="z", threshold=2.5):
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    if method == "z":
        z = np.abs(zscore(arr))
        return arr[z < threshold]
    elif method == "iqr":
        q1, q3 = np.percentile(arr, [25, 75])
        rng = q3 - q1
        return arr[(arr >= q1 - 1.5 * rng) & (arr <= q3 + 1.5 * rng)]
    return arr


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
