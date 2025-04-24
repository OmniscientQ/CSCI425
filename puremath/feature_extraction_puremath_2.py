import os
import numpy as np
import librosa
import csv
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from tqdm import tqdm


# === Config ===
input_folder = "./clips"
output_csv = os.path.join("./vocal_tract_summary.csv")

# === Parameters ===
frame_duration = 0.04
hop_duration = 0.01
lpc_order = 12
speed_of_sound = 343  # m/s

# === Header for final summary table ===
header = [
    "filename",
    "duration_sec",
    "mean_L", "median_L", "std_L", "min_L", "max_L", "range_L", "skew_L", "kurtosis_L", "delta_L_std",
    "mean_f1", "median_f1", "std_f1", "min_f1", "max_f1", "range_f1", "skew_f1", "kurtosis_f1",
    "mean_f2", "median_f2", "std_f2", "min_f2", "max_f2", "range_f2", "skew_f2", "kurtosis_f2",
    "mean_f2_f1", "std_f2_f1"
]

# === Open output CSV ===
with open(output_csv, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)

    # === Get list of MP3s ===
    mp3_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]
    # === Process each mp3 ===
    for filename in tqdm(mp3_files, desc="Processing MP3s"):
        filepath = os.path.join(input_folder, filename)
        print(f"Processing {filename}")

        y, sr = librosa.load(filepath, sr=None)
        frame_length = int(frame_duration * sr)
        hop_length = int(hop_duration * sr)

        L_values = []
        f1_values = []
        f2_values = []

        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            if np.allclose(frame, 0):
                continue
            frame *= np.hamming(len(frame))

            try:
                a = librosa.lpc(frame, order=lpc_order)
                w = np.linspace(0, np.pi, 512)
                freqs = w * sr / (2 * np.pi)
                h = 1 / np.abs(np.polyval(a, np.exp(1j * w)))

                peaks, _ = find_peaks(h, distance=20)
                formants = freqs[peaks]

                if len(formants) > 1 and 80 < formants[0] < 1200:
                    f1 = formants[0]
                    f2 = formants[1]
                    L = speed_of_sound / (4 * f1)

                    L_values.append(L * 100)  # in cm
                    f1_values.append(f1)
                    f2_values.append(f2)
            except np.linalg.LinAlgError:
                continue

        if len(L_values) == 0:
            print(f"  ➤ Skipped (no valid data)")
            continue

        # === Convert to arrays
        L_arr = np.array(L_values)
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f2_f1 = f2_arr / f1_arr

        # === Summary Stats
        def stats(arr):
            return [
                np.mean(arr), np.median(arr), np.std(arr),
                np.min(arr), np.max(arr), np.ptp(arr),
                skew(arr), kurtosis(arr)
            ]

        delta_L_std = np.std(np.diff(L_arr))
        duration = len(L_values) * hop_duration

        row = [filename, duration]
        row += stats(L_arr) + [delta_L_std]
        row += stats(f1_arr)
        row += stats(f2_arr)
        row += [np.mean(f2_f1), np.std(f2_f1)]

        writer.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])

print(f"\n✅ Summary written to: {output_csv}")
