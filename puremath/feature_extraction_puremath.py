import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv

# === Configuration ===
input_folder = "./"  # <-- change this
output_folder = os.path.join(input_folder, "csv")
os.makedirs(output_folder, exist_ok=True)

# === Parameters ===
frame_duration = 0.04  # 40ms
hop_duration = 0.01    # 10ms
lpc_order = 12
speed_of_sound = 343  # m/s

# === Processing loop ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")

        y, sr = librosa.load(filepath, sr=None)
        frame_length = int(frame_duration * sr)
        hop_length = int(hop_duration * sr)

        timestamps = []
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

                    timestamps.append(i / sr)
                    L_values.append(L * 100)
                    f1_values.append(f1)
                    f2_values.append(f2)
            except np.linalg.LinAlgError:
                continue

        # === Save to CSV ===
        base_name = os.path.splitext(filename)[0]
        csv_path = os.path.join(output_folder, f"{base_name}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Time (s)", "Vocal Tract Length (cm)", "f1 (Hz)", "f2 (Hz)"])
            for t, L, f1, f2 in zip(timestamps, L_values, f1_values, f2_values):
                writer.writerow([f"{t:.4f}", f"{L:.2f}", f"{f1:.2f}", f"{f2:.2f}"])

        print(f"  ➤ Saved to: {csv_path}")

print("\n✅ All files processed with formant tracking!")
