import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import io
from scipy.signal import find_peaks

# how fast the sensor records — 100 times per second
fs = 100

# names for each column in the data file
cols = [
    "ECG",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z",
    "PPG_IR", "PPG_Green", "PPG_Blue"
]

base_url = "https://raw.githubusercontent.com/ludvikalkhoury/DWL-Method/main/wrist-dataset/"


# downloads the txt file from github and turns it into a table we can work with
def load_data(url, nrows=10000):
    print("Downloading:", url)
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(data), header=None, names=cols, sep=",", nrows=nrows)
    return df

s1 = load_data(base_url + "Subject%201/Data_Sub_1.txt")
s2 = load_data(base_url + "Subject%202/Data_Sub_2.txt")
print("Data loaded successfully\n")


# finds the heartbeat peaks in the ecg and measures the gap between them
# those gaps are the nn intervals — basically the time between each heartbeat
def get_nn_intervals(ecg_signal, fs=100):
    # find_peaks looks for the tall spikes (r-peaks) in the ecg
    # distance=50 means two peaks must be at least 50 samples apart (0.5 seconds)
    # height filters out tiny bumps that aren't real heartbeats
    peaks, _ = find_peaks(ecg_signal, distance=50, height=np.mean(ecg_signal))

    # convert peak positions (sample numbers) to actual time in seconds
    peak_times = peaks / fs

    # subtract consecutive times to get the gap between each beat, then convert to ms
    nn_intervals = np.diff(peak_times) * 1000

    # throw away anything that's impossibly fast (200+ bpm) or impossibly slow (30 bpm)
    nn_intervals = nn_intervals[(nn_intervals > 300) & (nn_intervals < 2000)]
    return nn_intervals, peaks


# draws the ecg and puts a red X on every detected heartbeat so you can check it looks right
def plot_peaks(ecg_signal, peaks, title, fs=100, window=1000):
    t = np.arange(window) / fs
    visible_peaks = peaks[peaks < window]  # only show peaks inside the 10 second window

    plt.figure(figsize=(14, 4))
    plt.plot(t, ecg_signal[:window], label="ECG")
    plt.plot(visible_peaks / fs, ecg_signal[visible_peaks], "rx", markersize=10, label="R-peaks")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


# this class takes a list of nn intervals and computes 3 hrv numbers from them
class classifier:
    def __init__(self, rr_intervals_ms):
        self.rr = np.array(rr_intervals_ms, dtype=float)

    # sdnn — how spread out the intervals are overall
    # high sdnn = heart is flexible, low = heart is stiff/stressed
    def sdnn(self):
        if len(self.rr) < 2:
            return None
        return float(np.std(self.rr, ddof=1))

    # rmssd — how much each beat differs from the one right before it
    # more sensitive to quick changes than sdnn
    def rmssd(self):
        if len(self.rr) < 2:
            return None
        diffs = np.diff(self.rr)
        return float(np.sqrt(np.mean(diffs ** 2)))

    # pnn50 — what percent of beat gaps differ by more than 50ms from the previous one
    def pnn50(self):
        if len(self.rr) < 2:
            return None
        diffs = np.abs(np.diff(self.rr))
        return float((np.sum(diffs > 50) / len(diffs)) * 100)


# randomly picks some beats and marks them as bad (simulates noisy real-world data)
# returns a list of True/False — True means the beat is good, False means toss it
def flag_bad_beats(num_beats, bad_fraction, rng=None):
    if rng is None:
        rng = random.Random()
    if num_beats < 1:
        return np.array([], dtype=bool)
    num_bad = int(round(num_beats * bad_fraction))
    num_bad = min(num_bad, num_beats)
    bad_indices = set(rng.sample(range(num_beats), num_bad))
    return np.array([i not in bad_indices for i in range(num_beats)], dtype=bool)


# uses the True/False mask to keep only the good beats
def apply_bad_beat_mask(rr_intervals, good_mask):
    rr_intervals = np.array(rr_intervals, dtype=float)
    if len(rr_intervals) != len(good_mask):
        raise ValueError("RR interval array and mask must have the same length.")
    return rr_intervals[good_mask]


# runs sdnn, rmssd, pnn50 on whatever intervals you pass in
def compute_metrics(rr_intervals):
    calc = classifier(rr_intervals)
    return {
        "count": len(rr_intervals),
        "sdnn":  calc.sdnn(),
        "rmssd": calc.rmssd(),
        "pnn50": calc.pnn50(),
    }


# the main experiment — tests how much the hrv numbers change as you remove more bad beats
# runs multiple trials per noise level so results aren't just luck of the draw
def run_noise_test(rr_intervals, bad_percents=(10, 20, 30), trials=5, seed=42):
    rr_intervals = np.array(rr_intervals, dtype=float)
    rng = random.Random(seed)  # seed makes random choices repeatable
    results = []

    # first compute clean baseline with no bad beats removed
    baseline = compute_metrics(rr_intervals)
    results.append({"condition": "baseline", "trial": 0, "bad_percent": 0, **baseline})

    for pct in bad_percents:
        bad_fraction = pct / 100.0
        for trial in range(1, trials + 1):
            good_mask = flag_bad_beats(len(rr_intervals), bad_fraction, rng=rng)
            filtered_rr = apply_bad_beat_mask(rr_intervals, good_mask)
            metrics = compute_metrics(filtered_rr)
            results.append({
                "condition": f"{pct}% bad beats",
                "trial": trial,
                "bad_percent": pct,
                **metrics
            })

    return results


# prints every single trial as a table row
def print_results_table(results, subject_label):
    print(f"\n{'='*86}")
    print(f"HRV Noise Test — {subject_label}")
    print("-" * 86)
    print(f"{'Condition':<18} {'Trial':<6} {'Count':<8} {'SDNN':<12} {'RMSSD':<12} {'pNN50':<12}")
    print("-" * 86)
    for row in results:
        sdnn  = "None" if row["sdnn"]  is None else f"{row['sdnn']:.2f}"
        rmssd = "None" if row["rmssd"] is None else f"{row['rmssd']:.2f}"
        pnn50 = "None" if row["pnn50"] is None else f"{row['pnn50']:.2f}"
        print(f"{row['condition']:<18} {row['trial']:<6} {row['count']:<8} {sdnn:<12} {rmssd:<12} {pnn50:<12}")


# averages the trials for each noise level so you get one clean number per condition
def summarize_by_condition(results):
    grouped = {}
    for row in results:
        if row["condition"] == "baseline":
            continue
        grouped.setdefault(row["condition"], []).append(row)

    print("\nSummary by Condition")
    print("-" * 50)
    for condition, rows in grouped.items():
        sdnn_vals  = [r["sdnn"]  for r in rows if r["sdnn"]  is not None]
        rmssd_vals = [r["rmssd"] for r in rows if r["rmssd"] is not None]
        pnn50_vals = [r["pnn50"] for r in rows if r["pnn50"] is not None]
        print(condition)
        print(f"  Avg SDNN : {np.mean(sdnn_vals):.2f} ms"  if sdnn_vals  else "  Avg SDNN : None")
        print(f"  Avg RMSSD: {np.mean(rmssd_vals):.2f} ms" if rmssd_vals else "  Avg RMSSD: None")
        print(f"  Avg pNN50: {np.mean(pnn50_vals):.2f} %"  if pnn50_vals else "  Avg pNN50: None")
        print()


# plots nn intervals as a line so you can eyeball if the heartbeat timing looks consistent
# a flat line = steady heart rate, big jumps = something weird going on
def plot_nn_intervals(nn_intervals, title):
    plt.figure(figsize=(14, 4))
    plt.plot(nn_intervals, marker="o", markersize=3, linewidth=1)
    plt.axhline(np.mean(nn_intervals), color="r", linestyle="--", label=f"Mean: {np.mean(nn_intervals):.1f} ms")
    plt.title(title)
    plt.xlabel("Beat number")
    plt.ylabel("NN Interval (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# loop runs the whole pipeline for both subjects back to back
for label, df in [("Subject 1", s1), ("Subject 2", s2)]:
    print(f"\n--- {label} ---")
    ecg = df["ECG"].values

    nn, peaks = get_nn_intervals(ecg, fs=fs)

    if len(nn) == 0:
        print(f"No valid NN intervals found for {label}. Check ECG signal quality.")
        continue

    print(f"Peaks detected: {len(peaks)}")
    print(f"NN intervals:   {len(nn)}")
    print(f"Mean NN:        {np.mean(nn):.1f} ms")
    print(f"Est. HR:        {60000 / np.mean(nn):.1f} BPM")  # 60000ms / avg gap = beats per minute

    plot_peaks(ecg, peaks, f"{label} — ECG with R-peaks", fs=fs)
    plot_nn_intervals(nn, f"{label} — NN Intervals")

    results = run_noise_test(nn, bad_percents=(10, 20, 30), trials=5, seed=42)
    print_results_table(results, label)
    summarize_by_condition(results)