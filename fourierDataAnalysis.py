import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import io

# =========================
# SETTINGS
# =========================

fs = 100  # IMPORTANT: dataset uses 100 Hz

cols = [
    "ECG",
    "Acc_X", "Acc_Y", "Acc_Z",
    "Gyr_X", "Gyr_Y", "Gyr_Z",
    "PPG_IR", "PPG_Green", "PPG_Blue"
]

base_url = "https://raw.githubusercontent.com/ludvikalkhoury/DWL-Method/main/wrist-dataset/"

# =========================
# SAFE DATA LOADER
# =========================

def load_data(url, nrows=10000):
    print("Downloading:", url)
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(data), header=None, names=cols, sep=",", nrows=nrows)
    return df

# =========================
# LOAD DATA
# =========================

s1 = load_data(base_url + "Subject%201/Data_Sub_1.txt")
s2 = load_data(base_url + "Subject%202/Data_Sub_2.txt")

print("Data loaded successfully")

# =========================
# CLEAN PPG SIGNAL
# =========================

# for col in ["PPG_IR", "PPG_Green", "PPG_Blue"]:
#     print("s1 before correction:", s1[col])
#     print("s2 before correction:", s2[col])
#     print("s1[col] mean =", s1[col].mean())
#     print("s2[col] mean =", s2[col].mean())
#     s1[col] = s1[col] - s1[col].mean()
#     s2[col] = s2[col] - s2[col].mean()
#     print("s1 after correction:", s1[col])
#     print("s2 after correction:", s2[col])

# =========================
# TIME AXIS
# =========================

ecg_window = 1000  # 10 seconds (100 Hz)
t = np.arange(ecg_window) / fs

# =========================
# ECG + PPG PLOTS
# =========================

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("ECG and PPG Signals — Subjects 1 & 2")

axes[0].plot(t, s1["ECG"].values[:ecg_window])
axes[0].set_title("Subject 1 ECG")

axes[1].plot(t, s1["PPG_IR"].values[:ecg_window], label="Infrared")
axes[1].plot(t, s1["PPG_Green"].values[:ecg_window], label="Green")
axes[1].set_title("Subject 1 PPG")
axes[1].legend()

axes[2].plot(t, s2["ECG"].values[:ecg_window])
axes[2].set_title("Subject 2 ECG")

axes[3].plot(t, s2["PPG_IR"].values[:ecg_window], label="Infrared")
axes[3].plot(t, s2["PPG_Green"].values[:ecg_window], label="Green")
axes[3].set_title("Subject 2 PPG")
axes[3].legend()

for ax in axes:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("ecg_ppg.png", dpi=150)
plt.show()

# =========================
# FOURIER ANALYSIS
# =========================

class FourierAnalyzer:
    def __init__(self, fs):
        self.fs = fs

    def computeFFT(self, signal):
        signal = np.array(signal)
        n = len(signal)

        fftVals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1/self.fs)
        magnitude = np.abs(fftVals) / n

        return freqs, magnitude

    def meanCorrect(self, signal):
        # print("Mean before correction:", np.mean(signal))
        # print(signal)
        signalAfterCorrection = signal - np.mean(signal)
        # print("Mean after correction:", np.mean(signalAfterCorrection))
        # print("List after correction:", signalAfterCorrection)
        return signalAfterCorrection

    def plotFFT(self, signal, title, maxFreq=20):
        # Raw FFT
        rawFreqs, rawMag = self.computeFFT(signal)

        # Mean-corrected FFT
        corrected = self.meanCorrect(signal)
        corrFreqs, corrMag = self.computeFFT(corrected)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(rawFreqs, rawMag)
        axes[0].set_title(title + " - Raw FFT")
        axes[0].set_xlim(0, maxFreq)
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Magnitude")

        axes[1].plot(corrFreqs, corrMag)
        axes[1].set_title(title + " - Mean Corrected FFT")
        axes[1].set_xlim(0, maxFreq)
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Magnitude")

        plt.tight_layout()
        plt.show()

# =========================
# RUN FOURIER ANALYSIS
# =========================

analyzer = FourierAnalyzer(fs)

# Subject 1
analyzer.plotFFT(s1["ECG"].values, "Subject 1 ECG", maxFreq=40)
analyzer.plotFFT(s1["PPG_IR"].values, "Subject 1 PPG IR", maxFreq=15)

# Subject 2
analyzer.plotFFT(s2["ECG"].values, "Subject 2 ECG", maxFreq=40)
analyzer.plotFFT(s2["PPG_Green"].values, "Subject 2 PPG Green", maxFreq=15)

# =========================
# REPORT EXPLANATION
# =========================

"""
Fourier Transform converts a signal from time domain to frequency domain.

Why this matters for HRV:
- Identifies repeating patterns in heart activity
- Helps analyze physiological rhythms
- Used to study autonomic nervous system behavior

Mean correction:
- Removes DC offset (baseline shift)
- Prevents large spike at 0 Hz
- Makes frequency peaks clearer
"""