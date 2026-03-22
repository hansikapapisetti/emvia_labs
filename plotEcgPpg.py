import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#libraries imported
# pandas = reads data files
# matplotlib = draws graphs
# numpy = does math

# settings
fs = 256  # The sensor records 256 data points every second

# Names for the 10 columns in the data file
cols = ["ECG", "Acc_X", "Acc_Y", "Acc_Z",
        "Gyr_X", "Gyr_Y", "Gyr_Z",
        "PPG_IR", "PPG_Green", "PPG_Blue"]


# This is the link to the data files on GitHub
base_url = "https://raw.githubusercontent.com/ludvikalkhoury/DWL-Method/main/wrist-dataset/"

# Download Subject 1's data (only first 10,000 rows since the data set is huge)
s1 = pd.read_csv(base_url + "Subject%201/Data_Sub_1.txt",
                 header=None, names=cols, sep=',', nrows=10000)

# Download Subject 2's data
s2 = pd.read_csv(base_url + "Subject%202/Data_Sub_2.txt",
                 header=None, names=cols, sep=',', nrows=10000)


# #optional
# # The PPG sensor has a huge constant value (like 22000)
# # Subtracting the average removes that big constant
# # so we can actually see the pulse wave clearly
# for col in ["PPG_IR", "PPG_Green", "PPG_Blue"]:
#     s1[col] = s1[col] - s1[col].mean()  # fix Subject 1
#     s2[col] = s2[col] - s2[col].mean()  # fix Subject 2

#controls how many samples to plot
ecg_window = 2560  # 10 seconds × 256 samples/sec = 2560 samples


# Create a time axis from 0 to 10 seconds
# (dividing sample numbers by 256 converts them to seconds)
t = np.arange(ecg_window) / fs

# ECG + PPG GRAPH
# Creates a figure with 4 graphs stacked on top of each other
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("ECG and PPG Signals — Subjects 1 & 2", fontsize=14, fontweight='bold')

# Graph 1: Subject 1 ECG (heartbeat electrical signal)
axes[0].plot(t, s1["ECG"].values[:ecg_window], color='steelblue', linewidth=0.8)
axes[0].set_title("Subject 1 — ECG")
axes[0].set_ylabel("Amplitude")

# Graph 2: Subject 1 PPG (blood pulse measured by light sensor)
axes[1].plot(t, s1["PPG_IR"].values[:ecg_window], color='tomato', linewidth=0.8, label="Infrared")
axes[1].plot(t, s1["PPG_Green"].values[:ecg_window], color='seagreen', linewidth=0.8, label="Green")
axes[1].set_title("Subject 1 — PPG")
axes[1].legend()

# Graph 3: Subject 2 ECG
axes[2].plot(t, s2["ECG"].values[:ecg_window], color='steelblue', linewidth=0.8)
axes[2].set_title("Subject 2 — ECG")
axes[2].set_ylabel("Amplitude")

# Graph 4: Subject 2 PPG
axes[3].plot(t, s2["PPG_IR"].values[:ecg_window], color='tomato', linewidth=0.8, label="Infrared")
axes[3].plot(t, s2["PPG_Green"].values[:ecg_window], color='seagreen', linewidth=0.8, label="Green")
axes[3].set_title("Subject 2 — PPG")
axes[3].legend()

# Add "Time (s)" label to all 4 graphs
for ax in axes:
    ax.set_xlabel("Time (s)")

plt.tight_layout()                            # auto-fix spacing between graphs
plt.savefig("ecg_ppg_subjects.png", dpi=150)  # save as image file
plt.show()                                    # pop up the graph window

#For IMU graph:
# IMU = the motion sensor
# It measures movement (accelerometer) and rotation (gyroscope)
imu_window = 2560 # how many samples to plot for the IMU (accelerometer + gyroscope) graphs.
## In our code they're both 2560, so they both show 10 seconds of data.


fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
fig2.suptitle("IMU Data — Subject 1", fontsize=14, fontweight='bold')

# Graph 1: Accelerometer — measures how fast the wrist is moving
# X, Y, Z are the three directions (left-right, up-down, front-back)
axes2[0].plot(t, s1["Acc_X"].values[:imu_window], label="X")
axes2[0].plot(t, s1["Acc_Y"].values[:imu_window], label="Y")
axes2[0].plot(t, s1["Acc_Z"].values[:imu_window], label="Z")
axes2[0].set_title("Accelerometer")
axes2[0].set_ylabel("Acceleration")
axes2[0].legend()

# Graph 2: Gyroscope — measures how fast the wrist is rotating
axes2[1].plot(t, s1["Gyr_X"].values[:imu_window], label="X")
axes2[1].plot(t, s1["Gyr_Y"].values[:imu_window], label="Y")
axes2[1].plot(t, s1["Gyr_Z"].values[:imu_window], label="Z")
axes2[1].set_title("Gyroscope")
axes2[1].set_ylabel("Angular Velocity")
axes2[1].set_xlabel("Time (s)")
axes2[1].legend()

plt.tight_layout()
plt.savefig("imu_subject1.png", dpi=150)  # save IMU graph as image
plt.show()


#Work Cited
#Alkhoury, L., Choi, J., Chandran, V. D., De Carvalho, G. B., Pal, S., & Kam, M. (2022). Dual Wavelength Photoplethysmography Framework for Heart Rate Calculation. Sensors, 22(24), 9955.