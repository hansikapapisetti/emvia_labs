import math

import urllib.request

#Subject 1 R-peaks

#getting the data from github
def load_rpeaks_from_github(url):
    with urllib.request.urlopen(url) as response:
        lines = response.read().decode("utf-8").splitlines()

    r_peaks = []
    for line in lines:
        line = line.strip()
        if line != "":

            r_peaks.append(int(line))
    return r_peaks


#using the formula to categorize r_peaks
def compute_rr_intervals(r_peaks, fs=100):
    rr_intervals = []

    for i in range(len(r_peaks) - 1):
        diff_samples = r_peaks[i + 1] - r_peaks[i]
        rr_ms = (diff_samples / fs) * 1000
        rr_intervals.append(rr_ms)

    return rr_intervals


#calculating sdnn, rmssd, and pnn50 using formulas
class HRVCalculator:
    def __init__(self, rr_intervals):
        self.rr = rr_intervals

    def sdnn(self):
        n = len(self.rr)
        mean_rr = sum(self.rr) / n
        variance = sum((x - mean_rr) ** 2 for x in self.rr) / (n - 1)
        return math.sqrt(variance)

    def rmssd(self):
        diffs = [self.rr[i+1] - self.rr[i] for i in range(len(self.rr)-1)]
        mean_sq = sum(d**2 for d in diffs) / len(diffs)
        return math.sqrt(mean_sq)

    def pnn50(self):
        diffs = [abs(self.rr[i+1] - self.rr[i]) for i in range(len(self.rr)-1)]
        count = sum(1 for d in diffs if d > 50)
        return (count / len(diffs)) * 100


#running the test file and got outputs
if __name__ == "__main__":

    # RAW GitHub file link (must be raw version)
    url = "https://raw.githubusercontent.com/ludvikalkhoury/DWL-Method/main/wrist-dataset/Subject%201/R_Peak_Sub_1.txt"

    # print("Step 1: Downloading R-peak data from Github")

    r_peaks = load_rpeaks_from_github(url)

    # print(f"Step 2: Loaded {len(r_peaks)} R-peaks")

    rr_intervals = compute_rr_intervals(r_peaks, fs=100)

    # print(f"Step 3: Computed {len(rr_intervals)} RR intervals")

    calc = HRVCalculator(rr_intervals)

    print("\n===== HRV RESULTS =====")
    print("SDNN:", round(calc.sdnn(), 2), "ms")
    print("RMSSD:", round(calc.rmssd(), 2), "ms")
    print("pNN50:", round(calc.pnn50(), 2), "%")

#===== HRV RESULTS for Subject 1 =====
#SDNN: 73.9 ms
#RMSSD: 11.55 ms
#pNN50: 0.4 %