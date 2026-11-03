"""
Compare a one-stage and a two-stage SBCF fit for one participant whose
melatonin profile was not sampled during the night.

In the two-stage fit, the SBCF function is first fitted to the data
interpolated onto a 1-minute grid. This result is then used as the initial
guess for a second fit to the original data. Interpolation acts as an
implicit prior that favours a broad nocturnal peak, which can prevent the
fit from collapsing into a narrow peak when the declining limb is sparsely
sampled. Note that both fits may reach a similar R², so always inspect the
plotted curves and the resulting phase markers.

The dummy data were generated from a known BSBCF ground truth with 15%
multiplicative noise, which is plotted for comparison.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import dates
import melafit as mf

# Ground truth BSBCF parameters used to generate the dummy data
truth_p = {"phi": 0.125, "b": 2.0, "H": 60.0, "c": 0.35, "v": 0.35, "m": 0.15}

# Read sparsely sampled profile data from Excel spreadsheet
data = mf.read_data("./data/dummy_data_two_stage.xlsx")

# Prepare results directory and collector
result_path = "./results/two_stage_fit/"
os.makedirs(result_path, exist_ok=True)
collector = mf.ResultsCollector()

participant = 1

# Prepare data for the participant
p_data = mf.prepare_part_data(data, participant)

# One-stage fit: fit curve directly to the original data
res_one = mf.fit(p_data.Timestamp, p_data.Mel, mf.sbcf)

# Two-stage fit, stage 1: fit curve to the data interpolated onto a 1-minute
# grid; gaps longer than 6 h remain unfilled, no padding beyond the data
interp_t = mf.gen_time_range(p_data.Timestamp, step="1min")
interp_data = mf.resample_t(interp_t, p_data, max_gap="6h", lead_trail=False)
res_interp = mf.fit(interp_data.Timestamp, interp_data.Mel, mf.sbcf)

# Two-stage fit, stage 2: fit curve to the original data, starting from the
# stage 1 result
res_two = mf.fit(p_data.Timestamp, p_data.Mel, mf.sbcf, p0=res_interp)

# Compute resampled waveforms
resampled_t = mf.gen_time_range(p_data.Timestamp, step="1min")
resampled_one = mf.sbcf(t=resampled_t, p=res_one)
resampled_two = mf.sbcf(t=resampled_t, p=res_two)
resampled_truth = mf.bsbcf(t=resampled_t, p=truth_p)

# Compute area and COG for both fits and the ground truth
ac_one = mf.area_cog(resampled_t, resampled_one)
ac_two = mf.area_cog(resampled_t, resampled_two)
ac_truth = mf.area_cog(resampled_t, resampled_truth)

# Collect the results of the two-stage fit for this participant
meta = mf.SessionInfo(p_data)
collector.add(meta, res_two, ac_two)

# Print summary
print(meta)
print(f"One-stage: {res_one}")
print(f"One-stage: {ac_one}")
print(f"Two-stage: {res_two}")
print(f"Two-stage: {ac_two}")
print(f"Ground truth: {ac_truth}")

# Visualize results
title_str = (f"{meta}\n"
             f"One-stage: {ac_one}, R²={res_one.r2:.3f}\n"
             f"Two-stage: {ac_two}, R²={res_two.r2:.3f}\n"
             f"Ground truth: {ac_truth}")

plt.close("all")
plt.figure(figsize=(12, 5))
plt.scatter(p_data.Timestamp, p_data.Mel, c='b')
plt.plot(interp_data.Timestamp, interp_data.Mel, 'b:')
plt.plot(resampled_t, resampled_truth, 'c')
plt.plot(resampled_t, resampled_one, 'r')
plt.plot(resampled_t, resampled_two, 'g')
plt.xlabel("Time, hh:mm")
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
plt.ylabel("Concentration, pg/ml")
plt.title(title_str)
plt.legend(["Melatonin data", "Interpolated data", "Ground truth",
            "SBCF curve, one-stage fit", "SBCF curve, two-stage fit"])
plt.tight_layout()
plt.savefig(result_path + f"mel_data_{participant}_SBCF_two_stage.png")

# Show plot for 1 second
plt.pause(1.0)

# Save results to Excel file
collector.save(result_path, "results_two_stage_fit_SBCF.xlsx")
