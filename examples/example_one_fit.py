import os
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bsbcf, fit
from melafit.markers import area_cog
from melafit.results import SessionInfo, ResultsCollector
from melafit.utils import (read_data, prepare_part_data, gen_time_range)

# Read full profile data from Excel spreadsheet
data = read_data("./data/dummy_data_full.xlsx")

# Prepare results directory and collector
result_path = "./results/one_fit/"
os.makedirs(result_path, exist_ok=True)
collector = ResultsCollector()

participant = 1

# Prepare data for the participant
p_data = prepare_part_data(data, participant)

# Fit curve and compute resampled waveform
res = fit(p_data.Timestamp, p_data.Mel, bsbcf)
resampled_t = gen_time_range(p_data.Timestamp, step="1min")
resampled_f = bsbcf(t=resampled_t, p=res)

# Compute area and COG
ac = area_cog(resampled_t, resampled_f)

# Collect all results for this participant
meta = SessionInfo(p_data)
collector.add(meta, ac)

# Print summary
print(meta)
print(res)

# Visualize results
title_str = (f"{meta}, {ac}, R²={res.r2:.3f}")

plt.close("all")
plt.figure(figsize=(12, 5))
plt.scatter(p_data.Timestamp, p_data.Mel, c='b')
plt.plot(resampled_t, resampled_f, 'g')
plt.xlabel("Time, hh:mm")
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
plt.ylabel("Concentration, pg/ml")
plt.title(title_str)
plt.legend(["Melatonin data", "BSBCF curve", "Threshold"])
plt.savefig(result_path + f"mel_data_{participant}_BSBCF.png")

# Keep the figure open until a button is pressed
plt.waitforbuttonpress()

# Save results to Excel file
collector.save(result_path, "results_one_fit_BSBCF.xlsx")
