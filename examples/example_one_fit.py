import os
import matplotlib.pyplot as plt
from matplotlib import dates
import melafit as mf

# Read full profile data from Excel spreadsheet
data = mf.read_data("./data/dummy_data_full.xlsx")

# Prepare results directory and collector
result_path = "./results/one_fit/"
os.makedirs(result_path, exist_ok=True)
collector = mf.ResultsCollector()

participant = 1

# Prepare data for the participant
p_data = mf.prepare_part_data(data, participant)

# Fit curve and compute resampled waveform
res = mf.fit(p_data.Timestamp, p_data.Mel, mf.bsbcf)
resampled_t = mf.gen_time_range(p_data.Timestamp, step="1min")
resampled_f = mf.bsbcf(t=resampled_t, p=res)

# Compute area and COG
ac = mf.area_cog(resampled_t, resampled_f)

# Collect all results for this participant
meta = mf.SessionInfo(p_data)
collector.add(meta, res, ac)

# Print summary
print(meta)
print(res)
print(ac)

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
plt.legend(["Melatonin data", "BSBCF curve"])
plt.savefig(result_path + f"mel_data_{participant}_BSBCF.png")

# Keep the figure open until a button is pressed
plt.waitforbuttonpress()

# Save results to Excel file
collector.save(result_path, "results_one_fit_BSBCF.xlsx")
