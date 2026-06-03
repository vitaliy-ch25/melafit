"""
Determine DLMO using curve fitting for partial data.

Bimodal functions make little sense as we have only the rising slope. 
Besides that, we want to have fewer parameters to fit. For these two reasons, 
we exclude BSBCF and BBCF and leave only BCF and SBCF for this analysis. 
We also increase the default H since the maximum of the curve is not captured 
in the data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import melafit as mf

data_path = "./data/"
result_path = "./results/dlmo/"
result_filename = "results_dlmo"

data = mf.read_data(data_path + "dummy_data_dlmo.xlsx")

mel_funcs = [mf.bcf, mf.sbcf]

popup_figures = True
delta_t = "1min"
thresh_dlmo = 10 # pg/ml

participants = np.unique(data.Participant)

for mel_func in mel_funcs:
    os.makedirs(result_path, exist_ok=True)

    func_name = mel_func.__name__.upper()
    collector = mf.ResultsCollector()

    for participant in participants:

        try:
            p_data = mf.prepare_part_data(data, participant)
            print(p_data)

            # Get and customise default fit parameters
            p0, lb, ub = mf.func_defaults(p_data.Mel, mel_func)
            p0["H"] *= 4
            ub["H"] *= 4

            # Fit curve and compute resampled waveform
            res = mf.fit(p_data.Timestamp, p_data.Mel, mel_func, p0=p0, lb=lb, ub=ub)
            resampled_t = mf.gen_time_range(p_data.Timestamp, step=delta_t, full_day=False)
            resampled_f = mel_func(t=resampled_t, p=res)

            # Compute Dim Light Melatonin Onset (DLMO) with absolute threshold
            dlmo = mf.dlmo(resampled_t, resampled_f, thresh_dlmo, thresh_abs=True)

            # Collect results for this participant
            meta = mf.SessionInfo(p_data)
            collector.add(meta, res, dlmo)

            # Print summary
            print(meta)
            print(res)
            print(dlmo)

            # Visualize results
            title_str = f"Date: {meta.start.date()}, {dlmo}"

            plt.close("all")
            plt.figure(figsize=(12, 5))
            plt.scatter(p_data.Timestamp, p_data.Mel, c='b')
            plt.plot(resampled_t, resampled_f, 'g')
            plt.plot(resampled_t, dlmo.threshold * np.ones(resampled_t.shape), 'r')
            plt.xlabel("Time, hh:mm")
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
            plt.ylabel("Concentration, pg/ml")
            plt.title(title_str)
            plt.legend(["Melatonin data", f"{func_name} curve", "Threshold"])
            plt.savefig(result_path + f"mel_data_{participant}_{func_name}.png")

            if popup_figures:
                plt.pause(0.01)

        except Exception as err:
            print(f"Error processing data for participant {participant}: {err}")

    collector.save(result_path, result_filename + "_" + func_name)
