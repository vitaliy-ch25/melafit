"""
Determine phase markers from full melatonin profiles via fitting of all 
available analytical functions with a two-stage approach: fit interpolated data
first, then fit original data with parameters from the first fit as initial
guess.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from collections.abc import Mapping
import pandas as pd
import melafit as mf

data_path = "./data/"
result_path = "./results/full/"
result_filename = "results_full"

# If True, fit interpolated data first, then fit original data with parameters
# from the first fit as initial guess
two_stage_fit = True # False

if two_stage_fit:
    result_path += "two_stage_fit/"

data = mf.read_data(data_path + "dummy_data_full.xlsx")

mel_funcs = [mf.bcf, mf.sbcf, mf.bbcf, mf.bsbcf]

popup_figures = True
delta_t = "1min"
thresh_dlmo = 0.25

participants = np.unique(data.Participant)

os.makedirs(result_path, exist_ok=True)

for mel_func in mel_funcs:
    func_name = mel_func.__name__.upper()
    collector = mf.ResultsCollector()

    for participant in participants:

        try:
            p_data = mf.prepare_part_data(data, participant)
            print(p_data)

            # Get standard initial guess and bounds for mel_func
            p0, lb, ub = mf.func_defaults(p_data.Mel.values, mel_func)

            if two_stage_fit:
                fit_time = mf.gen_time_range(p_data.Timestamp, step="1min")
                fit_data = mf.resample_t(fit_time, p_data, max_gap="1hour", lead_trail=False)
                plt.plot(fit_data.Timestamp, fit_data.Mel, 'b')
                res = mf.fit(fit_data.Timestamp, fit_data.Mel, mel_func, p0=p0, lb=lb, ub=ub)
                res = mf.fit(p_data.Timestamp, p_data.Mel, mel_func, p0=res, lb=lb, ub=ub)
            else:
                res = mf.fit(p_data.Timestamp, p_data.Mel, mel_func, p0=p0, lb=lb, ub=ub)

            resampled_t = mf.gen_time_range(p_data.Timestamp, step=delta_t)
            resampled_f = mel_func(t=resampled_t, p=res)

            # Compute all markers
            ampl = mf.amplitude(resampled_f)
            mid = mf.midpoint(resampled_t, resampled_f, thresh_dlmo)
            ac = mf.area_cog(resampled_t, resampled_f)

            # Collect all results for this participant
            meta = mf.SessionInfo(p_data)
            collector.add(meta, res, ampl, mid, ac)

            # Print summary
            print(meta)
            print(res)
            print(mid, ac)
            print(f"COG-Midpoint={mf.phase_to_string(mf.phase_diff(ac.cog, mid.midpoint))}")

            # Visualize results
            title_str = f"Date: {meta.start.date()}, {mid}, {ac}, R²={res.r2:.3f}"

            plt.close("all")
            plt.figure(figsize=(12, 5))
            plt.scatter(p_data.Timestamp, p_data.Mel, c='b')
            plt.plot(resampled_t, resampled_f, 'g')
            plt.plot(resampled_t, mid.threshold * np.ones(resampled_t.shape), 'r')
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
