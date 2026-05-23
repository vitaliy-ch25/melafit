import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bcf, sbcf, bbcf, bsbcf, fit, func_defaults
from melafit.markers import midpoint
from melafit.results import SessionInfo, ResultsCollector
from melafit.utils import read_data, prepare_part_data, gen_time_range, phase_to_string

# EXPERIMENTAL: Determine DLMO using the curve fitting approach for partial data

data_path = "./data/"
result_path = "./results/dlmo/"
result_filename = "results_dlmo"

data = read_data(data_path + "dummy_data_dlmo.xlsx")

mel_funcs = [bcf, sbcf, bbcf, bsbcf]

popup_figures = True
delta_t = "1min"
thresh_dlmo = 10

participants = np.unique(data.Participant)

for mel_func in mel_funcs:
    os.makedirs(result_path, exist_ok=True)

    func_name = mel_func.__name__.upper()
    collector = ResultsCollector()

    for participant in participants:

        try:
            p_data = prepare_part_data(data, participant)
            print(p_data)

            # Get and customise default fit parameters
            p0, lb, ub = func_defaults(p_data.Mel, mel_func)
            p0["H"] *= 4
            ub["H"] *= 4

            # Fit curve and compute resampled waveform
            res = fit(p_data.Timestamp, p_data.Mel, mel_func, p0=p0, lb=lb, ub=ub)
            resampled_t = gen_time_range(p_data.Timestamp, step=delta_t)
            resampled_f = mel_func(t=resampled_t, p=res)

            # Compute midpoint, DLMOn and DLMOff (midpoint/DLMOff unreliable for partial data)
            mid = midpoint(resampled_t, resampled_f, thresh_dlmo, thresh_abs=True)

            # Collect results for this participant
            meta = SessionInfo(p_data)
            collector.add(meta, res, mid)

            # Print summary
            print(meta)
            print(res)
            dlmo_str=f"DLMO={phase_to_string(mid.dlmon)}"
            print(dlmo_str)

            # Visualize results
            title_str = f"Date: {meta.start.date()}, {dlmo_str}"

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
