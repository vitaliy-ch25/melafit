import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bcf, sbcf, bbcf, bsbcf, fit, func_defaults
from melafit.markers import midpoint
from melafit.results import SessionInfo, ResultsCollector
from melafit.utils import (read_data, prepare_part_data, gen_time_range,
                            phase_to_string, params_to_string)

# EXPERIMENTAL: Determine DLMO using the curve fitting approach for partial data

data_path = "./data/"
result_path = "./results/dlmo/"
result_filename = "results_dlmo"

data = read_data(data_path + "dummy_data_dlmo.xlsx")

mel_funcs = [bcf, sbcf, bbcf, bsbcf]

popup_figures = True
dt_minutes = 1.0
thresh_dlmo = 10

participants = np.unique(data.Participant)

for mel_func in mel_funcs:

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
            res = fit(p_data.Timestamp, p_data.Mel, mel_func,
                      p0=p0, lb=lb, ub=ub)
            resampled_time = gen_time_range(p_data.Timestamp.min(),
                                            p_data.Timestamp.max(),
                                            dt_minutes)
            resampled_curve = mel_func(t=resampled_time, p=res)

            # Compute DLMO only (midpoint/offset unreliable for partial data)
            resampled_time_dt = pd.DatetimeIndex(dates.num2date(resampled_time))
            mid = midpoint(resampled_time_dt, resampled_curve, thresh_dlmo,
                           thresh_abs=True)
            meta = SessionInfo(participant, p_data.Timestamp.min(),
                           p_data.Timestamp.max())

            # Collect results for this participant
            collector.add(meta, res, mid)

            # Print summary
            print(f"Fitted function: {mel_func.__name__.upper()}, "
                  f"parameters: {params_to_string(res)}")
            res_str = (f"Date: {meta.start.date()}, "
                       f"DLMOn={phase_to_string(mid.dlmon)}")
            print(res_str)

            # Visualize results
            plt.close("all")
            plt.figure(figsize=(12, 5))
            plt.scatter(p_data.Timestamp, p_data.Mel, c='b')
            plt.plot(resampled_time, resampled_curve, 'g')
            plt.plot(resampled_time,
                     mid.threshold * np.ones(resampled_time.shape), 'r')
            plt.xlabel("Time, hh:mm")
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
            plt.ylabel("Concentration, pg/ml")
            plt.title(res_str)
            plt.legend(["Melatonin data",
                        f"{mel_func.__name__.upper()} curve",
                        "Threshold"])
            plt.savefig(result_path +
                        f"mel_data_{participant}_{mel_func.__name__.upper()}.png")

            if popup_figures:
                plt.pause(0.01)

        except Exception as err:
            print(f"Error processing data for participant {participant}: {err}")

    collector.save(result_path,
                   result_filename + f"_{mel_func.__name__.upper()}")
