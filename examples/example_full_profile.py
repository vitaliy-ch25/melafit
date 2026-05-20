import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bcf, sbcf, bbcf, bsbcf, fit, rsquared
from melafit.markers import (amplitude, midpoint, area_cog, AnalysisInfo)
from melafit.utils import (read_data, prepare_part_data, compute_wave,
                            phase_to_string, phase_diff, params_to_string,
                            ResultsCollector)

data_path = "./data/"
result_path = "./results/full/"
result_filename = "results_full"

data = read_data(data_path + "dummy_data_full.xlsx")

mel_funcs = [bcf, sbcf, bbcf, bsbcf]

popup_figures = True
dt_minutes = 1.0
thresh_dlmo = 0.25

participants = np.unique(data.Participant)

for mel_func in mel_funcs:

    collector = ResultsCollector()

    for participant in participants:

        try:
            p_data = prepare_part_data(data, participant)
            print(p_data)

            # Fit curve and compute resampled waveform
            res = fit(p_data.Timedays, p_data.Mel, mel_func)
            r2 = rsquared(p_data.Mel, mel_func(t=p_data.Timedays, p=res))
            resampled_curve = compute_wave(p_data.Timedays.min(),
                                           p_data.Timedays.max(),
                                           dt_minutes, mel_func, res)
            resampled_time = pd.date_range(
                p_data.Timestamp.min(),
                periods=len(resampled_curve),
                freq=pd.Timedelta(minutes=dt_minutes))

            # Compute all markers
            ampl = amplitude(resampled_curve)
            mid = midpoint(resampled_time, resampled_curve, thresh_dlmo)
            ac = area_cog(resampled_time, resampled_curve)
            meta = AnalysisInfo(participant, p_data.Timestamp.min(),
                            mel_func.__name__.upper(), r2)

            # Collect all results for this participant
            collector.add(meta, res, ampl, mid, ac)

            # Print summary
            print(f"Fitted function: {meta.func}, "
                  f"parameters: {params_to_string(res)}")
            res_str = (f"Date: {meta.start.date()}, "
                       f"DLMOn={phase_to_string(mid.dlmon)}, "
                       f"DLMOff={phase_to_string(mid.dlmoff)}, "
                       f"Midpoint={phase_to_string(mid.midpoint)}, "
                       f"Area={ac.area:.3f}, "
                       f"COG={phase_to_string(ac.cog)}, "
                       f"R2={r2:.3f}")
            print(res_str)
            print(f"Phase difference COG-Midpoint: "
                  f"{phase_to_string(phase_diff(ac.cog, mid.midpoint))}")

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
                        f"{meta.func} curve",
                        "Threshold"])
            plt.savefig(result_path +
                        f"mel_data_{participant}_{meta.func}.png")

            if popup_figures:
                plt.pause(0.01)

        except Exception as err:
            print(f"Error processing data for participant {participant}: {err}")

    collector.save(result_path,
                   result_filename + f"_{mel_func.__name__.upper()}")
