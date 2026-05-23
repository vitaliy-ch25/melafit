import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bcf, sbcf, bbcf, bsbcf, fit
from melafit.markers import amplitude, midpoint, area_cog
from melafit.results import SessionInfo, ResultsCollector
from melafit.utils import (read_data, prepare_part_data, resample_time,
                            phase_to_string, phase_diff, params_to_string)

data_path = "./data/"
result_path = "./results/full/"
result_filename = "results_full"

data = read_data(data_path + "dummy_data_full.xlsx")

mel_funcs = [bcf, sbcf, bbcf, bsbcf]

popup_figures = True
delta_t = "1min"
thresh_dlmo = 0.25

participants = np.unique(data.Participant)

for mel_func in mel_funcs:

    collector = ResultsCollector()

    for participant in participants:

        try:
            p_data = prepare_part_data(data, participant)
            print(p_data)

            # Fit curve and compute resampled waveform
            res = fit(p_data.Timestamp, p_data.Mel, mel_func)
            resampled_time = resample_time(p_data.Timestamp, step=delta_t)
            resampled_curve = mel_func(t=resampled_time, p=res)

            # Compute all markers
            ampl = amplitude(resampled_curve)
            mid = midpoint(resampled_time, resampled_curve, thresh_dlmo)
            ac = area_cog(resampled_time, resampled_curve)
            meta = SessionInfo(p_data)

            # Collect all results for this participant
            collector.add(meta, res, ampl, mid, ac)

            # Print summary
            print(f"Fitted function: {mel_func.__name__.upper()}, "
                  f"parameters: {params_to_string(res)}")
            res_str = (f"Date: {meta.start.date()}, "
                       f"DLMOn={phase_to_string(mid.dlmon)}, "
                       f"DLMOff={phase_to_string(mid.dlmoff)}, "
                       f"Midpoint={phase_to_string(mid.midpoint)}, "
                       f"Area={ac.area:.3f}, "
                       f"COG={phase_to_string(ac.cog)}, "
                       f"R2={res.r2:.3f}")
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
