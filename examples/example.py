import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bcf, sbcf, bbcf, bsbcf, fit, rsquared
from melafit.markers import amplitude, midpoint
from melafit.utils import read_data, prepare_part_data, compute_wave, phase_to_string

data_path = "./data/"
result_path = "./results/"
result_filename = "results" # without file extension, '.xlsx' will be added

# Read data from Excel spreadsheet
data = read_data(data_path + "dummy_data.xlsx")

# Define functions to try for fitting to melatonin data
mel_funcs = [bcf, sbcf, bbcf, bsbcf]

popup_figures = True # If figures should appear on the screen
dt_minutes = 1.0 # Time resolution of resampled fitted curve in minutes
thresh_dlmo = 0.25 # Relative threshold for DLMOn & DLMOff

# Create folder for analysis results
os.makedirs(result_path, exist_ok=True)

# List of all participants
participants = np.unique(data.Participant)

# Loop over functions to fit to melatonin data
for mel_func in mel_funcs:

    # Empty dataframe for results
    results = pd.DataFrame()

    for participant in participants:

        try:
            # Prepare one participant's data for analysis
            p_data = prepare_part_data(data, participant)
            print(p_data)

            # Fit curve to raw data
            res = fit(p_data.Timedays, p_data.Mel, mel_func)

            # Compute goodness of fit with fitted curve and raw data
            fitted_curve = mel_func(t=p_data.Timedays, p=res.x)
            r2 = rsquared(p_data.Mel, fitted_curve)

            # Compute fitted curve resampled to one minute resolution, res.x contains the fitted func parameters
            resampled_curve = compute_wave(p_data.Timedays.min(), p_data.Timedays.max(), dt_minutes, mel_func, res.x)
            
            # Generate pandas timestamps for plotting of the fitted curve
            resampled_time = pd.date_range(p_data.Timestamp.min(), periods=len(resampled_curve), freq=pd.Timedelta(minutes=dt_minutes))

            # Find melatonin onset, offset and midpoint as phase (from 0.0 to 1.0, 1.0 = 24h)
            midpt, dlmon, dlmoff, thresh_abs = midpoint(resampled_time, resampled_curve, thresh_dlmo)

            # Convert phase to string representation of time as HH:MM
            dlmon_str = phase_to_string(dlmon)
            dlmoff_str = phase_to_string(dlmoff)
            midpt_str = phase_to_string(midpt)

            # Find amplitude relative to baseline        
            ampl = amplitude(resampled_curve)

            # Print fitted parameters
            print(res.x)

            # Save results
            results = pd.concat([results, pd.DataFrame(
                [[participant, p_data.Timestamp.min(), res.x,
                ampl, dlmon_str, dlmoff_str, midpt_str, r2]],
                columns=["Participant", "Start", "Curve_Params",
                        "Amplitude", "DLMOn", "DLMOff", "Midpoint", "R2"]
            )], ignore_index=True)

            res_str = (f"Start: {p_data.Timestamp.min()}, DLMOn={dlmon_str}, " +
                    f"DLMOff={dlmoff_str}, Midpoint={midpt_str}, R2={r2:.3f}")
            
            # Print markers and goodness of fit
            print(res_str)

            # Visualize results
            plt.close("all")
            plt.figure(figsize=(12, 5))
            plt.scatter(p_data.Timestamp, p_data.Mel, c='b') # Plot raw data
            plt.plot(resampled_time, resampled_curve, 'g') # Plot fitted curve
            plt.plot(resampled_time, thresh_abs * np.ones(resampled_time.shape), 'r')
            plt.xlabel("Time, hh:mm")
            plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
            plt.ylabel("Concentration, pg/ml")
            plt.title(res_str)
            plt.legend(["Melatonin data", f"{mel_func.__name__.upper()} curve", "Threshold"])
            plt.savefig(result_path + f"mel_data_{participant}_{mel_func.__name__.upper()}.png")
            
            if popup_figures:
                plt.pause(0.01)

        except Exception as err:
            print(f"Error processing data for participant {participant}: {err}")

    # Save results to Excel spreadsheet
    results.set_index("Participant", inplace=True)
    results.sort_index(inplace=True)
    results.to_excel(result_path + result_filename + f"_{mel_func.__name__.upper()}.xlsx")