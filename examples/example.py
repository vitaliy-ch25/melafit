import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import dates
from melafit.fitting import bsbcf, fit, rsquared

data_path = "./data/"
result_path = "./results/"
result_filename = "results.xlsx"

popup_figures = True

# Create folder for analysis results
os.makedirs(result_path, exist_ok=True)

# Read data from Excel spreadsheet
data = pd.read_excel(data_path + "dummy_data.xlsx")

# Enforce correct data types
data.Patient = data.Patient.astype(int, errors="ignore")
data.Date = pd.to_datetime(data.Date, dayfirst=True, errors="coerce").dt.date
data.Time = pd.to_datetime(data.Time.astype(str), errors="coerce").dt.time
data.Mel = data.Mel.astype(float, errors="ignore")

# Add combined datetime timestamp
data["Timestamp"] = data.apply(lambda x: dt.datetime.combine(x.Date, x.Time), axis=1)

patients = np.unique(data.Patient)
results = pd.DataFrame()

for patient in patients:

    try:

        # Select patient data
        pat_data = data.loc[data.Patient==patient]

        # Extract cumulative time in days
        base = pat_data.Timestamp.min()
        diff = pat_data.Timestamp - base
        pat_data["Cumtime"] = (diff.dt.total_seconds() / (24*60*60) +
                               base.hour / 24 + 
                               base.minute / (24*60) +
                               base.second / (24*60*60))

        print(pat_data)

        # Check and fix errors in timestamps
        idiff = np.diff(pat_data.Cumtime) < 0

        if any(idiff):
            ix = np.where(idiff)
            
            for i in ix:
                row = pat_data.iloc[i[0]+1]
                row.Timestamp += pd.Timedelta(days=1)
                row.Cumtime += 1.0
                pat_data.iloc[i[0]+1] = row
                print(f"Corrected one timestamp for patient {patient}")

        # Prepare data for curve fitting and visualization of results        
        step = 1.0 / (24*60) # 1 minute
        step_curve = pd.Timedelta(1, "minute")
        mel_time = np.arange(pat_data.Cumtime.min(), pat_data.Cumtime.max() + 1.1 * step, step)

        time_fit = pat_data.Cumtime
        data_fit = pat_data.Mel

        # Fit BSBCF curve to raw data
        res = fit(time_fit, data_fit)

        print(res.x)

        # Compute BSBCF curve resampled to one minute resolution
        mel_curve = bsbcf(t=mel_time, phi=res.x[0], b=res.x[1], H=res.x[2], c=res.x[3], v=res.x[4], m=res.x[5])
        mel_curve_time = np.arange(pat_data.Timestamp.min(), pat_data.Timestamp.max() + 2 * step_curve, step_curve)
        mel_curve_time = mel_curve_time[0:len(mel_curve)]

        fitted_curve = bsbcf(t=time_fit, phi=res.x[0], b=res.x[1], H=res.x[2], c=res.x[3], v=res.x[4], m=res.x[5])
        
        # Compute goodness of fit with fitted curve and raw data
        r2 = rsquared(data_fit, fitted_curve)
        
        print(f"R2={r2:.3f}")

        # Save results
        results = pd.concat([results, pd.DataFrame(
            [[patient, data.Timestamp[0], res.x[0], res.x[1], res.x[2],
              res.x[3], res.x[4], res.x[5], r2]],
            columns=["SubjID", "Start", "Phase", "Baseline", "Height",
                     "Width", "Skewness", "Bimodality", "R2"]
        )], ignore_index=True)
        # Note: the phase parameter is the parameter 'phi' of the fitted BSBCF
        # curve. For circadian phase, other derived measures must be used,
        # e.g. the midpoint between melatonin onset and offset times or the
        # center of gravity of the fitted curve, as e.g. in
        # Gabel et al. (2017) [https://doi.org/10.1038/s41598-017-07060-8]

        # Visualize results
        plt.close("all")
        plt.figure(figsize=(12, 5))
        plt.scatter(pat_data.Timestamp, pat_data.Mel, c='b')
        plt.plot(mel_curve_time, mel_curve, 'g-')
        plt.xlabel("Time, hh:mm")
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        plt.ylabel("Concentration, pg/ml")
        plt.title(f"Melatonin data and fitted BSBCF curve, R2={r2:.3f}, start: {data.Timestamp[0]}")
        plt.legend(["Melatonin data","BSBCF curve"])
        plt.savefig(result_path + f"mel_data_{patient}.png")
        
        if popup_figures:
            plt.pause(0.01)

    except Exception:
        print(f"Error processing data for patient {patient}")

# Save results to Excel spreadsheet
results.set_index("SubjID", inplace=True)
results.sort_index(inplace=True)
results.to_excel(result_path + result_filename)