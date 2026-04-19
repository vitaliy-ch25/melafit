import os
import pandas as pd
import scipy.optimize as opt
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import dates

data_path = "/mnt/c/Daten/MelDel/Data/"
result_path = "./results/"
result_filename = "result.csv"

popup_figures = False

os.makedirs(result_path, exist_ok=True)
data = pd.read_excel(data_path + "MelDel_edited.xlsx")
with open(result_path + result_filename, "w") as file:
    pass

def bsbcf(t: np.ndarray[np.float64],
          phi: np.float64,
          b: np.float64,
          H: np.float64,
          c: np.float64,
          v: np.float64,
          m: np.float64) -> np.ndarray[np.float64]:
    """
    Bimodal skewed baseline cosine function (Van Someren & Nagtegaal, 2007)

    Parameters
    ----------
        t : Numpy array of floats
            Cumulative time in days
        phi : float
            phase [-0.5 0.5], 0.5 = 12h
        b : float
            Baseline >=0
        H : float
            Height >0
        c : float
            Width [-1,1)
        v : float
            Skewness [-0.5 0.5]
        m : float
            Bimodality [0 1)

    Returns
    -------
        bsbcf_val : Numpy array of floats
            Values of the BSBCF function for the respective time points
    """
    
    phi = 2 * np.pi * phi
    t = 2 * np.pi * t

    bsbcf_val = b + H / (2 * (1 - c)) * (
        np.cos(t - phi + v * np.cos(t - phi)) + 
        m * np.cos(2 * t - 2 * phi - np.pi) - c + 
        abs(np.cos(t - phi + v * np.cos(t - phi)) + 
            m * np.cos(2 * t - 2 * phi - np.pi) - c))

    return bsbcf_val

def cost(p: np.ndarray[np.float64],
         t: np.ndarray[np.float64],
         y: np.ndarray[np.float64]) -> np.float64:
    """
    Cost function for melatonin fitting, penalizes the trivial solution when
    all model values = 0 (Gabel et al., 2017)

    Parameters
    ----------
        p : Numpy array of floats
            Function parameter vector
        t : Numpy array of floats
            X-values for curve fitting (time)
        y: Numpy array of floats
            Y-values for curve fitting (melatonin levels)

    Returns
    -------
        val : float
            Value of the cost function
    """

    phi = p[0]
    b = p[1]
    H = p[2]
    c = p[3]
    v = p[4]
    m = p[5]

    y_ = bsbcf(t, phi, b, H, c, v, m)

    return np.nanmean(np.square(y - y_)) / np.var(y_)

def rsquared(Y: np.ndarray[np.float64],
             y: np.ndarray[np.float64]) -> np.float64:
    
    """
    R2 goodness of fit

    Parameters
    ----------
        Y : Numpy array of floats
            Reference values
        y : Numpy array of floats
            Fitted values

    Returns
    -------
        r2 : float
            R2 value
    """

    err = Y - y
    Y_ = Y - np.nanmean(Y)
    r2 = 1 - np.nansum(np.square(err)) / np.nansum(np.square(Y_))

    return r2

def fit(time_fit: np.ndarray[np.float64],
        data_fit: np.ndarray[np.float64],
        cost: callable=cost) -> np.ndarray[np.float64]:
    """
    Melatonin data fitting routine

    Parameters
    ----------
        time_fit : Numpy array of floats
            X-values for curve fitting (time)
        data_fit : Numpy array of floats
            Y-values for curve fitting (melatonin levels)
        cost : callable
            Cost function for curve fitting (defaults to `cost`)

    Returns
    -------
        res : Numpy array of floats
            Parameters of the fitted function
    """

    minx = data_fit.min()
    maxx = data_fit.max()
    range = (maxx - minx)

    # Initial guess for BSBCF parameters
    param0 = [
        0, # phi
        minx, # b
        (maxx-minx), # H
        0, # c
        0, # v
        0 # m
        ]

    # Lower bounds for BSBCF parameters
    lb = [
        -0.5, # phi
        minx, # b
        0.5 * range, # H
        -1, # c
        -1, # v
        0 # m
    ]

    # Upper bounds for BSBCF parameters
    ub = [
        0.5, # phi
        maxx, # b
        2 * range, # H
        1 - 1e-6, # c
        1, # v
        1 - 1e-6 # m
    ]

    bounds = opt.Bounds(lb, ub)
    res = opt.minimize(fun=cost,
                       args=(time_fit, data_fit),
                       x0=param0,
                       bounds=bounds)

    return res

# Enforce correct data types
data.Patient = data.Patient.astype(int, errors="ignore")
data.Num = data.Num.astype(int, errors="ignore")
data.Date = pd.to_datetime(data.Date, dayfirst=True, errors="coerce").dt.date
data.Time = pd.to_datetime(data.Time.astype(str), errors="coerce").dt.time
data.Mel = data.Mel.astype(float, errors="ignore")

# Add combined datetime timestamp
data["Timestamp"] = data.apply(lambda x: dt.datetime.combine(x.Date, x.Time), axis=1)

patients = np.unique(data.Patient)

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

        idiff = np.diff(pat_data.Cumtime) < 0

        if any(idiff):
            ix = np.where(idiff)
            
            for i in ix:
                row = pat_data.iloc[i[0]+1]
                row.Timestamp += pd.Timedelta(days=1)
                row.Cumtime += 1.0
                pat_data.iloc[i[0]+1] = row
                print(f"Corrected one timestamp for patient {patient}")
                
        step = 1.0 / (24*60) # 1 minute
        step_curve = pd.Timedelta(1, "minute")
        mel_time = np.arange(pat_data.Cumtime.min(), pat_data.Cumtime.max() + 1.1 * step, step)

        time_fit = pat_data.Cumtime
        data_fit = pat_data.Mel

        res = fit(time_fit, data_fit)

        print(res.x)

        mel_curve = bsbcf(t=mel_time, phi=res.x[0], b=res.x[1], H=res.x[2], c=res.x[3], v=res.x[4], m=res.x[5])
        mel_curve_time = np.arange(pat_data.Timestamp.min(), pat_data.Timestamp.max() + 2 * step_curve, step_curve)
        mel_curve_time = mel_curve_time[0:len(mel_curve)]

        fitted_curve = bsbcf(t=time_fit, phi=res.x[0], b=res.x[1], H=res.x[2], c=res.x[3], v=res.x[4], m=res.x[5])
        r2 = rsquared(data_fit, fitted_curve)
        
        print(f"R2={r2:.3f}")

        with open(result_path + result_filename, "a") as file:
            file.write(f"SubjID={patient}, {data.Timestamp[0]}, " +
                       f"amplitude={res.x[2]}, width={res.x[3]}, " + 
                       f"skewness={res.x[4]}, bimodality={res.x[5]}, " +
                       f"R2={r2}\n")

        plt.close("all")
        plt.figure(figsize=(24, 8))
        plt.plot(mel_curve_time, mel_curve)
        plt.plot(pat_data.Timestamp, pat_data.Mel)
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