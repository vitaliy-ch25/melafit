import numpy as np
import pandas as pd
import datetime as dt
from melafit.fitting import *

def read_data(data_pathname: str) -> pd.DataFrame:

    # Read data from Excel spreadsheet
    data = pd.read_excel(data_pathname)

    # Enforce correct data types
    data.Participant = data.Participant.astype(int, errors="ignore")
    data.Date = pd.to_datetime(data.Date, dayfirst=True, errors="coerce").dt.date
    data.Time = pd.to_datetime(data.Time.astype(str), errors="coerce").dt.time
    data.Mel = data.Mel.astype(float, errors="ignore")

    # Add combined datetime timestamp
    data["Timestamp"] = data.apply(lambda x: dt.datetime.combine(x.Date, x.Time), axis=1)

    return data

def prepare_part_data(data: pd.DataFrame,
                      participant: str | np.int64) -> pd.DataFrame:
    """
    Prepare one participant's data for analysis

    Parameters
    ----------
        data : pandas DataFrame
            All participants' data
        participant : string or integer
            Participant's identifier

    Returns
    -------
        p_data : pandas DataFrame
            Prepared data for one participant
    """

    # Select participant's data
    p_data = data.loc[data.Participant==participant]

    # Extract cumulative time in days
    base = p_data.Timestamp.min()
    diff = p_data.Timestamp - base
    p_data["Timedays"] = (diff.dt.total_seconds() / (24*60*60) +
                            base.hour / 24 + 
                            base.minute / (24*60) +
                            base.second / (24*60*60))

    print(p_data)

    # Check and fix errors in timestamps
    idiff = np.diff(p_data.Timedays) < 0

    if any(idiff):
        ix = np.where(idiff)
        
        for i in ix:
            row = p_data.iloc[i[0]+1]
            row.Timestamp += pd.Timedelta(days=1)
            row.Timedays += 1.0
            p_data.iloc[i[0]+1] = row
            print(f"Corrected one timestamp for participant {participant}")

    return p_data

def compute_wave(tmin: np.float64,
                 tmax: np.float64,
                 dt_minutes: np.float64,
                 f: callable,
                 p: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Compute waveform resampled to given time resolution

    Parameters
    ----------
        tmin : float
            Start time (1.0 = 24 hours)
        tmax : float
            Stop time (inclusive, 1.0 = 24 hours))
        dt_minutes : float
            Time increment in minutes
        f : callable
            Waveform function
        p : Numpy array of floats
            Waveform parameter vector

    Returns
    -------
        curve_val : Numpy array of floats
            Values of the waveform function for the respective time points
    """

    step = 1.0 / (dt_minutes * 24 * 60)
    time_curve = np.arange(tmin, tmax + 1.1 * step, step)
    curve_val = f(t=time_curve, p=p)

    return curve_val