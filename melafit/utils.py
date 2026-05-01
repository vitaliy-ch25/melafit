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
                      participant: str | int) -> pd.DataFrame:
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
            Stop time (inclusive, 1.0 = 24 hours)
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

def day_profile(data: pd.Series,
                binsize: int = 60,
                double: bool = False, 
                stderr: bool = False,
                repfirst: bool = False)->tuple[pd.Series, pd.Series]:
    """
    Compute averaged day profile of a (quasi-)periodic time series

    Parameters
    ----------
        data : pandas Series
            Time series data
        binsize : int
            Bin size in minutes (defaults to 60)
        double: bool
            Prepare data for double plot (defaults to False)
        stderr : bool
            Compute standard errors per bin (defaults to False)
        repfirst : bool
            Add first bin at 00:00 to the end (defaults to False)

    Returns
    -------
        profile : tuple[pd.Series, pd.Series]
            Bin averages and standard errors with index in hours (0..24)
    """

    # Bin data, ensure centering of the data points around bin centers
    smpstr=str(binsize)+'min'
    profile = data.shift(0.5, freq=smpstr).resample(smpstr).mean()
    profile = profile.groupby(profile.index.hour + profile.index.minute/60)

    # Compute average profile and standard deviations for each bin
    profile_mean = profile.mean()
    profile_std = profile.std()
    
    # If standard errors requested, compute these from std's and bin counts
    if stderr:
        profile_std  = profile_std / np.sqrt(profile.count())

    # Concatenate results
    profile = pd.DataFrame(data=pd.concat([profile_mean, profile_std],
                                          axis=1))
    
    # Prepare data for double plot if requested
    if double:
        profile = profile.append([profile])
        
    # Add first bin at 00:00 to the end
    if repfirst:
        profile = pd.concat([profile, pd.DataFrame(profile.iloc[0,:]).T])
    
    # Split returned results up for maximum flexibility
    return profile.iloc[:,0], profile.iloc[:,1]

def phase_to_string(phase: np.float64) -> str:
    """
    Convert phase representation of time (0.0 to 1.0) to string

    Parameters
    ----------
        phase : float
            Time as phase (0.0 to 1.0, 1.0 = 24h)

    Returns
    -------
        string : str
            String representation of phase
    """

    td = pd.Timedelta(days=phase)
    td_in_seconds = td.total_seconds()

    hours, remainder = divmod(td_in_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    hours = int(hours)
    minutes = int(minutes)

    if hours < 10:
        hours = "0{}".format(hours)

    if minutes < 10:
        minutes = "0{}".format(minutes)

    string = "{}:{}".format(hours, minutes)        

    return string

def abs_threshold(values: np.ndarray[np.float64],
                  thresh_rel: np.float64) -> np.float64:
    """
    Compute absolute threshold from relative threshold

    Parameters
    ----------
        values : Numpy array of floats
            Waveform values

    Returns
    -------
        thresh_abs : float
            Absolute threshold
    """

    baseline = np.min(values)
    range = np.max(values) - baseline
    thresh_abs = baseline + thresh_rel * range

    return thresh_abs
