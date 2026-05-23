"""
melafit.utils: Utility Functions for Melatonin Data Processing and Analysis

This module provides support functions for reading, preprocessing, and
analyzing melatonin time series data, including waveform computation,
day profile averaging, and time/phase conversion utilities.

Data I/O Functions:
-------------------
- read_data : Load melatonin data from Excel spreadsheet
- prepare_part_data : Extract and preprocess single participant's data

Time Conversion:
----------------
- to_days        : Convert timestamps to float days since the Unix UTC epoch
- from_days      : Convert float days since the Unix UTC epoch to DatetimeIndex
- gen_time_range : Generate a time axis as float days since UTC epoch

Time Series Analysis:
---------------------
- day_profile : Compute averaged 24h profile from multi-day time series

Time/Phase Conversion:
----------------------
- time_to_phase : Convert time values to phase representation (0.0 to 1.0)
- phase_to_string : Format phase values as human-readable time strings (HH:MM)
- string_to_phase : Convert HH:MM time string back to phase representation
- phase_diff : Compute difference between two phases with proper wrapping

Parameter/Threshold Utilities:
------------------------------
- abs_threshold : Convert relative threshold to absolute threshold value
- params_to_string : Format curve fitting parameters as human-readable strings

Data Format:
------------
Input melatonin data should be an Excel file with columns:
- Participant : Study participant ID
- Date : Sample date
- Time : Sample time of day
- Mel : Melatonin concentration value

Notes:
------
- Times are internally stored as days (1.0 = 24 hours) or phases (1.0 = 24h)
- All time arithmetic properly handles modulo 24h wrapping
- Day profiles are computed by averaging across multiple circadian cycles
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import scipy.optimize as opt
from collections.abc import Mapping

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_days(timestamps: np.ndarray | pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """
    Convert timestamps to float days since the Unix UTC epoch (1970-01-01).

    The integer part is the day count and the fractional part is the fraction
    of the day. Timezone-naive input is assumed to be UTC; timezone-aware
    input is converted to UTC before the calculation.

    Parameters
    ----------
        timestamps : np.ndarray, pd.Series, or pd.DatetimeIndex
            Timestamps to convert (datetime64, Timestamp, or compatible)

    Returns
    -------
        days : np.ndarray of float
            Float days since 1970-01-01 UTC
    """
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ((ts - _EPOCH) / pd.Timedelta(days=1)).to_numpy()


def from_days(days: np.ndarray) -> pd.DatetimeIndex:
    """
    Convert float days since the Unix UTC epoch to a DatetimeIndex.

    Inverse of :func:`to_days`. Returns a timezone-aware DatetimeIndex (UTC).

    Parameters
    ----------
        days : np.ndarray of float
            Float days since 1970-01-01 UTC

    Returns
    -------
        timestamps : pd.DatetimeIndex
            UTC-aware timestamps corresponding to the input day values
    """
    return pd.DatetimeIndex(_EPOCH + pd.to_timedelta(days, unit='D'))


def read_data(data_pathname: str) -> pd.DataFrame:
    """
    Read data to be analyzed from an Excel spreadsheet.

    Column must be named as follows:
    * *Participant* for study participant ID
    * *Date* for dates of the respective samples
    * *Time* for sample timestamps
    * *Mel* for melatonin level values

    Parameters
    ----------
        data_pathname : str
            Pathname of the Excel spreadsheet file to read data from

    Returns
    -------
        data : pandas DataFrame
            Data for all participants read from the Excel table
    """

    data = pd.read_excel(data_pathname)

    data.Participant = data.Participant.astype(int, errors="ignore")
    data.Date = (pd.to_datetime(data.Date, dayfirst=True, errors="coerce")
                 .dt.date)
    data.Time = pd.to_datetime(data.Time.astype(str), errors="coerce").dt.time
    data.Mel = data.Mel.astype(float, errors="ignore")

    data["Timestamp"] = data.apply(
        lambda x: dt.datetime.combine(x.Date, x.Time), axis=1)

    return data


def prepare_part_data(data: pd.DataFrame,
                      participant: str | int) -> pd.DataFrame:
    """
    Prepare one participant's data for analysis.

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

    p_data = data.loc[data.Participant == participant].copy()
    p_data = p_data.drop(columns=['Date', 'Time'])

    idiff = np.diff(to_days(p_data.Timestamp.values)) < 0

    if any(idiff):
        ix = np.where(idiff)

        for i in ix:
            idx = p_data.index[i[0] + 1]
            p_data.loc[idx, 'Timestamp'] += pd.Timedelta(days=1)
            print(f"Corrected one timestamp for participant {participant}")

    return p_data


def gen_time_range(
    series: pd.Series | None = None,
    *,
    tmin: pd.Timestamp | None = None,
    tmax: pd.Timestamp | None = None,
    step: str | pd.Timedelta,
    full_day: bool = True,
) -> np.ndarray:
    """
    Generate a resampled time axis as float days since the Unix UTC epoch.

    The integer part of each value is the day count and the fractional part
    is the fraction of the day. The returned array is compatible with the
    waveform functions and plottable on matplotlib date axes.

    Parameters
    ----------
        series : pd.Series, optional
            Timestamp series used to infer tmin and/or tmax when not given
            explicitly
        tmin : pd.Timestamp, optional
            Start of the time range; overrides series.min() when given
        tmax : pd.Timestamp, optional
            End of the time range (inclusive); overrides series.max() when given
        step : str or pd.Timedelta
            Time step as a pandas offset string (e.g. ``"1min"``, ``"5min"``,
            ``"1h"``) or a :class:`pd.Timedelta`
        full_day : bool
            If True and (tmax - tmin) < 24 h, extend tmax to tmin + 24 h
            (defaults to True)

    Returns
    -------
        time_range : Numpy array of floats
            Time axis as float days since 1970-01-01 UTC
    """
    if series is not None:
        if tmin is None:
            tmin = series.min()
        if tmax is None:
            tmax = series.max()
    if tmin is None or tmax is None:
        raise ValueError("Provide series or both tmin and tmax")

    tmin_num = to_days([tmin])[0]
    tmax_num = to_days([tmax])[0]

    if full_day and (tmax_num - tmin_num) < 1.0:
        tmax_num = tmin_num + 1.0

    step_days = pd.Timedelta(step).total_seconds() / 86400
    return np.arange(tmin_num, tmax_num + 1.1 * step_days, step_days)



def day_profile(times: np.ndarray | pd.DatetimeIndex,
                values: np.ndarray,
                binsize: int = 60,
                double: bool = False,
                stderr: bool = False,
                repfirst: bool = False) -> tuple[pd.Series, pd.Series]:
    """
    Compute averaged day profile of a (quasi-)periodic time series.

    Parameters
    ----------
        times : np.ndarray or pandas DatetimeIndex
            Time stamps as a DatetimeIndex or as float days since the UTC
            epoch (as returned by :func:`gen_time_range`)
        values : numpy array
            Data values
        binsize : int
            Bin size in minutes (defaults to 60)
        double : bool
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

    if isinstance(times, np.ndarray):
        times = from_days(times)

    data = pd.Series(index=times, data=values)

    smpstr = str(binsize) + 'min'
    profile = data.shift(0.5, freq=smpstr).resample(smpstr).mean()
    profile = profile.groupby(profile.index.hour + profile.index.minute / 60)

    profile_mean = profile.mean()
    profile_std = profile.std()

    if stderr:
        profile_std = profile_std / np.sqrt(profile.count())

    profile = pd.DataFrame(data=pd.concat([profile_mean, profile_std],
                                          axis=1))

    if double:
        profile = pd.concat([profile, profile])

    if repfirst:
        profile = pd.concat([profile, pd.DataFrame(profile.iloc[0, :]).T])

    return profile.iloc[:, 0], profile.iloc[:, 1]


def time_to_phase(t: np.float64,
                  hours: bool = False) -> np.float64:
    """
    Convert time values to phase representation (0.0 to 1.0, 1.0 = 24h).

    Parameters
    ----------
        t : float
            Time value (in days or hours)
        hours : bool
            If True, time value is in hours and will be converted to phase by
            dividing by 24. If False, time value is in days (1.0 = 24h).
            Defaults to False.

    Returns
    -------
        phase : float
            Time as phase (0.0 to 1.0, 1.0 = 24h)
    """

    if hours:
        t = t / 24.0

    if t < 0:
        return np.ceil(-t) + t
    else:
        return t - np.floor(t)


def phase_to_string(phase: np.float64) -> str:
    """
    Convert phase representation of time (0.0 to 1.0) to HH:MM string.

    Negative phases produce a negative string (e.g. -0.25 -> "-06:00").

    Parameters
    ----------
        phase : float
            Time as phase (0.0 to 1.0, 1.0 = 24h)

    Returns
    -------
        string : str
            String representation of phase in HH:MM format

    See also
    --------
        :func:`string_to_phase` : Inverse conversion from HH:MM string to phase
    """

    if phase < 0:
        sign_str = "-"
        phase = -phase
    else:
        sign_str = ""

    td = pd.Timedelta(days=phase)
    td_in_seconds = td.total_seconds()

    hours, remainder = divmod(td_in_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    hours = int(hours)
    minutes = int(minutes)

    string = f"{sign_str}{hours:02d}:{minutes:02d}"

    return string


def string_to_phase(string: str) -> np.float64:
    """
    Convert HH:MM time string to phase representation (0.0 to 1.0, 1.0 = 24h).

    Negative strings (e.g. "-06:00") produce negative phase output.
    Inverse of :func:`phase_to_string`.

    Parameters
    ----------
        string : str
            Time string in HH:MM format (24-hour, no am/pm).
            Negative times are prefixed with "-" (e.g. "-02:30")

    Returns
    -------
        phase : float
            Time as phase. Negative input produces negative phase output.

    Raises
    ------
        ValueError
            If the string is not in HH:MM format

    See also
    --------
        :func:`phase_to_string` : Inverse conversion from phase to HH:MM string
    """

    string = string.strip()

    if string.startswith("-"):
        sign = -1.0
        string = string[1:]
    else:
        sign = 1.0

    try:
        parts = string.split(":")
        if len(parts) != 2:
            raise ValueError
        hours = int(parts[0])
        minutes = int(parts[1])
    except (ValueError, AttributeError):
        raise ValueError(
            f"Invalid time string '{string}': expected HH:MM format "
            f"(24-hour, no am/pm)")

    phase = sign * (hours + minutes / 60.0) / 24.0

    return phase


def abs_threshold(values: np.ndarray,
                  thresh_rel: np.float64) -> np.float64:
    """
    Compute absolute threshold from relative threshold.

    Parameters
    ----------
        values : Numpy array of floats
            Waveform values
        thresh_rel : float
            Relative threshold, fraction of range peak-to-baseline (0 to 1)

    Returns
    -------
        thresh_abs : float
            Absolute threshold
    """

    baseline = np.min(values)
    val_range = np.max(values) - baseline
    thresh_abs = baseline + thresh_rel * val_range

    return thresh_abs


def phase_diff(phase1: np.float64,
               phase2: np.float64) -> np.float64:
    """
    Compute difference between two phases (0.0 to 1.0, 1.0 = 24h).

    Parameters
    ----------
        phase1 : float
            First time as phase (0.0 to 1.0, 1.0 = 24h)
        phase2 : float
            Second time as phase (0.0 to 1.0, 1.0 = 24h)

    Returns
    -------
        dp : float
            Difference between the two phases, adjusted to be in the range
            -0.5 to 0.5 (i.e., -12h to 12h)
    """

    phase1 = time_to_phase(phase1, hours=False)
    phase2 = time_to_phase(phase2, hours=False)

    dp = phase1 - phase2

    if dp < -0.5:
        dp += 1.0
    elif dp > 0.5:
        dp -= 1.0

    return dp


def params_to_string(params: Mapping | np.ndarray, ndec: int = 3) -> str:
    """
    Convert curve fitting parameters to string.

    Parameters
    ----------
        params : dict or Numpy array of floats
            Curve fitting parameters
        ndec : int
            Number of decimal places to display (defaults to 3)

    Returns
    -------
        string : str
            String representation of curve fitting parameters
    """

    if isinstance(params, Mapping):
        param_strs = [f"{key}={value:.{ndec}f}"
                      for key, value in params.items()]
    else:
        param_strs = [f"p{i}={value:.{ndec}f}"
                      for i, value in enumerate(params)]

    string = ", ".join(param_strs)

    return string
