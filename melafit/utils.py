"""
melafit.utils: Utility Functions for Melatonin Data Processing and Analysis

This module provides support functions for reading, preprocessing, and
analyzing melatonin time series data, including waveform computation,
day profile averaging, and time/phase conversion utilities.

Data I/O Functions:
-------------------
- read_data : Load melatonin data from Excel spreadsheet
- prepare_part_data : Extract and preprocess single participant's data

Waveform Functions:
-------------------
- compute_wave : Generate a full 24h waveform curve at specified time resolution

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

Results Management:
-------------------
- ResultsCollector : Accumulate per-participant marker results from
    typed result objects and save to Excel

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

    p_data = data.loc[data.Participant == participant]

    base = p_data.Timestamp.min()
    diff = p_data.Timestamp - base
    p_data["Timedays"] = (diff.dt.total_seconds() / (24 * 60 * 60) +
                          base.hour / 24 +
                          base.minute / (24 * 60) +
                          base.second / (24 * 60 * 60))

    idiff = np.diff(p_data.Timedays) < 0

    if any(idiff):
        ix = np.where(idiff)

        for i in ix:
            idx = p_data.index[i[0] + 1]
            p_data.loc[idx, 'Timestamp'] += pd.Timedelta(days=1)
            p_data.loc[idx, 'Timedays'] += 1.0
            print(f"Corrected one timestamp for participant {participant}")

    return p_data


def compute_wave(tmin: np.float64,
                 tmax: np.float64,
                 dt_minutes: np.float64,
                 f: callable,
                 p: dict | np.ndarray,
                 full_wave: bool = True) -> np.ndarray:
    """
    Compute waveform resampled to given time resolution.

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
        p : Dictionary or Numpy array of floats
            Waveform parameter vector
        full_wave : bool
            If True and (tmax-tmin) < 1.0, tmax = tmin + 1.0 (defaults to
            True)

    Returns
    -------
        curve_val : Numpy array of floats
            Values of the waveform function for the respective time points
    """

    if full_wave and ((tmax - tmin) < 1.0):
        tmax = tmin + 1.0

    step = 1.0 / (dt_minutes * 24 * 60)
    time_curve = np.arange(tmin, tmax + 1.1 * step, step)
    curve_val = f(t=time_curve, p=p)

    return curve_val


def day_profile(times: pd.DatetimeIndex,
                values: np.ndarray,
                binsize: int = 60,
                double: bool = False,
                stderr: bool = False,
                repfirst: bool = False) -> tuple[pd.Series, pd.Series]:
    """
    Compute averaged day profile of a (quasi-)periodic time series.

    Parameters
    ----------
        times : pandas DatetimeIndex
            Time stamps
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


def params_to_string(params: dict | np.ndarray, ndec: int = 3) -> str:
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

    if isinstance(params, dict):
        param_strs = [f"{key}={value:.{ndec}f}"
                      for key, value in params.items()]
    else:
        param_strs = [f"p{i}={value:.{ndec}f}"
                      for i, value in enumerate(params)]

    string = ", ".join(param_strs)

    return string


class ResultsCollector:
    """
    Accumulates per-participant melatonin marker results and saves them to
    an Excel spreadsheet.

    The :meth:`add` method accepts any combination of
    :class:`~melafit.markers.AnalysisResult` subclass instances in any order.
    It calls :meth:`to_dict` on each and merges the resulting fields into a
    single row per participant. Missing fields appear as NaN in the output.
    Timing fields are stored as HH:MM strings in the Excel output.

    The caller is responsible for appending the waveform function name to
    the filename when saving.

    Examples
    --------
        collector = ResultsCollector()
        collector.add(meta, res, ampl, mid, ac)         # full profile
        collector.add(meta, res, mid)                    # DLMO only
        collector.save("./results/", "results_BSBCF")    # after loop

    See also
    --------
        :class:`melafit.markers.AnalysisResult` : Abstract base class for all
        result types
        :class:`melafit.fitting.FitResult` : Optimization result with
        AnalysisResult interface
    """

    def __init__(self):
        self._records = []

    def add(self, *args) -> None:
        """
        Add one analysis run to the collector.

        Accepts any combination of :class:`~melafit.markers.AnalysisResult`
        subclass instances in any order. Exactly one
        :class:`~melafit.markers.AnalysisInfo` is required.

        Parameters
        ----------
            *args : AnalysisResult
                Any combination of AnalysisResult subclass instances from
                one analysis run

        Raises
        ------
            TypeError
                If any argument is not a AnalysisResult subclass instance
            ValueError
                If no AnalysisInfo is provided among the arguments
        """

        # Local import to avoid circular dependency
        from melafit.markers import AnalysisResult, AnalysisInfo

        record = {}
        meta_found = False

        for obj in args:
            if not isinstance(obj, AnalysisResult):
                raise TypeError(
                    f"ResultsCollector.add() received unsupported type "
                    f"{type(obj).__name__}")
            if isinstance(obj, AnalysisInfo):
                meta_found = True
            record.update(obj.to_dict())

        if not meta_found:
            raise ValueError("ResultsCollector.add() requires an AnalysisInfo "
                             "instance among its arguments")

        self._records.append(record)

    def save(self, result_path: str, filename: str) -> None:
        """
        Save accumulated results to an Excel spreadsheet.

        Silently does nothing if no results have been added. Creates the
        result directory if it does not exist.

        Parameters
        ----------
            result_path : str
                Directory to save the results file to
            filename : str
                Filename without extension; '.xlsx' is appended automatically
        """

        if not self._records:
            return

        df = pd.DataFrame(self._records)
        df.set_index("participant", inplace=True)
        df.sort_index(inplace=True)

        os.makedirs(result_path, exist_ok=True)
        df.to_excel(os.path.join(result_path, filename + ".xlsx"))
