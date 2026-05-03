import numpy as np
import pandas as pd
from melafit.utils import day_profile, abs_threshold, time_to_phase

def amplitude(values: np.ndarray) -> np.float64:
    """
    Peak-to-baseline amplitude of fitted waveform

    Parameters
    ----------
        values : Numpy array of floats
            Waveform values

    Returns
    -------
        ampl : float
            Peak-to-baseline amplitude
    """

    return np.max(values) - np.min(values)


def midpoint(times: pd.DatetimeIndex,
             values: np.ndarray,
             threshold: np.float64,
             thresh_abs: bool = False
             ) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Compute melatonin midpoint, DLMOn and DLMOff times. NOTE: This function
    assumes that there is at least 24h of data. If this is not the case, the
    results may be inaccurate. When working with waveforms, make sure to
    generate a full 24h curve which is usually possible even with shorter
    raw data the curve was fitted to.

    Parameters
    ----------
        times : pandas DatetimeIndex
            Datetime values
        values : Numpy array of floats
            Melatonin waveform values
        threshold: float
            Relative threshold, fraction of range peak-to-baseline (0 to 1)
        thresh_abs: bool
            If True, the given threshold is absolute. Otherwise, the absolute
            threshold is computed from the given relative threshold and the
            range of values (defaults to False)

    Returns
    -------
        result : tuple[float, float, float, float]
            Melatonin midpoint, DLMOn and DLMOff times as phase (from 0.0 to
            1.0, 1.0 = 24h), and absolute threshold

    See also
    --------
         melafit.utils.compute_wave: Compute waveform resampled to given time
         resolution
    """

    data_series = pd.Series(index=times, data=values)        
    d_profile = day_profile(data_series, binsize=1)[0]

    if not thresh_abs:
        threshold = abs_threshold(values, threshold)

    idx_on = np.argwhere((d_profile.values[:-1] < threshold) &
                         (d_profile.values[1:] >= threshold))[0]
    idx_off = np.argwhere((d_profile.values[:-1] >= threshold) &
                          (d_profile.values[1:] < threshold))[0]

    time_on = d_profile.index.values[idx_on][0] / 24.0
    time_off = d_profile.index.values[idx_off][0] / 24.0

    if time_on > time_off:
        time_off += 1.0

    time_midpoint = 0.5 * (time_on + time_off)

    time_midpoint = time_to_phase(time_midpoint)
    time_on = time_to_phase(time_on)
    time_off = time_to_phase(time_off)

    return time_midpoint, time_on, time_off, threshold

def area_cog(times: pd.DatetimeIndex,
             values: np.ndarray,
             baseline: np.float64 | None = None
             ) -> tuple[np.float64, np.float64]:
    """
    Center of gravity of area under the curve

    Parameters
    ----------
        times : pandas DatetimeIndex
            Datetime values
        values : Numpy array of floats
            Waveform values
        baseline : float or None
            Baseline for area computation. Equals to minimum of values if
            None is given (default)

    Returns
    -------
        area : float
            Area under the curve
        cog : float
            Center of gravity of area under the curve as phase (from 0.0 to
            1.0, 1.0 = 24h)
    """

    if baseline is None:
        baseline = np.min(values)
    
    data_series = pd.Series(index=times, data=values)
    d_profile = day_profile(data_series, binsize=1)[0]

    times = d_profile.index.values / 24.0
    values = d_profile.values

    idx_on = np.argwhere((values[:-1] <= baseline) &
                         (values[1:] > baseline))[0][0]

    times = np.concatenate([times[idx_on:], 1.0 + times[:idx_on]])
    values = np.concatenate([values[idx_on:], values[:idx_on]])

    area = np.sum(values)
    cog = np.dot(values, times) / area

    # Convert COG to phase (from 0.0 to 1.0, 1.0 = 24h)
    cog = time_to_phase(cog)

    return area, cog