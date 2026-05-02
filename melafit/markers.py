import numpy as np
import pandas as pd
from melafit.utils import day_profile, abs_threshold

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


def midpoint(times: np.ndarray,
             values: np.ndarray,
             thresh_rel: np.ndarray
             ) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Compute melatonin midpoint, DLMOn and DLMOff times

    Parameters
    ----------
        times : Numpy array of floats
            Time values (1.0 = 24h)
        values : Numpy array of floats
            Melatonin waveform values
        thresh_rel: float
            Relative threshold, fraction of range peak-to-baseline (0 to 1)

    Returns
    -------
        result : tuple[float, float, float, float]
            Melatonin midpoint, DLMOn and DLMOff times (from 0.0 to 1.0,
            1.0 = 24h), and absolute threshold
    """

    resampled_data = pd.Series(index=times, data=values)        
    d_profile = day_profile(resampled_data, binsize=1)[0]

    thresh_abs = abs_threshold(values, thresh_rel)

    idx_on = np.argwhere((d_profile.values[:-1] < thresh_abs) &
                         (d_profile.values[1:] >= thresh_abs))[0]
    idx_off = np.argwhere((d_profile.values[:-1] >= thresh_abs) &
                          (d_profile.values[1:] < thresh_abs))[0]

    time_on = d_profile.index.values[idx_on][0] / 24.0
    time_off = d_profile.index.values[idx_off][0] / 24.0

    if time_on > time_off:
        time_off += 1.0

    time_midpoint = 0.5 * (time_on + time_off)

    time_midpoint = np.modf(time_midpoint)[0]
    time_on = np.modf(time_on)[0]
    time_off = np.modf(time_off)[0]

    return time_midpoint, time_on, time_off, thresh_abs