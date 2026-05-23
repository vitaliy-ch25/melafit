"""
melafit.markers: Circadian Phase Markers from Melatonin Data

This module provides functions to compute clinically relevant circadian phase
markers and rhythm characteristics from melatonin concentration curves.
Result types are defined in :mod:`melafit.results`.

Markers Computed:
-----------------
- Amplitude : Peak-to-baseline difference of the melatonin curve.
    Indicator of circadian rhythm strength/robustness.

- DLMOn/DLMOff (Dim Light Melatonin Onset/Offset) : Times when melatonin
    crosses a given threshold (absolute or relative). Key markers of
    circadian phase for clinical assessment.

- Midpoint : Time of melatonin level in the middle between DLMOn and DLMOff.
    Used to define melatonin phase angle.

- Area Under Curve : Total melatonin secretion over 24 hours. Indicator of
    total circulating melatonin quantity.

- Center of Gravity (COG) : Shape-based phase marker computed as the weighted
    average time of melatonin secretion. Used to define melatonin phase angle.
    More robust to noise and partial data than threshold-based markers
    including DLMOn/Off and midpoint. Coincides with midpoint for symmetric
    curves.

Functions:
----------
- amplitude : Compute peak-to-baseline amplitude
- midpoint : Compute DLMOn/DLMOff times and melatonin midpoint
- area_cog : Compute area under curve and center of gravity

Notes:
------
- All timing fields in result dataclasses are stored as phase values
  (0.0 to 1.0, where 1.0 = 24h)
- Threshold-based markers (DLMOn/Off) require at least 24h of data or a
  full 24h fitted curve

References:
-----------
- [Benloucif et al. '08](https://pmc.ncbi.nlm.nih.gov/articles/PMC2276833/)
- [Kolodyazhniy et al. '12](https://doi.org/10.3109/07420528.2012.700669)
"""

import numpy as np
import pandas as pd
from melafit.results import AmplitudeResult, MidpointResult, AreaCogResult
from melafit.utils import (day_profile, abs_threshold, time_to_phase,
                            phase_to_string)


def amplitude(values: np.ndarray) -> AmplitudeResult:
    """
    Peak-to-baseline amplitude of fitted waveform.

    Parameters
    ----------
        values : Numpy array of floats
            Waveform values

    Returns
    -------
        result : AmplitudeResult
            Wrapped amplitude and baseline values
    """

    baseline = np.min(values)
    return AmplitudeResult(amplitude=np.max(values) - baseline,
                           baseline=baseline)


def midpoint(times: np.ndarray | pd.DatetimeIndex,
             values: np.ndarray,
             threshold: np.float64,
             thresh_abs: bool = False) -> MidpointResult:
    """
    Compute melatonin midpoint, DLMOn and DLMOff times.

    NOTE: This function assumes that there is at least 24h of data. If
    this is not the case, the results may be inaccurate. When working
    with waveforms, make sure to generate a full 24h curve which is
    usually possible even with shorter raw data the curve was fitted to.

    Parameters
    ----------
        times : np.ndarray or pandas DatetimeIndex
            Datetime values as a DatetimeIndex or as float days since the
            UTC epoch (as returned by :func:`melafit.utils.gen_time_range`)
        values : Numpy array of floats
            Melatonin waveform values
        threshold : float
            Relative threshold, fraction of range peak-to-baseline (0 to 1),
            or absolute threshold value if thresh_abs=True
        thresh_abs : bool
            If True, the given threshold is absolute. Otherwise, the absolute
            threshold is computed from the given relative threshold and the
            range of values (defaults to False)

    Returns
    -------
        result : MidpointResult
            Dataclass containing dlmon, dlmoff, midpoint (all as phase
            values) and the absolute threshold used

    See also
    --------
        :func:`melafit.utils.gen_time_range` : Generate resampled time axis
    """

    d_profile = day_profile(times, values, binsize=1)[0]

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

    return MidpointResult(dlmon=time_on,
                          dlmoff=time_off,
                          midpoint=time_midpoint,
                          threshold=threshold)


def area_cog(times: np.ndarray | pd.DatetimeIndex,
             values: np.ndarray,
             baseline: np.float64 | None = None) -> AreaCogResult:
    """
    Area under the curve and center of gravity of melatonin waveform.

    Parameters
    ----------
        times : np.ndarray or pandas DatetimeIndex
            Datetime values as a DatetimeIndex or as float days since the
            UTC epoch (as returned by :func:`melafit.utils.gen_time_range`)
        values : Numpy array of floats
            Waveform values
        baseline : float or None
            Baseline for area computation. Equals to minimum of values if
            None is given (defaults to None)

    Returns
    -------
        result : AreaCogResult
            Dataclass containing area under the curve and center of gravity
            as phase (0.0 to 1.0, 1.0 = 24h)

    See also
    --------
        :func:`melafit.utils.gen_time_range` : Generate resampled time axis
    """

    if baseline is None:
        baseline = np.min(values)

    bin_minutes = 1

    d_profile = day_profile(times, values, binsize=bin_minutes)[0]

    times = d_profile.index.values / 24.0
    values = d_profile.values

    idx_on = np.argwhere((values[:-1] <= baseline) &
                         (values[1:] > baseline))[0][0]

    times = np.concatenate([times[idx_on:], 1.0 + times[:idx_on]])
    values = np.concatenate([values[idx_on:], values[:idx_on]]) - baseline

    area = np.sum(values)
    cog = np.dot(values, times) / area

    # Convert COG to phase (from 0.0 to 1.0, 1.0 = 24h)
    cog = time_to_phase(cog)

    # Normalize area by bin size in minutes
    area /= (24.0 * 60.0 / bin_minutes)

    return AreaCogResult(area=area, cog=cog)
