"""
melafit.markers: Circadian Phase Markers from Melatonin Data

This module provides functions to compute clinically relevant circadian phase
markers and rhythm characteristics from melatonin concentration curves,
together with structured result dataclasses.

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

Result Dataclasses:
-------------------
- MelaResult : Abstract base class defining the to_dict() interface for
    all result types
- AnalysisInfo : Identifying information about an analysis (participant, start,
    waveform function name, goodness of fit)
- AmplitudeResult : Wrapper for amplitude marker
- MidpointResult : Wrapper for midpoint, DLMOn, DLMOff and threshold
- AreaCogResult : Wrapper for area under curve and center of gravity

All result dataclasses inherit from MelaResult and implement `to_dict()`,
returning a dictionary suitable for tabular output. Timing fields (phase
values) are formatted as HH:MM strings; other fields as native types.

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from melafit.utils import (day_profile, abs_threshold, time_to_phase,
                            phase_to_string)


class MelaResult(ABC):
    """
    Abstract base class for all melafit result types.

    All subclasses must implement :meth:`to_dict`, returning a dictionary
    suitable for tabular output (e.g. for use with
    :class:`melafit.utils.ResultsCollector`).
    """

    @abstractmethod
    def to_dict(self) -> dict:
        """Return result fields as a dictionary for tabular output."""
        ...


@dataclass
class AnalysisInfo(MelaResult):
    """
    Identifying information about a single melatonin analysis.

    Attributes
    ----------
        participant : int or str
            Participant identifier
        start : pd.Timestamp
            Start timestamp of the analysis session
        func : str
            Name of the waveform function used for fitting
        r2 : float
            R² goodness of fit (defaults to NaN if not provided, which is
            convenient when r2 is not computed, e.g. for partial-data DLMO
            detection)
    """

    participant: int | str
    start: pd.Timestamp
    func: str
    r2: np.float64 = float("nan")

    def to_dict(self) -> dict:
        """
        Return all fields as a flat dictionary.

        Returns
        -------
            d : dict
                All fields in their native types.
        """
        return asdict(self)


@dataclass
class AmplitudeResult(MelaResult):
    """
    Result of peak-to-baseline amplitude computation.

    Attributes
    ----------
        amplitude : float
            Peak-to-baseline amplitude of the waveform
    """

    amplitude: np.float64

    def to_dict(self) -> dict:
        """
        Return all fields as a flat dictionary.

        Returns
        -------
            d : dict
                All fields in their native types.
        """
        return asdict(self)


@dataclass
class MidpointResult(MelaResult):
    """
    Result of midpoint, DLMOn and DLMOff computation.

    Timing fields are stored as phase values (0.0 to 1.0, 1.0 = 24h).
    Use :meth:`to_dict` to obtain HH:MM string representations.

    Attributes
    ----------
        dlmon : float
            Dim light melatonin onset time as phase
        dlmoff : float
            Dim light melatonin offset time as phase
        midpoint : float
            Melatonin midpoint time as phase
        threshold : float
            Absolute threshold value used for the computation
    """

    dlmon: np.float64
    dlmoff: np.float64
    midpoint: np.float64
    threshold: np.float64

    def to_dict(self) -> dict:
        """
        Return timing fields as HH:MM string representations.

        Returns
        -------
            d : dict
                Dictionary with keys 'dlmon', 'dlmoff', 'midpoint' mapped
                to their HH:MM string representations. 'threshold' is
                included unchanged as a float.
        """
        return {
            "dlmon": phase_to_string(self.dlmon),
            "dlmoff": phase_to_string(self.dlmoff),
            "midpoint": phase_to_string(self.midpoint),
            "threshold": self.threshold,
        }


@dataclass
class AreaCogResult(MelaResult):
    """
    Result of area under curve and center of gravity computation.

    Attributes
    ----------
        area : float
            Area under the curve
        cog : float
            Center of gravity as phase (0.0 to 1.0, 1.0 = 24h)
    """

    area: np.float64
    cog: np.float64

    def to_dict(self) -> dict:
        """
        Return timing fields as HH:MM string representations.

        Returns
        -------
            d : dict
                Dictionary with key 'cog' mapped to its HH:MM string
                representation. 'area' is included unchanged as a float.
        """
        return {
            "area": self.area,
            "cog": phase_to_string(self.cog),
        }


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
            Wrapped amplitude value
    """

    return AmplitudeResult(amplitude=np.max(values) - np.min(values))


def midpoint(times: pd.DatetimeIndex,
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
        times : pandas DatetimeIndex
            Datetime values
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
        :func:`melafit.utils.compute_wave` : Compute waveform resampled to
        given time resolution
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


def area_cog(times: pd.DatetimeIndex,
             values: np.ndarray,
             baseline: np.float64 | None = None) -> AreaCogResult:
    """
    Area under the curve and center of gravity of melatonin waveform.

    Parameters
    ----------
        times : pandas DatetimeIndex
            Datetime values
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
        :func:`melafit.utils.compute_wave` : Compute waveform resampled to
        given time resolution
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
