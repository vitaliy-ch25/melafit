"""
melafit.results: Structured Result Classes for Melatonin Analysis

This module defines all result types returned by the melafit analysis
pipeline and the :class:`ResultsCollector` utility for accumulating them.

Classes:
--------
- AnalysisRecord   : Abstract base class defining the to_dict() interface
- SessionInfo      : Identifying information about a data acquisition session
- AmplitudeResult  : Wrapper for the peak-to-baseline amplitude marker
- MidpointResult   : Wrapper for DLMOn, DLMOff, midpoint and threshold
- AreaCogResult    : Wrapper for area under curve and center of gravity
- FitResult        : Wrapper for a scipy optimization result; implements
    the Mapping protocol so it can be used directly wherever a parameter
    dict is expected (waveform functions, etc.)
- ResultsCollector : Accumulates per-participant results and saves them
    to an Excel spreadsheet

Notes:
------
- Timing fields in result dataclasses are stored as phase values
  (0.0 to 1.0, where 1.0 = 24h); :meth:`to_dict` converts them to
  HH:MM strings.
"""

import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict, InitVar
from melafit.utils import phase_to_string


class AnalysisRecord(ABC):
    """
    Abstract base class for all melafit analysis result types.

    All subclasses must implement :meth:`to_dict`, returning a dictionary
    suitable for tabular output (e.g. for use with
    :class:`melafit.utils.ResultsCollector`).
    """

    @abstractmethod
    def to_dict(self) -> dict:
        """Return result fields as a dictionary for tabular output."""
        ...


@dataclass
class SessionInfo(AnalysisRecord):
    """
    Identifying information about a data acquisition session.

    Parameters
    ----------
        p_data : pd.DataFrame
            Single-participant DataFrame as returned by
            :func:`melafit.utils.prepare_part_data`. ``participant``,
            ``start``, and ``end`` are derived from it automatically.

    Attributes
    ----------
        participant : int or str
            Participant identifier
        start : pd.Timestamp
            Start timestamp of the acquisition session
        end : pd.Timestamp
            End timestamp of the acquisition session
    """

    p_data: InitVar[pd.DataFrame]
    participant: int | str = field(init=False)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)

    def __post_init__(self, p_data: pd.DataFrame):
        self.participant = p_data.Participant.iloc[0]
        self.start = p_data.Timestamp.min()
        self.end = p_data.Timestamp.max()

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
class AmplitudeResult(AnalysisRecord):
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
class MidpointResult(AnalysisRecord):
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
class AreaCogResult(AnalysisRecord):
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


class FitResult(AnalysisRecord, Mapping):
    """
    Wrapper of optimization result implementing the :class:`AnalysisRecord`
    and :class:`collections.abc.Mapping` interfaces. Mapping support allows
    ``FitResult`` to be passed directly wherever a parameter dict is accepted
    (waveform functions, etc.). All
    standard scipy attributes (``x``, ``fun``, ``success``, ``nit``, etc.)
    are preserved in the field ``result`` of type
    :class:`scipy.optimize.OptimizeResult`.

    Attributes
    ----------
        wave_func : callable
            Melatonin wave approximation function for which the parameters
            were fitted
        result : scipy.optimize.OptimizeResult
            Result of the optimization procedure including fitted parameters
            in the field ``x``
        r2 : float
            R² goodness of fit of the optimized parameters against the data
            passed to :func:`melafit.fitting.fit`
    """

    def __init__(self, result: opt.OptimizeResult, wave_func: callable,
                 param_names: list | None, r2: np.float64 = float("nan")):
        self.wave_func = wave_func
        self.result = result
        self._param_names = param_names
        self.r2 = r2

    # ------------------------------------------------------------------
    # Mapping protocol — enables direct use as a parameter dict
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        if self._param_names is None:
            raise KeyError(key)
        try:
            return self.result.x[self._param_names.index(key)]
        except ValueError:
            raise KeyError(key)

    def __iter__(self):
        return iter(self._param_names or [])

    def __len__(self):
        return len(self._param_names or [])

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return fitted parameters, function name, and R² as a named dictionary.

        Returns
        -------
            d : dict
                ``{"func": name, name: value, ..., "r2": value}`` — function
                name first, then one entry per fitted parameter, then R².
                Parameter entries are empty if no parameter names are available
                (e.g. a custom function with array-only bounds).
        """
        d = {"func": self.wave_func.__name__}
        d.update(dict(self))
        d["r2"] = self.r2
        return d


class ResultsCollector:
    """
    Accumulates per-participant melatonin marker results and saves them to
    an Excel spreadsheet.

    The :meth:`add` method accepts any combination of
    :class:`AnalysisRecord` subclass instances in any order. It calls
    :meth:`to_dict` on each and merges the resulting fields into a single
    row per participant. Missing fields appear as NaN in the output.
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
        :class:`AnalysisRecord` : Abstract base class for all result types
        :class:`FitResult` : Optimization result with AnalysisRecord interface
    """

    def __init__(self):
        self._records = []

    def add(self, *args) -> None:
        """
        Add one analysis run to the collector.

        Accepts any combination of :class:`AnalysisRecord` subclass
        instances in any order. Exactly one :class:`SessionInfo` is
        required.

        Parameters
        ----------
            *args : AnalysisRecord
                Any combination of AnalysisRecord subclass instances from
                one analysis run

        Raises
        ------
            TypeError
                If any argument is not a AnalysisRecord subclass instance
            ValueError
                If no SessionInfo is provided among the arguments
        """

        record = {}
        meta_found = False

        for obj in args:
            if not isinstance(obj, AnalysisRecord):
                raise TypeError(
                    f"ResultsCollector.add() received unsupported type "
                    f"{type(obj).__name__}")
            if isinstance(obj, SessionInfo):
                meta_found = True
            record.update(obj.to_dict())

        if not meta_found:
            raise ValueError("ResultsCollector.add() requires a SessionInfo "
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
