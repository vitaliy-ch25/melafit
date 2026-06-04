"""
Tests for melafit package: fitting, markers and utils modules.

Run from the repo root with:
    python -m unittest tests/test_melafit.py
or with pytest:
    pytest tests/test_melafit.py
"""

import unittest
import numpy as np
import pandas as pd
import datetime as dt
import scipy.optimize as opt
from collections.abc import Mapping

from melafit.fitting import (bcf, sbcf, bbcf, bsbcf, cost, rsquared,
                              func_defaults, fit,
                              params_to_array, array_to_params,
                              _resolve_params, BCF_PARAM_NAMES,
                              SBCF_PARAM_NAMES, BBCF_PARAM_NAMES,
                              BSBCF_PARAM_NAMES, BUILTIN_PARAM_NAMES)
from melafit.markers import amplitude, dlmo, midpoint, area_cog
from melafit.results import (FitResult, AnalysisRecord, SessionInfo,
                              AmplitudeResult, DLMOResult, MidpointResult,
                              AreaCogResult, ResultsCollector)
from melafit.utils import (read_data, prepare_part_data, to_days, from_days,
                            gen_time_range, day_profile, time_to_phase,
                            phase_to_string, string_to_phase, abs_threshold,
                            phase_diff, params_to_string)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Canonical parameters for a plausible melatonin waveform
BSBCF_PARAMS_ARRAY = np.array([0.875, 2.0, 80.0, 0.5, 0.3, 0.1])
BCF_PARAMS_ARRAY   = np.array([0.875, 2.0, 80.0, 0.5])
SBCF_PARAMS_ARRAY  = np.array([0.875, 2.0, 80.0, 0.5, 0.3])
BBCF_PARAMS_ARRAY  = np.array([0.875, 2.0, 80.0, 0.5, 0.1])

BSBCF_PARAMS_DICT = dict(zip(BSBCF_PARAM_NAMES, BSBCF_PARAMS_ARRAY))
BCF_PARAMS_DICT   = dict(zip(BCF_PARAM_NAMES,   BCF_PARAMS_ARRAY))
SBCF_PARAMS_DICT  = dict(zip(SBCF_PARAM_NAMES,  SBCF_PARAMS_ARRAY))
BBCF_PARAMS_DICT  = dict(zip(BBCF_PARAM_NAMES,  BBCF_PARAMS_ARRAY))

# Time array covering one and two full days at 1-minute resolution
T = np.linspace(0, 1, 1440, endpoint=False)
T2 = np.linspace(0, 2, 2 * 1440, endpoint=False)

# Full-profile and DLMO dummy data paths (from repo root)
DUMMY_DATA_FULL = "./data/dummy_data_full.xlsx"

# ---------------------------------------------------------------------------
# fitting.py — _resolve_params tests
# ---------------------------------------------------------------------------

class TestResolveParams(unittest.TestCase):
    """Tests for _resolve_params()."""

    def test_array_passed_through_unchanged(self):
        p = np.array([1.0, 2.0, 3.0])
        result = _resolve_params(p)
        np.testing.assert_array_equal(result, p)

    def test_dict_converted_to_array(self):
        p = {"phi": 0.875, "b": 2.0, "H": 80.0, "c": 0.5}
        result = _resolve_params(p)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([0.875, 2.0, 80.0, 0.5]))

    def test_dict_values_in_insertion_order(self):
        """Values must come out in the order keys were inserted."""
        p = {"phi": 0.1, "b": 0.2, "H": 0.3, "c": 0.4}
        result = _resolve_params(p)
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))

    def test_result_is_ndarray(self):
        result = _resolve_params(BSBCF_PARAMS_DICT)
        self.assertIsInstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# fitting.py — params_to_array / array_to_params tests
# ---------------------------------------------------------------------------

class TestParamConversion(unittest.TestCase):
    """Tests for params_to_array() and array_to_params()."""

    def test_array_to_params_bsbcf(self):
        result = array_to_params(BSBCF_PARAMS_ARRAY, bsbcf)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result.keys()), BSBCF_PARAM_NAMES)
        for key, val in zip(BSBCF_PARAM_NAMES, BSBCF_PARAMS_ARRAY):
            self.assertAlmostEqual(result[key], val)

    def test_array_to_params_bcf(self):
        result = array_to_params(BCF_PARAMS_ARRAY, bcf)
        self.assertEqual(list(result.keys()), BCF_PARAM_NAMES)

    def test_array_to_params_sbcf(self):
        result = array_to_params(SBCF_PARAMS_ARRAY, sbcf)
        self.assertEqual(list(result.keys()), SBCF_PARAM_NAMES)

    def test_array_to_params_bbcf(self):
        result = array_to_params(BBCF_PARAMS_ARRAY, bbcf)
        self.assertEqual(list(result.keys()), BBCF_PARAM_NAMES)

    def test_array_to_params_unknown_function_raises(self):
        def my_func(t, p): return t
        with self.assertRaises(ValueError):
            array_to_params(np.array([1.0, 2.0]), my_func)

    def test_params_to_array_bsbcf(self):
        result = params_to_array(BSBCF_PARAMS_DICT)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, BSBCF_PARAMS_ARRAY)

    def test_params_to_array_bcf(self):
        result = params_to_array(BCF_PARAMS_DICT)
        np.testing.assert_array_equal(result, BCF_PARAMS_ARRAY)

    def test_roundtrip_array_to_dict_to_array(self):
        """Converting array -> dict -> array should give the original array."""
        d = array_to_params(BSBCF_PARAMS_ARRAY, bsbcf)
        result = params_to_array(d)
        np.testing.assert_array_almost_equal(result, BSBCF_PARAMS_ARRAY)

    def test_roundtrip_dict_to_array_to_dict(self):
        """Converting dict -> array -> dict should give the original dict."""
        arr = params_to_array(BSBCF_PARAMS_DICT)
        result = array_to_params(arr, bsbcf)
        self.assertEqual(result, BSBCF_PARAMS_DICT)


# ---------------------------------------------------------------------------
# fitting.py — waveform function tests
# ---------------------------------------------------------------------------

class TestWaveformFunctions(unittest.TestCase):
    """Tests for bcf, sbcf, bbcf, bsbcf — both array and dict input."""

    def _check_output(self, func, params_array, params_dict):
        """Helper: check shape, type, non-negativity, array/dict equivalence."""
        result_array = func(t=T, p=params_array)
        result_dict  = func(t=T, p=params_dict)

        self.assertEqual(result_array.shape, T.shape)
        self.assertIsInstance(result_array, np.ndarray)
        self.assertTrue(np.all(result_array >= 0),
                        f"{func.__name__} produced negative values")
        np.testing.assert_array_equal(result_array, result_dict)

    def test_bcf(self):
        self._check_output(bcf, BCF_PARAMS_ARRAY, BCF_PARAMS_DICT)

    def test_sbcf(self):
        self._check_output(sbcf, SBCF_PARAMS_ARRAY, SBCF_PARAMS_DICT)

    def test_bbcf(self):
        self._check_output(bbcf, BBCF_PARAMS_ARRAY, BBCF_PARAMS_DICT)

    def test_bsbcf(self):
        self._check_output(bsbcf, BSBCF_PARAMS_ARRAY, BSBCF_PARAMS_DICT)

    def test_bcf_flat_when_H_zero(self):
        """With H=0, bcf should return b everywhere."""
        p = np.array([0.0, 5.0, 0.0, 0.0])
        np.testing.assert_allclose(bcf(t=T, p=p), 5.0, atol=1e-10)

    def test_bsbcf_peak_near_phi(self):
        """Peak of bsbcf should occur near phi (within +-3h = +-0.125 days)."""
        result = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        peak_phase = T[np.argmax(result)]
        phi = BSBCF_PARAMS_ARRAY[0]
        diff = min(abs(peak_phase - phi), 1.0 - abs(peak_phase - phi))
        self.assertLess(diff, 0.125)

    def test_all_functions_return_finite(self):
        for func, params in [(bcf,   BCF_PARAMS_ARRAY),
                             (sbcf,  SBCF_PARAMS_ARRAY),
                             (bbcf,  BBCF_PARAMS_ARRAY),
                             (bsbcf, BSBCF_PARAMS_ARRAY)]:
            result = func(t=T, p=params)
            self.assertTrue(np.all(np.isfinite(result)),
                            f"{func.__name__} returned non-finite values")


# ---------------------------------------------------------------------------
# fitting.py — cost function tests
# ---------------------------------------------------------------------------

class TestCostFunction(unittest.TestCase):
    """Tests for cost()."""

    def setUp(self):
        self.y = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)

    def test_perfect_fit_is_near_zero(self):
        val = cost(p=BSBCF_PARAMS_ARRAY, t=T, y=self.y, f=bsbcf)
        self.assertLess(val, 1e-6)

    def test_cost_is_non_negative(self):
        noisy_y = self.y + np.random.normal(0, 5, size=self.y.shape)
        val = cost(p=BSBCF_PARAMS_ARRAY, t=T, y=noisy_y, f=bsbcf)
        self.assertGreaterEqual(val, 0.0)

    def test_eps_prevents_division_by_zero(self):
        flat_params = np.array([0.0, 5.0, 0.0, 0.0])
        try:
            val = cost(p=flat_params, t=T, y=self.y, f=bcf)
            self.assertTrue(np.isfinite(val))
        except ZeroDivisionError:
            self.fail("cost() raised ZeroDivisionError with flat prediction")

    def test_custom_eps_via_cost_p(self):
        """cost_p dict should override eps."""

        # Take a degenerate bad fit (all parameters zero)
        val_default = cost(p=0*BSBCF_PARAMS_ARRAY, t=T, y=self.y, f=bsbcf)
        val_custom  = cost(p=0*BSBCF_PARAMS_ARRAY, t=T, y=self.y, f=bsbcf,
                           cost_p={"eps": 1.0})

        # Check that the cost value is different with the custom eps
        # (since the default eps is very small, the cost should be much smaller
        # with eps=1.0)
        self.assertNotAlmostEqual(val_default, val_custom)

    def test_none_cost_p_same_as_empty_dict(self):
        val_none  = cost(p=BSBCF_PARAMS_ARRAY, t=T, y=self.y, f=bsbcf,
                         cost_p=None)
        val_empty = cost(p=BSBCF_PARAMS_ARRAY, t=T, y=self.y, f=bsbcf,
                         cost_p={})
        self.assertAlmostEqual(val_none, val_empty)

    def test_worse_with_wrong_params(self):
        wrong = np.array([0.0, 2.0, 80.0, 0.5, 0.3, 0.1])
        val_correct = cost(p=BSBCF_PARAMS_ARRAY, t=T, y=self.y, f=bsbcf)
        val_wrong   = cost(p=wrong,               t=T, y=self.y, f=bsbcf)
        self.assertLess(val_correct, val_wrong)


# ---------------------------------------------------------------------------
# fitting.py — rsquared tests
# ---------------------------------------------------------------------------

class TestRSquared(unittest.TestCase):
    """Tests for rsquared()."""

    def test_perfect_fit_gives_one(self):
        y = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.assertAlmostEqual(rsquared(Y=y, y=y), 1.0, places=10)

    def test_r2_decreases_with_noise(self):
        y = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.assertLess(rsquared(Y=y, y=y + np.random.normal(0, 10, y.shape)),
                        1.0)

    def test_r2_is_finite(self):
        y = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.assertTrue(np.isfinite(rsquared(Y=y, y=y)))


# ---------------------------------------------------------------------------
# fitting.py — func_defaults tests
# ---------------------------------------------------------------------------

class TestFuncDefaults(unittest.TestCase):
    """Tests for func_defaults() — returns dicts."""

    def setUp(self):
        self.data = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)

    def _check_defaults(self, func, param_names):
        p0, lb, ub = func_defaults(self.data, func)
        for d in [p0, lb, ub]:
            self.assertIsInstance(d, dict)
        self.assertEqual(list(p0.keys()), param_names)
        self.assertEqual(list(lb.keys()), param_names)
        self.assertEqual(list(ub.keys()), param_names)
        for key in param_names:
            self.assertLess(lb[key], ub[key])
            self.assertGreaterEqual(p0[key], lb[key])
            self.assertLessEqual(p0[key], ub[key])

    def test_bcf_defaults(self):
        self._check_defaults(bcf, BCF_PARAM_NAMES)

    def test_sbcf_defaults(self):
        self._check_defaults(sbcf, SBCF_PARAM_NAMES)

    def test_bbcf_defaults(self):
        self._check_defaults(bbcf, BBCF_PARAM_NAMES)

    def test_bsbcf_defaults(self):
        self._check_defaults(bsbcf, BSBCF_PARAM_NAMES)

    def test_unknown_function_raises(self):
        def my_func(t, p): return t
        with self.assertRaises(NotImplementedError):
            func_defaults(self.data, my_func)


# ---------------------------------------------------------------------------
# fitting.py — fit() tests
# ---------------------------------------------------------------------------

class TestFit(unittest.TestCase):
    """Tests for fit()."""

    def setUp(self):
        self.y = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.t = T

    def test_returns_fitresult(self):
        res = fit(self.t, self.y)
        self.assertIsInstance(res, FitResult)
        self.assertIsInstance(res.result, opt.OptimizeResult)
        self.assertIsInstance(res, AnalysisRecord)
        self.assertIsInstance(res, Mapping)
        self.assertIn('x', res.result)

    def test_returns_param_dict(self):
        res = fit(self.t, self.y, f=bsbcf)
        p = res.to_dict()
        self.assertIsInstance(p, dict)
        self.assertEqual(list(p.keys()), ["func"] + BSBCF_PARAM_NAMES + ["r2"])

    def test_param_dict_consistent_with_array(self):
        res = fit(self.t, self.y, f=bsbcf)
        for i, key in enumerate(BSBCF_PARAM_NAMES):
            self.assertAlmostEqual(res.to_dict()[key], res.result.x[i], 
                                   places=10)

    def test_param_dict_correct_keys_all_functions(self):
        for func, names, params in [
            (bcf,   BCF_PARAM_NAMES,   BCF_PARAMS_ARRAY),
            (sbcf,  SBCF_PARAM_NAMES,  SBCF_PARAMS_ARRAY),
            (bbcf,  BBCF_PARAM_NAMES,  BBCF_PARAMS_ARRAY),
            (bsbcf, BSBCF_PARAM_NAMES, BSBCF_PARAMS_ARRAY),
        ]:
            y = func(t=T, p=params)
            res = fit(T, y, f=func)
            self.assertEqual(list(res.to_dict().keys()), ["func"] + names + ["r2"])

    def test_bsbcf_converges(self):
        res = fit(self.t, self.y, f=bsbcf)
        self.assertTrue(res.result.success or res.result.fun < 0.01)

    def test_bcf_converges(self):
        res = fit(T, bcf(t=T, p=BCF_PARAMS_ARRAY), f=bcf)
        self.assertTrue(res.result.success or res.result.fun < 0.01)

    def test_sbcf_converges(self):
        res = fit(T, sbcf(t=T, p=SBCF_PARAMS_ARRAY), f=sbcf)
        self.assertTrue(res.result.success or res.result.fun < 0.01)

    def test_bbcf_converges(self):
        res = fit(T, bbcf(t=T, p=BBCF_PARAMS_ARRAY), f=bbcf)
        self.assertTrue(res.result.success or res.result.fun < 0.01)

    def test_custom_p0_lb_ub_as_dicts(self):
        p0, lb, ub = func_defaults(self.y, bsbcf)
        res = fit(self.t, self.y, f=bsbcf, p0=p0, lb=lb, ub=ub)
        self.assertIsNotNone(res)

    def test_custom_p0_lb_ub_as_arrays(self):
        p0 = params_to_array(BSBCF_PARAMS_DICT)
        lb = np.array([-0.5, 0.0, 30.0, -1.0, -1.0, 0.0])
        ub = np.array([ 0.5, 5.0, 160.0, 1-1e-6, 1.0, 1-1e-6])
        res = fit(self.t, self.y, f=bsbcf, p0=p0, lb=lb, ub=ub)
        self.assertIsNotNone(res)

    def test_custom_params_not_overwritten_for_builtin_functions(self):
        p0 = {"phi": 0.875, "b": 2.0, "H": 80.0, 
              "c": 0.5, "v": 0.3, "m": 0.1}
        lb = {"phi": 0.870, "b": 1.9, "H": 79.0,
              "c": 0.49, "v": 0.29, "m": 0.09}
        ub = {"phi": 0.880, "b": 2.1, "H": 81.0,
              "c": 0.51, "v": 0.31, "m": 0.11}

        res = fit(self.t, self.y, f=bsbcf, p0=p0, lb=lb, ub=ub)
        self.assertTrue(res.result.success or res.result.fun < 0.01)
        # Verify the result stays within the tight custom bounds
        p = res.to_dict()
        for key in BSBCF_PARAM_NAMES:
            self.assertGreaterEqual(p[key], lb[key] - 1e-10)
            self.assertLessEqual(p[key], ub[key] + 1e-10)

    def test_fitresult_to_dict(self):
        res = fit(self.t, self.y, f=bsbcf)
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("func", d)
        self.assertIsInstance(d["func"], str)
        self.assertEqual(d["func"], "bsbcf")
        for key in BSBCF_PARAM_NAMES:
            self.assertIn(key, d.keys())
            self.assertIsInstance(d[key], float)
        self.assertIn("r2", d)
        self.assertIsInstance(d["r2"], float)
        self.assertTrue(np.isfinite(d["r2"]))

    def test_fitresult_r2_is_finite(self):
        res = fit(self.t, self.y, f=bsbcf)
        self.assertTrue(hasattr(res, "r2"))
        self.assertTrue(np.isfinite(res.r2))
        self.assertLessEqual(res.r2, 1.0)

    def test_fitresult_usable_as_wave_param(self):
        res = fit(self.t, self.y, f=bsbcf)
        # FitResult (Mapping over params) should be passable wherever a param
        # dict is accepted; dict(res) gives the plain param dict equivalent
        curve_direct = bsbcf(t=self.t, p=res)
        curve_dict   = bsbcf(t=self.t, p=dict(res))
        np.testing.assert_array_almost_equal(curve_direct, curve_dict)
        t = gen_time_range(tmin=pd.Timestamp("2024-01-01"), tmax=pd.Timestamp("2024-01-02"), step="1min")
        wave_direct = bsbcf(t=t, p=res)
        wave_dict   = bsbcf(t=t, p=dict(res))
        np.testing.assert_array_almost_equal(wave_direct, wave_dict)

    def test_cost_p_passed_through(self):
        res = fit(self.t, self.y, f=bsbcf, cost_p={"eps": 1e-6})
        self.assertIsInstance(res, FitResult)

    def test_custom_waveform_function_requires_manual_bounds(self):
        def cosine_wave(t, p):
            p = _resolve_params(p)
            return p[0] + p[1] * np.cos(2 * np.pi * t)

        with self.assertRaises(ValueError):
            fit(self.t,
                cosine_wave(self.t, p=np.array([1.0, 1.0])),
                f=cosine_wave)

    def test_custom_waveform_function_fits_with_manual_bounds(self):
        def cosine_wave(t, p):
            p = _resolve_params(p)
            return p[0] + p[1] * np.cos(2 * np.pi * t)

        true_params = {"offset": 2.0, "amplitude": 3.0}
        y = cosine_wave(self.t, true_params)
        p0 = {"offset": 1.0, "amplitude": 1.0}
        lb = {"offset": 0.0, "amplitude": 0.0}
        ub = {"offset": 5.0, "amplitude": 5.0}

        res = fit(self.t, y, f=cosine_wave, p0=p0, lb=lb, ub=ub)
        self.assertTrue(res.result.success or res.result.fun < 1e-6)
        p = res.to_dict()
        self.assertIsInstance(p, dict)
        self.assertEqual(list(p.keys()), ["func", "offset", "amplitude", "r2"])
        self.assertAlmostEqual(p["offset"], true_params["offset"], places=3)
        self.assertAlmostEqual(p["amplitude"],
                               true_params["amplitude"], places=3)

    def test_with_real_data(self):
        """fit() should converge on real dummy data."""
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timestamp, p_data.Mel, f=bsbcf)
        self.assertTrue(res.result.success or res.result.fun < 1.0)
        p = res.to_dict()
        self.assertIsInstance(p, dict)
        self.assertEqual(list(p.keys()), ["func"] + BSBCF_PARAM_NAMES + ["r2"])
        self.assertTrue(np.isfinite(res.r2))


# ---------------------------------------------------------------------------
# fitting.py + markers.py — Series input tests
# ---------------------------------------------------------------------------

class TestSeriesInput(unittest.TestCase):
    """np.ndarray and pd.Series (and pd.DatetimeIndex for times) are interchangeable."""

    def setUp(self):
        n = 1440
        self.t_arr = np.linspace(0, 1, n, endpoint=False)
        self.y_arr = bsbcf(t=self.t_arr, p=BSBCF_PARAMS_ARRAY)
        self.t_ser = pd.Series(self.t_arr)
        self.y_ser = pd.Series(self.y_arr)
        self.dt_idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        self.dt_ser = pd.Series(self.dt_idx)

    def test_fit_float_series_times(self):
        res = fit(self.t_ser, self.y_arr, f=bsbcf)
        self.assertIsInstance(res, FitResult)

    def test_fit_datetime_series_times(self):
        res = fit(self.dt_ser, self.y_arr, f=bsbcf)
        self.assertIsInstance(res, FitResult)

    def test_fit_datetimeindex_times(self):
        res = fit(self.dt_idx, self.y_arr, f=bsbcf)
        self.assertIsInstance(res, FitResult)

    def test_fit_series_values(self):
        res = fit(self.t_arr, self.y_ser, f=bsbcf)
        self.assertIsInstance(res, FitResult)

    def test_fit_both_series(self):
        res = fit(self.t_ser, self.y_ser, f=bsbcf)
        self.assertIsInstance(res, FitResult)

    def test_fit_series_matches_array(self):
        res_arr = fit(self.t_arr, self.y_arr, f=bsbcf)
        res_ser = fit(self.t_ser, self.y_ser, f=bsbcf)
        np.testing.assert_array_almost_equal(res_arr.result.x, res_ser.result.x)

    def test_amplitude_series_values(self):
        result = amplitude(self.y_ser)
        self.assertIsInstance(result, AmplitudeResult)
        self.assertAlmostEqual(result.amplitude,
                               amplitude(self.y_arr).amplitude)

    def test_dlmo_series_values(self):
        result = dlmo(self.dt_idx, self.y_ser, 0.25)
        self.assertIsInstance(result, DLMOResult)
        self.assertAlmostEqual(result.dlmo,
                               dlmo(self.dt_idx, self.y_arr, 0.25).dlmo)

    def test_dlmo_datetime_series_times(self):
        result = dlmo(self.dt_ser, self.y_arr, 0.25)
        self.assertIsInstance(result, DLMOResult)
        self.assertAlmostEqual(result.dlmo,
                               dlmo(self.dt_idx, self.y_arr, 0.25).dlmo)

    def test_midpoint_series_values(self):
        result = midpoint(self.dt_idx, self.y_ser, 0.25)
        self.assertIsInstance(result, MidpointResult)

    def test_area_cog_series_values(self):
        result = area_cog(self.dt_idx, self.y_ser)
        self.assertIsInstance(result, AreaCogResult)

    def test_day_profile_series_values(self):
        mean, std = day_profile(self.dt_idx, self.y_ser)
        self.assertIsInstance(mean, pd.Series)

    def test_day_profile_datetime_series_times(self):
        mean, std = day_profile(self.dt_ser, self.y_arr)
        self.assertIsInstance(mean, pd.Series)

    def test_fit_with_dataframe_columns(self):
        """The motivating use case: mf.fit(p_data.Timestamp, p_data.Mel, ...)"""
        df = pd.DataFrame({"Timestamp": self.dt_idx, "Mel": self.y_arr})
        res = fit(df.Timestamp, df.Mel, f=bsbcf)
        self.assertIsInstance(res, FitResult)


# ---------------------------------------------------------------------------
# markers.py — amplitude tests
# ---------------------------------------------------------------------------

class TestAmplitude(unittest.TestCase):
    """Tests for amplitude()."""

    def test_returns_amplitude_result(self):
        result = amplitude(np.array([2.0, 5.0, 10.0, 3.0]))
        self.assertIsInstance(result, AmplitudeResult)

    def test_known_amplitude(self):
        result = amplitude(np.array([2.0, 5.0, 10.0, 3.0]))
        self.assertAlmostEqual(result.amplitude, 8.0)
        self.assertAlmostEqual(result.baseline, 2.0)

    def test_flat_signal_is_zero(self):
        result = amplitude(np.full(100, 5.0))
        self.assertAlmostEqual(result.amplitude, 0.0)
        self.assertAlmostEqual(result.baseline, 5.0)

    def test_amplitude_on_waveform(self):
        result = amplitude(bsbcf(t=T, p=BSBCF_PARAMS_ARRAY))
        self.assertGreater(result.amplitude, 0.0)
        self.assertTrue(np.isfinite(result.amplitude))

    def test_to_dict_returns_dict(self):
        result = amplitude(np.array([2.0, 5.0, 10.0, 3.0]))
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("amplitude", d)
        self.assertAlmostEqual(d["amplitude"], 8.0)
        self.assertIn("baseline", d)
        self.assertAlmostEqual(d["baseline"], 2.0)


# ---------------------------------------------------------------------------
# results.py — DLMOResult tests
# ---------------------------------------------------------------------------

class TestDLMOResult(unittest.TestCase):
    """Tests for DLMOResult dataclass."""

    def test_construction(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        self.assertAlmostEqual(result.dlmo, 0.875)
        self.assertAlmostEqual(result.threshold, 10.0)

    def test_is_analysis_record(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        self.assertIsInstance(result, AnalysisRecord)

    def test_str_contains_dlmo(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        s = str(result)
        self.assertIn("DLMO", s)
        self.assertIn(phase_to_string(0.875), s)

    def test_to_dict_returns_dict(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("dlmo", d)
        self.assertIn("threshold", d)

    def test_to_dict_dlmo_is_string(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        d = result.to_dict()
        self.assertIsInstance(d["dlmo"], str)

    def test_to_dict_threshold_is_float(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        d = result.to_dict()
        self.assertIsInstance(d["threshold"], float)
        self.assertAlmostEqual(d["threshold"], 10.0)

    def test_to_dict_match_phase_to_string(self):
        result = DLMOResult(dlmo=0.875, threshold=10.0)
        d = result.to_dict()
        self.assertEqual(d["dlmo"], phase_to_string(result.dlmo))


# ---------------------------------------------------------------------------
# markers.py — dlmo tests
# ---------------------------------------------------------------------------

class TestDLMO(unittest.TestCase):
    """Tests for dlmo()."""

    def setUp(self):
        self.values = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(T),
            freq=pd.Timedelta(minutes=1))

    def test_returns_dlmo_result(self):
        result = dlmo(self.times, self.values, 0.25)
        self.assertIsInstance(result, DLMOResult)

    def test_dlmo_in_range(self):
        result = dlmo(self.times, self.values, 0.25)
        self.assertGreaterEqual(result.dlmo, 0.0)
        self.assertLess(result.dlmo, 1.0)

    def test_relative_threshold_converted_correctly(self):
        result = dlmo(self.times, self.values, 0.25)
        self.assertAlmostEqual(result.threshold,
                               abs_threshold(self.values, 0.25))

    def test_absolute_threshold_mode(self):
        result = dlmo(self.times, self.values, 10.0, thresh_abs=True)
        self.assertAlmostEqual(result.threshold, 10.0)

    def test_to_dict_returns_dict(self):
        result = dlmo(self.times, self.values, 0.25)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        for key in ["dlmo", "threshold"]:
            self.assertIn(key, d)
        self.assertIsInstance(d["dlmo"], str)
        self.assertIsInstance(d["threshold"], float)

    def test_to_dict_match_phase_to_string(self):
        result = dlmo(self.times, self.values, 0.25)
        d = result.to_dict()
        self.assertEqual(d["dlmo"], phase_to_string(result.dlmo))

    def test_threshold_never_crossed_raises(self):
        with self.assertRaises(ValueError):
            dlmo(self.times, self.values, 1e9, thresh_abs=True)

    def test_agrees_with_midpoint_dlmon(self):
        """dlmo() onset should equal midpoint() dlmon for the same waveform."""
        r_dlmo = dlmo(self.times, self.values, 0.25)
        r_mid  = midpoint(self.times, self.values, 0.25)
        self.assertAlmostEqual(r_dlmo.dlmo, r_mid.dlmon, places=4)

    def test_with_real_data(self):
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timestamp, p_data.Mel, f=sbcf)
        tmin, tmax = p_data.Timestamp.min(), p_data.Timestamp.max()
        t = gen_time_range(tmin=tmin, tmax=tmax, step="1min")
        curve = sbcf(t=t, p=res)
        result = dlmo(t, curve, 0.25)
        self.assertGreaterEqual(result.dlmo, 0.0)
        self.assertLess(result.dlmo, 1.0)


# ---------------------------------------------------------------------------
# markers.py — midpoint tests
# ---------------------------------------------------------------------------

class TestMidpoint(unittest.TestCase):
    """Tests for midpoint()."""

    def setUp(self):
        self.values = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(T),
            freq=pd.Timedelta(minutes=1))

    def test_returns_midpoint_result(self):
        result = midpoint(self.times, self.values, 0.25)
        self.assertIsInstance(result, MidpointResult)

    def test_all_phases_in_range(self):
        result = midpoint(self.times, self.values, 0.25)
        for val in [result.midpoint, result.dlmon, result.dlmoff]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)

    def test_relative_threshold_converted_correctly(self):
        result = midpoint(self.times, self.values, 0.25)
        self.assertAlmostEqual(result.threshold,
                               abs_threshold(self.values, 0.25))

    def test_absolute_threshold_mode(self):
        result = midpoint(self.times, self.values, 10.0, thresh_abs=True)
        self.assertAlmostEqual(result.threshold, 10.0)

    def test_to_dict_returns_dict(self):
        result = midpoint(self.times, self.values, 0.25)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        for key in ["dlmon", "dlmoff", "midpoint"]:
            self.assertIn(key, d)
            self.assertIsInstance(d[key], str)
        self.assertIn("threshold", d)
        self.assertIsInstance(d["threshold"], float)

    def test_to_dict_match_phase_to_string(self):
        result = midpoint(self.times, self.values, 0.25)
        d = result.to_dict()
        self.assertEqual(d["dlmon"], phase_to_string(result.dlmon))
        self.assertEqual(d["dlmoff"], phase_to_string(result.dlmoff))
        self.assertEqual(d["midpoint"], phase_to_string(result.midpoint))

    def test_threshold_never_crossed_raises(self):
        with self.assertRaises(ValueError):
            midpoint(self.times, self.values, 1e9, thresh_abs=True)

    def test_with_real_data(self):
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timestamp, p_data.Mel, f=bsbcf)
        tmin, tmax = p_data.Timestamp.min(), p_data.Timestamp.max()
        t = gen_time_range(tmin=tmin, tmax=tmax, step="1min")
        curve = bsbcf(t=t, p=res)
        result = midpoint(t, curve, 0.25)
        for val in [result.midpoint, result.dlmon, result.dlmoff]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)


# ---------------------------------------------------------------------------
# markers.py — area_cog tests
# ---------------------------------------------------------------------------

class TestAreaCog(unittest.TestCase):
    """Tests for area_cog()."""

    def setUp(self):
        self.values = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(T),
            freq=pd.Timedelta(minutes=1))

    def test_returns_area_cog_result(self):
        result = area_cog(self.times, self.values)
        self.assertIsInstance(result, AreaCogResult)

    def test_area_is_positive(self):
        result = area_cog(self.times, self.values)
        self.assertGreater(result.area, 0.0)

    def test_cog_in_range(self):
        result = area_cog(self.times, self.values)
        self.assertGreaterEqual(result.cog, 0.0)
        self.assertLess(result.cog, 1.0)

    def test_custom_baseline_changes_area(self):
        r_default = area_cog(self.times, self.values)
        r_custom  = area_cog(self.times, self.values,
                             baseline=min(self.values) + 1.0)
        self.assertNotAlmostEqual(r_default.area, r_custom.area)

    def test_to_dict_returns_dict(self):
        result = area_cog(self.times, self.values)
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("area", d)
        self.assertIn("cog", d)
        self.assertIsInstance(d["area"], float)
        self.assertIsInstance(d["cog"], str)

    def test_to_dict_match_phase_to_string(self):
        result = area_cog(self.times, self.values)
        d = result.to_dict()
        self.assertEqual(d["cog"], phase_to_string(result.cog))

    def test_baseline_never_crossed_raises(self):
        with self.assertRaises(ValueError):
            area_cog(self.times, self.values, baseline=1e9)

    def test_zero_area_raises(self):
        # Alternating +1/-1 over 1440 points: has an upward crossing at
        # baseline=0 but the positive and negative contributions cancel exactly.
        vals = np.where(np.arange(len(self.times)) % 2 == 0, 1.0, -1.0)
        with self.assertRaises(ValueError):
            area_cog(self.times, vals, baseline=0.0)

    def test_with_real_data(self):
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timestamp, p_data.Mel, f=bsbcf)
        tmin, tmax = p_data.Timestamp.min(), p_data.Timestamp.max()
        t = gen_time_range(tmin=tmin, tmax=tmax, step="1min")
        curve = bsbcf(t=t, p=res)
        result = area_cog(t, curve)
        self.assertGreater(result.area, 0.0)
        self.assertGreaterEqual(result.cog, 0.0)
        self.assertLess(result.cog, 1.0)


# ---------------------------------------------------------------------------
# markers.py — markers computed directly from raw data (no curve fitting)
# ---------------------------------------------------------------------------

class TestMarkersFromRawData(unittest.TestCase):
    """
    Confirm that all markers can be computed directly from sparse raw
    participant data without fitting a curve first.

    The dummy file contains 3-hourly samples (~18 points per participant),
    which exercises the day_profile linear-interpolation path internally
    used by dlmo(), midpoint(), and area_cog().
    """

    def setUp(self):
        data = read_data(DUMMY_DATA_FULL)
        participant = np.unique(data.Participant)[0]
        p_data = prepare_part_data(data, participant)
        self.times = p_data.Timestamp
        self.values = p_data.Mel

    def test_amplitude_from_raw_data(self):
        result = amplitude(self.values)
        self.assertIsInstance(result, AmplitudeResult)
        self.assertGreater(result.amplitude, 0.0)
        self.assertTrue(np.isfinite(result.amplitude))
        self.assertTrue(np.isfinite(result.baseline))

    def test_dlmo_from_raw_data(self):
        result = dlmo(self.times, self.values, 0.25)
        self.assertIsInstance(result, DLMOResult)
        self.assertGreaterEqual(result.dlmo, 0.0)
        self.assertLess(result.dlmo, 1.0)
        self.assertTrue(np.isfinite(result.threshold))

    def test_midpoint_from_raw_data(self):
        result = midpoint(self.times, self.values, 0.25)
        self.assertIsInstance(result, MidpointResult)
        for val in [result.dlmon, result.dlmoff, result.midpoint]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)
        self.assertTrue(np.isfinite(result.threshold))

    def test_area_cog_from_raw_data(self):
        result = area_cog(self.times, self.values)
        self.assertIsInstance(result, AreaCogResult)
        self.assertGreater(result.area, 0.0)
        self.assertGreaterEqual(result.cog, 0.0)
        self.assertLess(result.cog, 1.0)

    def test_to_dict_works_for_all_markers_from_raw_data(self):
        """to_dict() should work on results computed directly from raw data."""
        r_amp  = amplitude(self.values)
        r_dlmo = dlmo(self.times, self.values, 0.25)
        r_mid  = midpoint(self.times, self.values, 0.25)
        r_ac   = area_cog(self.times, self.values)
        self.assertIsInstance(r_amp.to_dict(), dict)
        self.assertIsInstance(r_dlmo.to_dict(), dict)
        self.assertIsInstance(r_mid.to_dict(), dict)
        self.assertIsInstance(r_ac.to_dict(), dict)

    def test_all_markers_all_participants_from_raw_data(self):
        """Every marker should succeed for every participant in the dummy file."""
        data = read_data(DUMMY_DATA_FULL)
        for participant in np.unique(data.Participant):
            p_data = prepare_part_data(data, participant)
            times  = p_data.Timestamp
            values = p_data.Mel
            with self.subTest(participant=participant):
                r_amp = amplitude(values)
                self.assertGreater(r_amp.amplitude, 0.0)
                r_dlmo = dlmo(times, values, 0.25)
                self.assertIsInstance(r_dlmo, DLMOResult)
                r_mid = midpoint(times, values, 0.25)
                self.assertIsInstance(r_mid, MidpointResult)
                r_ac = area_cog(times, values)
                self.assertIsInstance(r_ac, AreaCogResult)


# ---------------------------------------------------------------------------
# utils.py — read_data tests
# ---------------------------------------------------------------------------

class TestReadData(unittest.TestCase):
    """Tests for read_data()."""

    def setUp(self):
        self.data = read_data(DUMMY_DATA_FULL)

    def test_returns_dataframe(self):
        self.assertIsInstance(self.data, pd.DataFrame)

    def test_required_columns_present(self):
        for col in ["Participant", "Date", "Time", "Mel", "Timestamp"]:
            self.assertIn(col, self.data.columns)

    def test_participant_is_int(self):
        self.assertTrue(
            np.issubdtype(self.data.Participant.dtype, np.integer))

    def test_mel_is_float(self):
        self.assertTrue(
            np.issubdtype(self.data.Mel.dtype, np.floating))

    def test_timestamp_is_datetime(self):
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(self.data.Timestamp))

    def test_not_empty(self):
        self.assertGreater(len(self.data), 0)


# ---------------------------------------------------------------------------
# utils.py — prepare_part_data tests
# ---------------------------------------------------------------------------

class TestPreparePartData(unittest.TestCase):
    """Tests for prepare_part_data()."""

    def setUp(self):
        self.data = read_data(DUMMY_DATA_FULL)
        self.participant = np.unique(self.data.Participant)[0]
        self.p_data = prepare_part_data(self.data, self.participant)

    def test_returns_dataframe(self):
        self.assertIsInstance(self.p_data, pd.DataFrame)

    def test_timedays_column_absent(self):
        self.assertNotIn("Timedays", self.p_data.columns)

    def test_timestamp_monotonically_increasing(self):
        self.assertTrue(np.all(
            np.diff(self.p_data.Timestamp) >= np.timedelta64(0)))

    def test_only_selected_participant(self):
        self.assertTrue(
            np.all(self.p_data.Participant == self.participant))


# ---------------------------------------------------------------------------
# utils.py — to_days / from_days tests
# ---------------------------------------------------------------------------

class TestToDays(unittest.TestCase):
    """Tests for to_days()."""

    def test_unix_epoch_is_zero(self):
        result = to_days(pd.DatetimeIndex(["1970-01-01T00:00:00"]))
        self.assertAlmostEqual(result[0], 0.0, places=10)

    def test_one_day_later_is_one(self):
        result = to_days(pd.DatetimeIndex(["1970-01-02T00:00:00"]))
        self.assertAlmostEqual(result[0], 1.0, places=10)

    def test_fractional_part_matches_time_of_day(self):
        result = to_days(pd.DatetimeIndex(["1970-01-01T06:00:00"]))
        self.assertAlmostEqual(result[0], 0.25, places=10)

    def test_naive_and_utc_aware_agree(self):
        naive = to_days(pd.DatetimeIndex(["2024-03-15T12:00:00"]))
        aware = to_days(pd.DatetimeIndex(["2024-03-15T12:00:00+00:00"]))
        self.assertAlmostEqual(naive[0], aware[0], places=10)

    def test_non_utc_aware_is_converted(self):
        utc   = to_days(pd.DatetimeIndex(["2024-01-01T12:00:00+00:00"]))
        plus2 = to_days(pd.DatetimeIndex(["2024-01-01T14:00:00+02:00"]))
        self.assertAlmostEqual(utc[0], plus2[0], places=10)

    def test_returns_numpy_array(self):
        result = to_days(pd.DatetimeIndex(["2024-01-01"]))
        self.assertIsInstance(result, np.ndarray)

    def test_accepts_datetime64_array(self):
        arr = np.array(["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                       dtype="datetime64[ns]")
        result = to_days(arr)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[1] - result[0], 1.0, places=10)


class TestFromDays(unittest.TestCase):
    """Tests for from_days()."""

    def test_zero_is_unix_epoch(self):
        result = from_days(np.array([0.0]))
        self.assertEqual(result[0], pd.Timestamp("1970-01-01", tz="UTC"))

    def test_one_is_one_day_later(self):
        result = from_days(np.array([1.0]))
        self.assertEqual(result[0], pd.Timestamp("1970-01-02", tz="UTC"))

    def test_returns_datetimeindex(self):
        result = from_days(np.array([0.0, 1.0]))
        self.assertIsInstance(result, pd.DatetimeIndex)

    def test_roundtrip_with_to_days(self):
        original = pd.DatetimeIndex(["2024-03-15T08:30:00+00:00",
                                     "2024-03-15T20:45:00+00:00"])
        result = from_days(to_days(original))
        for a, b in zip(original, result):
            self.assertAlmostEqual(a.value, b.value, delta=1000)


# utils.py — gen_time_range tests
# ---------------------------------------------------------------------------

class TestResampleTime(unittest.TestCase):
    """Tests for gen_time_range()."""

    def test_output_is_array(self):
        result = gen_time_range(tmin=pd.Timestamp("2024-01-01"),
                               tmax=pd.Timestamp("2024-01-02"), step="1min")
        self.assertIsInstance(result, np.ndarray)

    def test_full_day_extends_short_range(self):
        tmin = pd.Timestamp("2024-01-01 12:00")
        tmax = pd.Timestamp("2024-01-01 19:12")
        result_full  = gen_time_range(tmin=tmin, tmax=tmax, step="1min", full_day=True)
        result_short = gen_time_range(tmin=tmin, tmax=tmax, step="1min", full_day=False)
        self.assertGreater(len(result_full), len(result_short))

    def test_step_size_is_correct(self):
        tmin = pd.Timestamp("2024-01-01")
        tmax = pd.Timestamp("2024-01-02")
        for step_str, dt_min in [("1min", 1.0), ("5min", 5.0), ("15min", 15.0)]:
            result = gen_time_range(tmin=tmin, tmax=tmax, step=step_str, full_day=False)
            step = result[1] - result[0]
            self.assertAlmostEqual(step, dt_min / (24 * 60), places=10)


# ---------------------------------------------------------------------------
# utils.py — day_profile tests
# ---------------------------------------------------------------------------

class TestDayProfile(unittest.TestCase):
    """Tests for day_profile()."""

    def setUp(self):
        self.times = pd.date_range("2024-01-01", periods=len(T2), freq="1min")
        self.values = (bsbcf(t=T2, p=BSBCF_PARAMS_ARRAY) + 
                  np.random.normal(0, 5, size=T2.shape))
        self.values = self.values - np.min(self.values) + 1.0

    def test_returns_two_series(self):
        mean, std = day_profile(self.times, self.values)
        self.assertIsInstance(mean, pd.Series)
        self.assertIsInstance(std, pd.Series)

    def test_double_doubles_length(self):
        mean_single, _ = day_profile(self.times, self.values, double=False)
        mean_double, _ = day_profile(self.times, self.values, double=True)
        self.assertEqual(len(mean_double), 2 * len(mean_single))

    def test_repfirst_adds_one_row(self):
        mean_normal,   _ = day_profile(self.times, self.values, repfirst=False)
        mean_repfirst, _ = day_profile(self.times, self.values, repfirst=True)
        self.assertEqual(len(mean_repfirst), len(mean_normal) + 1)

    def test_mean_std_values_are_finite(self):
        mean, std = day_profile(self.times, self.values)
        self.assertTrue(np.all(np.isfinite(mean.values)))
        self.assertTrue(np.all(np.isfinite(std.values)))

# ---------------------------------------------------------------------------
# utils.py — time_to_phase tests
# ---------------------------------------------------------------------------

class TestTimeToPhase(unittest.TestCase):
    """Tests for time_to_phase()."""

    def test_zero(self):
        self.assertAlmostEqual(time_to_phase(0.0), 0.0)

    def test_one_wraps_to_zero(self):
        self.assertAlmostEqual(time_to_phase(1.0), 0.0)

    def test_half(self):
        self.assertAlmostEqual(time_to_phase(0.5), 0.5)

    def test_greater_than_one(self):
        self.assertAlmostEqual(time_to_phase(1.75), 0.75)

    def test_negative(self):
        self.assertAlmostEqual(time_to_phase(-0.25), 0.75)

    def test_hours_mode_6h(self):
        self.assertAlmostEqual(time_to_phase(6.0, hours=True), 0.25)

    def test_hours_mode_24h_wraps(self):
        self.assertAlmostEqual(time_to_phase(24.0, hours=True), 0.0)

    def test_result_always_in_unit_interval(self):
        for t in np.linspace(-3.0, 3.0, 61):
            result = time_to_phase(t)
            self.assertGreaterEqual(result, 0.0)
            self.assertLess(result, 1.0)


# ---------------------------------------------------------------------------
# utils.py — phase_to_string tests
# ---------------------------------------------------------------------------

class TestPhaseToString(unittest.TestCase):
    """Tests for phase_to_string()."""

    def test_zero(self):
        self.assertEqual(phase_to_string(0.0), "00:00")

    def test_half(self):
        self.assertEqual(phase_to_string(0.5), "12:00")

    def test_quarter(self):
        self.assertEqual(phase_to_string(0.25), "06:00")

    def test_three_quarters(self):
        self.assertEqual(phase_to_string(0.75), "18:00")

    def test_negative_has_minus_sign(self):
        self.assertTrue(phase_to_string(-0.25).startswith("-"))

    def test_returns_string(self):
        self.assertIsInstance(phase_to_string(0.5), str)

    def test_format_hhmm(self):
        import re
        self.assertRegex(phase_to_string(0.875), r"^-?\d{2}:\d{2}$")


# ---------------------------------------------------------------------------
# utils.py — abs_threshold tests
# ---------------------------------------------------------------------------

class TestAbsThreshold(unittest.TestCase):
    """Tests for abs_threshold()."""

    def test_zero_relative_gives_minimum(self):
        self.assertAlmostEqual(
            abs_threshold(np.array([2.0, 5.0, 10.0]), 0.0), 2.0)

    def test_one_relative_gives_maximum(self):
        self.assertAlmostEqual(
            abs_threshold(np.array([2.0, 5.0, 10.0]), 1.0), 10.0)

    def test_quarter_relative(self):
        self.assertAlmostEqual(
            abs_threshold(np.array([0.0, 100.0]), 0.25), 25.0)

    def test_result_between_min_and_max(self):
        values = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        thresh = abs_threshold(values, 0.25)
        self.assertGreaterEqual(thresh, np.min(values))
        self.assertLessEqual(thresh, np.max(values))


# ---------------------------------------------------------------------------
# utils.py — phase_diff tests
# ---------------------------------------------------------------------------

class TestPhaseDiff(unittest.TestCase):
    """Tests for phase_diff()."""

    def test_identical_phases(self):
        self.assertAlmostEqual(phase_diff(0.5, 0.5), 0.0)

    def test_result_always_in_range(self):
        for p1 in np.linspace(0, 1, 13):
            for p2 in np.linspace(0, 1, 13):
                result = phase_diff(p1, p2)
                self.assertGreaterEqual(result, -0.5)
                self.assertLessEqual(result, 0.5)

    def test_antisymmetric(self):
        self.assertAlmostEqual(phase_diff(0.3, 0.7), -phase_diff(0.7, 0.3))

    def test_wrap_around_midnight(self):
        self.assertAlmostEqual(phase_diff(0.05, 0.95), 0.1, places=10)

    def test_out_of_range_inputs_normalised(self):
        self.assertAlmostEqual(phase_diff(0.3, 0.1), phase_diff(1.3, 1.1),
                               places=10)


# ---------------------------------------------------------------------------
# utils.py — params_to_string tests
# ---------------------------------------------------------------------------

class TestParamsToString(unittest.TestCase):
    """Tests for params_to_string()."""

    def test_dict_input_uses_key_names(self):
        """Dict input should produce key=value pairs."""
        result = params_to_string(BSBCF_PARAMS_DICT)
        for key in BSBCF_PARAM_NAMES:
            self.assertIn(key, result)

    def test_array_input_uses_positional_names(self):
        """Array input should produce p0=value, p1=value, ... pairs."""
        result = params_to_string(BSBCF_PARAMS_ARRAY)
        for i in range(len(BSBCF_PARAMS_ARRAY)):
            self.assertIn(f"p{i}=", result)

    def test_returns_string(self):
        self.assertIsInstance(params_to_string(BSBCF_PARAMS_DICT), str)
        self.assertIsInstance(params_to_string(BSBCF_PARAMS_ARRAY), str)

    def test_default_decimal_places(self):
        """Default ndec=3 should produce values with 3 decimal places."""
        result = params_to_string({"phi": 0.875})
        self.assertIn("phi=0.875", result)

    def test_custom_decimal_places(self):
        """ndec parameter should control decimal places."""
        result = params_to_string({"phi": 0.875}, ndec=1)
        self.assertIn("phi=0.9", result)

    def test_comma_separated(self):
        """Multiple parameters should be comma-separated."""
        result = params_to_string(BSBCF_PARAMS_DICT)
        self.assertIn(",", result)

    def test_dict_values_correct(self):
        """Values in output should match input dict values."""
        p = {"phi": 0.500, "b": 2.000}
        result = params_to_string(p, ndec=3)
        self.assertIn("phi=0.500", result)
        self.assertIn("b=2.000", result)

    def test_empty_dict_returns_empty_string(self):
        self.assertEqual(params_to_string({}), "")

    def test_fit_result_works(self):
        """params_to_string should accept a FitResult directly (Mapping over params)."""
        res = fit(T, bsbcf(t=T, p=BSBCF_PARAMS_ARRAY), f=bsbcf)
        result = params_to_string(res)
        self.assertIsInstance(result, str)
        for key in BSBCF_PARAM_NAMES:
            self.assertIn(key, result)


# ---------------------------------------------------------------------------
# utils.py — string_to_phase tests
# ---------------------------------------------------------------------------

class TestStringToPhase(unittest.TestCase):
    """Tests for string_to_phase()."""

    def test_zero(self):
        self.assertAlmostEqual(string_to_phase("00:00"), 0.0)

    def test_quarter(self):
        self.assertAlmostEqual(string_to_phase("06:00"), 0.25)

    def test_half(self):
        self.assertAlmostEqual(string_to_phase("12:00"), 0.5)

    def test_three_quarters(self):
        self.assertAlmostEqual(string_to_phase("18:00"), 0.75)

    def test_with_minutes(self):
        self.assertAlmostEqual(string_to_phase("06:30"),
                               6.5 / 24.0, places=10)

    def test_negative(self):
        self.assertAlmostEqual(string_to_phase("-06:00"), -0.25)

    def test_negative_with_minutes(self):
        self.assertAlmostEqual(string_to_phase("-02:30"),
                               -2.5 / 24.0, places=10)

    def test_inverse_of_phase_to_string(self):
        """string_to_phase should be the inverse of phase_to_string."""
        for phase in [0.0, 0.25, 0.5, 0.75, 0.875, -0.25, -0.5]:
            roundtripped = string_to_phase(phase_to_string(phase))
            self.assertAlmostEqual(roundtripped, phase, places=4)

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            string_to_phase("not a time")

    def test_too_many_parts_raises(self):
        with self.assertRaises(ValueError):
            string_to_phase("12:30:45")

    def test_strips_whitespace(self):
        self.assertAlmostEqual(string_to_phase("  06:00  "), 0.25)


# ---------------------------------------------------------------------------
# results.py — SessionInfo tests
# ---------------------------------------------------------------------------

def _make_p_data(participant, start, end):
    """Build a minimal participant DataFrame for SessionInfo tests."""
    timestamps = pd.date_range(start=start, end=end, freq="1h")
    return pd.DataFrame({
        "Participant": participant,
        "Timestamp": timestamps,
    })


class TestSessionInfo(unittest.TestCase):
    """Tests for SessionInfo dataclass."""

    def test_construction_from_p_data(self):
        p_data = _make_p_data(1, "2024-01-01 21:00", "2024-01-02 09:00")
        meta = SessionInfo(p_data)
        self.assertEqual(meta.participant, 1)
        self.assertEqual(meta.start, pd.Timestamp("2024-01-01 21:00"))
        self.assertEqual(meta.end, pd.Timestamp("2024-01-02 09:00"))

    def test_string_participant(self):
        """participant field should accept strings."""
        p_data = _make_p_data("P01", "2024-01-01", "2024-01-02")
        meta = SessionInfo(p_data)
        self.assertEqual(meta.participant, "P01")

    def test_to_dict_returns_dict(self):
        p_data = _make_p_data(1, "2024-01-01 21:00", "2024-01-02 09:00")
        meta = SessionInfo(p_data)
        d = meta.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(list(d.keys()), ["participant", "start", "end"])
        self.assertEqual(d["participant"], 1)
        self.assertNotIn("func", d)
        self.assertNotIn("r2", d)


# ---------------------------------------------------------------------------
# utils.py — ResultsCollector tests
# ---------------------------------------------------------------------------

class TestResultsCollector(unittest.TestCase):
    """Tests for ResultsCollector."""

    def setUp(self):
        self.tmpdir = "/tmp/melafit_test_results/"
        self.filename = "test_results"
        # Generate one analysis run for testing
        self.func = bsbcf
        self.values = self.func(t=T, p=BSBCF_PARAMS_ARRAY)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(T),
            freq=pd.Timedelta(minutes=1))
        self.res = fit(T, self.values, f=self.func)
        self.meta = SessionInfo(_make_p_data(1, "2024-01-01 00:00", "2024-01-02 00:00"))
        self.ampl = amplitude(self.values)
        self.mid = midpoint(self.times, self.values, 0.25)
        self.ac = area_cog(self.times, self.values)

    def tearDown(self):
        import shutil
        import os
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def test_empty_collector_save_does_nothing(self):
        """Saving an empty collector should not raise or create files."""
        import os
        collector = ResultsCollector()
        collector.save(self.tmpdir, self.filename)
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        self.assertFalse(os.path.exists(filepath))

    def test_add_requires_meta_info(self):
        """add() should raise ValueError if no SessionInfo is provided."""
        collector = ResultsCollector()
        with self.assertRaises(ValueError):
            collector.add(self.res, self.ampl)

    def test_add_rejects_unknown_type(self):
        """add() should raise TypeError on unsupported argument type."""
        collector = ResultsCollector()
        with self.assertRaises(TypeError):
            collector.add(self.meta, "not a valid object")

    def test_add_full_profile(self):
        """add() should accept all marker types together."""
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        self.assertEqual(len(collector._records), 1)

    def test_add_partial_dlmo(self):
        """add() should work with only SessionInfo, fit result and midpoint."""
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.mid)
        self.assertEqual(len(collector._records), 1)

    def test_argument_order_irrelevant(self):
        """add() should produce identical records regardless of arg order."""
        c1 = ResultsCollector()
        c2 = ResultsCollector()
        c1.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        c2.add(self.ac, self.mid, self.ampl, self.res, self.meta)
        self.assertEqual(c1._records, c2._records)

    def test_record_contains_meta_fields(self):
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        record = collector._records[0]
        for key in ["participant", "start", "end", "func", "r2"]:
            self.assertIn(key, record)

    def test_record_contains_marker_fields(self):
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        record = collector._records[0]
        for key in ["amplitude", "baseline", "dlmon", "dlmoff", "midpoint",
                    "threshold", "area", "cog"] + BUILTIN_PARAM_NAMES[self.func]:
            self.assertIn(key, record)

    def test_record_timing_fields_are_strings(self):
        """Timing fields should be stored as HH:MM strings."""
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        record = collector._records[0]
        for key in ["dlmon", "dlmoff", "midpoint", "cog"]:
            self.assertIsInstance(record[key], str)

    def test_save_creates_file(self):
        """save() should create the Excel file at the expected location."""
        import os
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        collector.save(self.tmpdir, self.filename)
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        self.assertTrue(os.path.exists(filepath))

    def test_save_filename_with_xlsx_extension(self):
        """save() should not double the .xlsx extension if already present."""
        import os
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        collector.save(self.tmpdir, self.filename + ".xlsx")
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        self.assertTrue(os.path.exists(filepath))
        bad_path = os.path.join(self.tmpdir, self.filename + ".xlsx.xlsx")
        self.assertFalse(os.path.exists(bad_path))

    def test_save_creates_directory(self):
        """save() should create the result directory if missing."""
        import os
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        collector.save(self.tmpdir, self.filename)
        self.assertTrue(os.path.isdir(self.tmpdir))

    def test_saved_excel_readable(self):
        """Saved Excel file should be readable with expected columns."""
        import os
        collector = ResultsCollector()
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        collector.save(self.tmpdir, self.filename)
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        df = pd.read_excel(filepath, index_col="participant")
        self.assertGreater(len(df), 0)
        for col in ["end", "func", "r2", "amplitude", "baseline", "dlmon",
                    "dlmoff", "midpoint", "area", "cog"] + BUILTIN_PARAM_NAMES[self.func]:
            self.assertIn(col, df.columns)

    def test_multiple_participants_sorted_in_excel(self):
        """Excel output should have participants sorted by ID."""
        import os
        collector = ResultsCollector()
        meta2 = SessionInfo(_make_p_data(3, "2024-01-02 00:00", "2024-01-03 00:00"))
        meta3 = SessionInfo(_make_p_data(2, "2024-01-03 00:00", "2024-01-04 00:00"))
        collector.add(self.meta, self.res, self.ampl, self.mid, self.ac)
        collector.add(meta2, self.res, self.ampl, self.mid, self.ac)
        collector.add(meta3, self.res, self.ampl, self.mid, self.ac)
        collector.save(self.tmpdir, self.filename)
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        df = pd.read_excel(filepath, index_col="participant")
        self.assertEqual(list(df.index), [1, 2, 3])

    def test_dlmo_workflow_nan_fields(self):
        """A DLMO-style add() should leave non-applicable fields as NaN."""
        import os
        meta_dlmo = SessionInfo(_make_p_data(1, "2024-01-01 18:00", "2024-01-02 06:00"))
        collector = ResultsCollector()
        collector.add(meta_dlmo, self.res, self.mid)
        collector.save(self.tmpdir, self.filename)
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        df = pd.read_excel(filepath, index_col="participant")
        # r2 is always computed by fit() so it should be finite
        self.assertTrue(np.isfinite(df["r2"].iloc[0]))
        # area and cog should be missing as columns (no AreaCogResult added)
        self.assertNotIn("area", df.columns)

    def test_add_dlmo_result(self):
        """add() should accept a DLMOResult alongside meta and fit result."""
        dl = dlmo(self.times, self.values, 0.25)
        collector = ResultsCollector()
        collector.add(self.meta, self.res, dl)
        self.assertEqual(len(collector._records), 1)

    def test_dlmo_result_record_contains_dlmo_field(self):
        """Record with DLMOResult should have 'dlmo' key."""
        dl = dlmo(self.times, self.values, 0.25)
        collector = ResultsCollector()
        collector.add(self.meta, self.res, dl)
        record = collector._records[0]
        self.assertIn("dlmo", record)
        self.assertIsInstance(record["dlmo"], str)

    def test_dlmo_result_workflow_excel(self):
        """DLMO-only workflow: 'dlmo' column present, 'dlmoff'/'midpoint' absent."""
        import os
        dl = dlmo(self.times, self.values, 0.25)
        collector = ResultsCollector()
        collector.add(self.meta, self.res, dl)
        collector.save(self.tmpdir, self.filename)
        filepath = os.path.join(self.tmpdir, self.filename + ".xlsx")
        df = pd.read_excel(filepath, index_col="participant")
        self.assertIn("dlmo", df.columns)
        self.assertNotIn("dlmoff", df.columns)
        self.assertNotIn("midpoint", df.columns)

    def test_dlmo_result_argument_order_irrelevant(self):
        """add() with DLMOResult should be order-independent."""
        dl = dlmo(self.times, self.values, 0.25)
        c1 = ResultsCollector()
        c2 = ResultsCollector()
        c1.add(self.meta, self.res, dl)
        c2.add(dl, self.res, self.meta)
        self.assertEqual(c1._records, c2._records)


# ---------------------------------------------------------------------------
# markers.py + utils.py — interior NaN handling (sparse / Excel-like input)
# ---------------------------------------------------------------------------

class TestInteriorNaNHandling(unittest.TestCase):
    """
    Verify behaviour when input values contain interior NaNs, as can happen
    when melatonin data are read from an Excel table with empty cells.

    Input is modelled as hourly samples (one row per hour, 25 rows covering
    one full day) so that the 1-min day_profile bins are sparse — matching
    the typical Excel-read use case.  A block of consecutive interior samples
    is replaced with NaN to simulate missing mid-night measurements.
    """

    def setUp(self):
        # 25 hourly timestamps spanning 24 h (one full day + wrap point)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=25,
            freq="1h")

        # Ground-truth waveform evaluated at each hour
        t_hours = np.arange(25) / 24.0
        self.values_clean = bsbcf(t=t_hours, p=BSBCF_PARAMS_ARRAY)

        # Introduce NaN at hours 10–14 (interior gap, away from the DLMO
        # crossing which falls around 21:00 for phi=0.875)
        self.values_nan = self.values_clean.copy()
        self.values_nan[10:15] = np.nan

    # --- day_profile ---

    def test_day_profile_with_interior_nans_returns_finite_mean(self):
        mean, _ = day_profile(self.times, self.values_nan, binsize=60,
                              interp='linear')
        self.assertTrue(np.all(np.isfinite(mean.values)),
                        "day_profile mean should be finite with interior NaNs "
                        "when interp='linear'")

    def test_day_profile_interior_nan_close_to_clean(self):
        """Interpolated profile should be close to the clean profile."""
        mean_clean, _ = day_profile(self.times, self.values_clean, binsize=60,
                                    interp='linear')
        mean_nan,   _ = day_profile(self.times, self.values_nan,   binsize=60,
                                    interp='linear')
        np.testing.assert_allclose(mean_nan.values, mean_clean.values,
                                   atol=5.0,
                                   err_msg="Interior NaN gap should be "
                                           "bridged by linear interpolation")

    # --- dlmo ---

    def test_dlmo_with_interior_nans_returns_result(self):
        result = dlmo(self.times, self.values_nan, 0.25)
        self.assertIsInstance(result, DLMOResult)

    def test_dlmo_with_interior_nans_in_range(self):
        result = dlmo(self.times, self.values_nan, 0.25)
        self.assertGreaterEqual(result.dlmo, 0.0)
        self.assertLess(result.dlmo, 1.0)

    def test_dlmo_with_interior_nans_close_to_clean(self):
        r_clean = dlmo(self.times, self.values_clean, 0.25)
        r_nan   = dlmo(self.times, self.values_nan,   0.25)
        diff = abs(phase_diff(r_nan.dlmo, r_clean.dlmo))
        self.assertLess(diff, 1.0 / 24.0,
                        "DLMO with interior NaNs should agree with clean "
                        "result within 1 hour")

    # --- midpoint ---

    def test_midpoint_with_interior_nans_returns_result(self):
        result = midpoint(self.times, self.values_nan, 0.25)
        self.assertIsInstance(result, MidpointResult)

    def test_midpoint_with_interior_nans_all_phases_in_range(self):
        result = midpoint(self.times, self.values_nan, 0.25)
        for val in [result.dlmon, result.dlmoff, result.midpoint]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)

    def test_midpoint_with_interior_nans_close_to_clean(self):
        r_clean = midpoint(self.times, self.values_clean, 0.25)
        r_nan   = midpoint(self.times, self.values_nan,   0.25)
        for attr in ("dlmon", "dlmoff", "midpoint"):
            diff = abs(phase_diff(getattr(r_nan, attr),
                                  getattr(r_clean, attr)))
            self.assertLess(diff, 1.0 / 24.0,
                            f"{attr} with interior NaNs should agree with "
                            "clean result within 1 hour")

    # --- area_cog ---

    def test_area_cog_with_interior_nans_returns_result(self):
        result = area_cog(self.times, self.values_nan)
        self.assertIsInstance(result, AreaCogResult)

    def test_area_cog_with_interior_nans_finite(self):
        result = area_cog(self.times, self.values_nan)
        self.assertTrue(np.isfinite(result.area))
        self.assertTrue(np.isfinite(result.cog))

    def test_area_cog_with_interior_nans_close_to_clean(self):
        r_clean = area_cog(self.times, self.values_clean)
        r_nan   = area_cog(self.times, self.values_nan)
        self.assertAlmostEqual(r_nan.area, r_clean.area, delta=r_clean.area * 0.1,
                               msg="Area should agree within 10 % with interior NaNs")
        diff = abs(phase_diff(r_nan.cog, r_clean.cog))
        self.assertLess(diff, 1.0 / 24.0,
                        "COG with interior NaNs should agree with clean "
                        "result within 1 hour")


if __name__ == "__main__":
    unittest.main()