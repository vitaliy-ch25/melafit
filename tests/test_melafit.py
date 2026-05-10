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

from melafit.fitting import (bcf, sbcf, bbcf, bsbcf, cost, rsquared,
                              func_defaults, fit, params_to_array,
                              array_to_params, _resolve_params,
                              BCF_PARAM_NAMES, SBCF_PARAM_NAMES,
                              BBCF_PARAM_NAMES, BSBCF_PARAM_NAMES,
                              PARAM_NAMES)
from melafit.markers import amplitude, midpoint, area_cog
from melafit.utils import (read_data, prepare_part_data, compute_wave,
                            day_profile, time_to_phase, phase_to_string,
                            abs_threshold, phase_diff)

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

# Time array covering one full day at 1-minute resolution
T = np.linspace(0, 1, 1440, endpoint=False)

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

    def test_returns_optimize_result(self):
        res = fit(self.t, self.y)
        self.assertIsInstance(res, opt.OptimizeResult)
        self.assertIn('x', res)

    def test_returns_param_dict(self):
        res = fit(self.t, self.y, f=bsbcf)
        self.assertIsInstance(res.p, dict)
        self.assertEqual(list(res.p.keys()), BSBCF_PARAM_NAMES)

    def test_param_dict_consistent_with_array(self):
        res = fit(self.t, self.y, f=bsbcf)
        for i, key in enumerate(BSBCF_PARAM_NAMES):
            self.assertAlmostEqual(res.p[key], res.x[i], places=10)

    def test_param_dict_correct_keys_all_functions(self):
        for func, names, params in [
            (bcf,   BCF_PARAM_NAMES,   BCF_PARAMS_ARRAY),
            (sbcf,  SBCF_PARAM_NAMES,  SBCF_PARAMS_ARRAY),
            (bbcf,  BBCF_PARAM_NAMES,  BBCF_PARAMS_ARRAY),
            (bsbcf, BSBCF_PARAM_NAMES, BSBCF_PARAMS_ARRAY),
        ]:
            y = func(t=T, p=params)
            res = fit(T, y, f=func)
            self.assertEqual(list(res.p.keys()), names)

    def test_bsbcf_converges(self):
        res = fit(self.t, self.y, f=bsbcf)
        self.assertTrue(res.success or res.fun < 0.01)

    def test_bcf_converges(self):
        res = fit(T, bcf(t=T, p=BCF_PARAMS_ARRAY), f=bcf)
        self.assertTrue(res.success or res.fun < 0.01)

    def test_sbcf_converges(self):
        res = fit(T, sbcf(t=T, p=SBCF_PARAMS_ARRAY), f=sbcf)
        self.assertTrue(res.success or res.fun < 0.01)

    def test_bbcf_converges(self):
        res = fit(T, bbcf(t=T, p=BBCF_PARAMS_ARRAY), f=bbcf)
        self.assertTrue(res.success or res.fun < 0.01)

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

    def test_cost_p_passed_through(self):
        res = fit(self.t, self.y, f=bsbcf, cost_p={"eps": 1e-6})
        self.assertIsInstance(res, opt.OptimizeResult)

    def test_with_real_data(self):
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timedays.values, p_data.Mel.values, f=bsbcf)
        self.assertTrue(res.success or res.fun < 1.0)
        self.assertIsInstance(res.p, dict)
        self.assertEqual(list(res.p.keys()), BSBCF_PARAM_NAMES)


# ---------------------------------------------------------------------------
# markers.py — amplitude tests
# ---------------------------------------------------------------------------

class TestAmplitude(unittest.TestCase):
    """Tests for amplitude()."""

    def test_known_amplitude(self):
        self.assertAlmostEqual(
            amplitude(np.array([2.0, 5.0, 10.0, 3.0])), 8.0)

    def test_flat_signal_is_zero(self):
        self.assertAlmostEqual(amplitude(np.full(100, 5.0)), 0.0)

    def test_amplitude_on_waveform(self):
        ampl = amplitude(bsbcf(t=T, p=BSBCF_PARAMS_ARRAY))
        self.assertGreater(ampl, 0.0)
        self.assertTrue(np.isfinite(ampl))


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

    def test_returns_four_values(self):
        self.assertEqual(len(midpoint(self.times, self.values, 0.25)), 4)

    def test_all_phases_in_range(self):
        midpt, dlmon, dlmoff, _ = midpoint(self.times, self.values, 0.25)
        for val in [midpt, dlmon, dlmoff]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)

    def test_relative_threshold_converted_correctly(self):
        _, _, _, thresh = midpoint(self.times, self.values, 0.25)
        self.assertAlmostEqual(thresh, abs_threshold(self.values, 0.25))

    def test_absolute_threshold_mode(self):
        _, _, _, thresh = midpoint(
            self.times, self.values, 10.0, thresh_abs=True)
        self.assertAlmostEqual(thresh, 10.0)

    def test_with_real_data(self):
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timedays.values, p_data.Mel.values, f=bsbcf)
        curve = compute_wave(p_data.Timedays.min(), p_data.Timedays.max(),
                             1.0, bsbcf, res.p)
        times = pd.date_range(p_data.Timestamp.min(), periods=len(curve),
                              freq=pd.Timedelta(minutes=1))
        midpt, dlmon, dlmoff, _ = midpoint(times, curve, 0.25)
        for val in [midpt, dlmon, dlmoff]:
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

    def test_returns_two_values(self):
        self.assertEqual(len(area_cog(self.times, self.values)), 2)

    def test_area_is_positive(self):
        area, _ = area_cog(self.times, self.values)
        self.assertGreater(area, 0.0)

    def test_cog_in_range(self):
        _, cog = area_cog(self.times, self.values)
        self.assertGreaterEqual(cog, 0.0)
        self.assertLess(cog, 1.0)

    def test_custom_baseline_changes_area(self):
        area_default, _ = area_cog(self.times, self.values)
        area_custom, _  = area_cog(self.times, self.values,
                                   baseline=min(self.values) + 1.0)
        self.assertNotAlmostEqual(area_default, area_custom)

    def test_with_real_data(self):
        data = read_data(DUMMY_DATA_FULL)
        p_data = prepare_part_data(data, np.unique(data.Participant)[0])
        res = fit(p_data.Timedays.values, p_data.Mel.values, f=bsbcf)
        curve = compute_wave(p_data.Timedays.min(), p_data.Timedays.max(),
                             1.0, bsbcf, res.p)
        times = pd.date_range(p_data.Timestamp.min(), periods=len(curve),
                              freq=pd.Timedelta(minutes=1))
        area, cog = area_cog(times, curve)
        self.assertGreater(area, 0.0)
        self.assertGreaterEqual(cog, 0.0)
        self.assertLess(cog, 1.0)


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

    def test_timedays_column_present(self):
        self.assertIn("Timedays", self.p_data.columns)

    def test_timedays_monotonically_increasing(self):
        self.assertTrue(np.all(np.diff(self.p_data.Timedays.values) >= 0))

    def test_timedays_starts_at_correct_hour(self):
        base = self.p_data.Timestamp.min()
        expected = base.hour / 24.0 + base.minute / (24 * 60)
        self.assertAlmostEqual(self.p_data.Timedays.iloc[0], expected,
                               places=4)

    def test_only_selected_participant(self):
        self.assertTrue(
            np.all(self.p_data.Participant == self.participant))


# ---------------------------------------------------------------------------
# utils.py — compute_wave tests
# ---------------------------------------------------------------------------

class TestComputeWave(unittest.TestCase):
    """Tests for compute_wave() — both array and dict parameter input."""

    def test_output_is_array(self):
        result = compute_wave(0.0, 1.0, 1.0, bsbcf, BSBCF_PARAMS_ARRAY)
        self.assertIsInstance(result, np.ndarray)

    def test_dict_and_array_give_same_result(self):
        result_array = compute_wave(0.0, 1.0, 1.0, bsbcf, BSBCF_PARAMS_ARRAY)
        result_dict  = compute_wave(0.0, 1.0, 1.0, bsbcf, BSBCF_PARAMS_DICT)
        np.testing.assert_array_equal(result_array, result_dict)

    def test_full_wave_extends_short_range(self):
        result_full  = compute_wave(0.5, 0.8, 1.0, bsbcf, BSBCF_PARAMS_ARRAY,
                                    full_wave=True)
        result_short = compute_wave(0.5, 0.8, 1.0, bsbcf, BSBCF_PARAMS_ARRAY,
                                    full_wave=False)
        self.assertGreater(len(result_full), len(result_short))

    def test_output_is_finite(self):
        result = compute_wave(0.0, 1.0, 1.0, bsbcf, BSBCF_PARAMS_ARRAY)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_accepts_fit_result_param_dict(self):
        """compute_wave should accept res.p directly."""
        res = fit(T, bsbcf(t=T, p=BSBCF_PARAMS_ARRAY), f=bsbcf)
        result = compute_wave(0.0, 1.0, 1.0, bsbcf, res.p)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.all(np.isfinite(result)))


# ---------------------------------------------------------------------------
# utils.py — day_profile tests
# ---------------------------------------------------------------------------

class TestDayProfile(unittest.TestCase):
    """Tests for day_profile()."""

    def setUp(self):
        times = pd.date_range("2024-01-01", periods=1440, freq="1min")
        values = bsbcf(t=T, p=BSBCF_PARAMS_ARRAY)
        self.series = pd.Series(index=times, data=values)

    def test_returns_two_series(self):
        mean, std = day_profile(self.series)
        self.assertIsInstance(mean, pd.Series)
        self.assertIsInstance(std, pd.Series)

    def test_double_doubles_length(self):
        mean_single, _ = day_profile(self.series, double=False)
        mean_double, _ = day_profile(self.series, double=True)
        self.assertEqual(len(mean_double), 2 * len(mean_single))

    def test_repfirst_adds_one_row(self):
        mean_normal,   _ = day_profile(self.series, repfirst=False)
        mean_repfirst, _ = day_profile(self.series, repfirst=True)
        self.assertEqual(len(mean_repfirst), len(mean_normal) + 1)

    def test_mean_values_are_finite(self):
        mean, _ = day_profile(self.series)
        self.assertTrue(np.all(np.isfinite(mean.values)))


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


if __name__ == "__main__":
    unittest.main()
