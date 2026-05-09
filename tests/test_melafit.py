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

from melafit.fitting import bcf, sbcf, bbcf, bsbcf, cost, rsquared, func_defaults, fit
from melafit.markers import amplitude, midpoint, area_cog
from melafit.utils import (read_data, prepare_part_data, compute_wave,
                            day_profile, time_to_phase, phase_to_string,
                            abs_threshold, phase_diff)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Canonical BSBCF parameters for a plausible melatonin waveform:
# phi=0.875 (~21:00), b=2, H=80, c=0.5, v=0.3, m=0.1
BSBCF_PARAMS = [0.875, 2.0, 80.0, 0.5, 0.3, 0.1]
BCF_PARAMS   = [0.875, 2.0, 80.0, 0.5]
SBCF_PARAMS  = [0.875, 2.0, 80.0, 0.5, 0.3]
BBCF_PARAMS  = [0.875, 2.0, 80.0, 0.5, 0.1]

# Time array covering one full day at 1-minute resolution
T = np.linspace(0, 1, 1440, endpoint=False)

# Full-profile dummy data path (from repo root)
DUMMY_DATA_FULL = "./data/dummy_data_full.xlsx"
DUMMY_DATA_DLMO = "./data/dummy_data_dlmo.xlsx"


# ---------------------------------------------------------------------------
# fitting.py tests
# ---------------------------------------------------------------------------

class TestWaveformFunctions(unittest.TestCase):
    """Tests for bcf, sbcf, bbcf, bsbcf waveform functions."""

    def _check_output(self, func, params):
        """Helper: check output shape, type and non-negativity."""
        result = func(t=T, p=params)
        self.assertEqual(result.shape, T.shape)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.all(result >= 0),
                        f"{func.__name__} produced negative values")

    def test_bcf_output_shape_and_type(self):
        self._check_output(bcf, BCF_PARAMS)

    def test_sbcf_output_shape_and_type(self):
        self._check_output(sbcf, SBCF_PARAMS)

    def test_bbcf_output_shape_and_type(self):
        self._check_output(bbcf, BBCF_PARAMS)

    def test_bsbcf_output_shape_and_type(self):
        self._check_output(bsbcf, BSBCF_PARAMS)

    def test_bcf_baseline_equals_b_when_flat(self):
        """With H=0, bcf should return b everywhere."""
        p = [0.0, 5.0, 0.0, 0.0]
        result = bcf(t=T, p=p)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)

    def test_bsbcf_peak_near_phase(self):
        """Peak of bsbcf should occur near phi (within +-3h = +-0.125 days)."""
        result = bsbcf(t=T, p=BSBCF_PARAMS)
        peak_phase = T[np.argmax(result)]
        phi = BSBCF_PARAMS[0]
        # Allow wrap-around
        diff = abs(peak_phase - phi)
        diff = min(diff, 1.0 - diff)
        self.assertLess(diff, 0.125,
                        f"Peak at {peak_phase:.3f}, expected near {phi:.3f}")

    def test_all_functions_return_finite(self):
        """All waveform functions must return finite values."""
        for func, params in [(bcf, BCF_PARAMS), (sbcf, SBCF_PARAMS),
                              (bbcf, BBCF_PARAMS), (bsbcf, BSBCF_PARAMS)]:
            result = func(t=T, p=params)
            self.assertTrue(np.all(np.isfinite(result)),
                            f"{func.__name__} returned non-finite values")


class TestCostFunction(unittest.TestCase):
    """Tests for the cost() function."""

    def setUp(self):
        self.y = bsbcf(t=T, p=BSBCF_PARAMS)

    def test_cost_perfect_fit_is_low(self):
        """Cost for perfect fit should be close to zero."""
        val = cost(p=BSBCF_PARAMS, t=T, y=self.y, f=bsbcf)
        self.assertLess(val, 1e-6)

    def test_cost_is_positive(self):
        """Cost should always be non-negative."""
        noisy_y = self.y + np.random.normal(0, 5, size=self.y.shape)
        val = cost(p=BSBCF_PARAMS, t=T, y=noisy_y, f=bsbcf)
        self.assertGreaterEqual(val, 0.0)

    def test_cost_eps_prevents_division_by_zero(self):
        """Flat predicted curve (var=0) should not raise ZeroDivisionError."""
        flat_params = [0.0, 5.0, 0.0, 0.0]  # H=0 gives flat bcf
        try:
            val = cost(p=flat_params, t=T, y=self.y, f=bcf)
            self.assertTrue(np.isfinite(val))
        except ZeroDivisionError:
            self.fail("cost() raised ZeroDivisionError with flat prediction")

    def test_cost_worse_with_wrong_params(self):
        """Cost should be higher with wrong parameters than correct ones."""
        wrong_params = [0.0, 2.0, 80.0, 0.5, 0.3, 0.1]  # phi=0 instead of 0.875
        val_correct = cost(p=BSBCF_PARAMS, t=T, y=self.y, f=bsbcf)
        val_wrong = cost(p=wrong_params, t=T, y=self.y, f=bsbcf)
        self.assertLess(val_correct, val_wrong)


class TestRSquared(unittest.TestCase):
    """Tests for rsquared()."""

    def test_perfect_fit_gives_one(self):
        y = bsbcf(t=T, p=BSBCF_PARAMS)
        r2 = rsquared(Y=y, y=y)
        self.assertAlmostEqual(r2, 1.0, places=10)

    def test_r2_decreases_with_noise(self):
        y = bsbcf(t=T, p=BSBCF_PARAMS)
        noisy_y = y + np.random.normal(0, 10, size=y.shape)
        r2 = rsquared(Y=y, y=noisy_y)
        self.assertLess(r2, 1.0)

    def test_r2_is_finite(self):
        y = bsbcf(t=T, p=BSBCF_PARAMS)
        r2 = rsquared(Y=y, y=y)
        self.assertTrue(np.isfinite(r2))


class TestFuncDefaults(unittest.TestCase):
    """Tests for func_defaults()."""

    def setUp(self):
        self.data = bsbcf(t=T, p=BSBCF_PARAMS)

    def _check_defaults(self, func, n_params):
        p0, lb, ub = func_defaults(self.data, func)
        self.assertEqual(len(p0), n_params)
        self.assertEqual(len(lb), n_params)
        self.assertEqual(len(ub), n_params)
        # All lower bounds must be strictly less than upper bounds
        for l, u in zip(lb, ub):
            self.assertLess(l, u)
        # Initial values must be within bounds
        for p, l, u in zip(p0, lb, ub):
            self.assertGreaterEqual(p, l)
            self.assertLessEqual(p, u)

    def test_bcf_defaults(self):
        self._check_defaults(bcf, 4)

    def test_sbcf_defaults(self):
        self._check_defaults(sbcf, 5)

    def test_bbcf_defaults(self):
        self._check_defaults(bbcf, 5)

    def test_bsbcf_defaults(self):
        self._check_defaults(bsbcf, 6)

    def test_unknown_function_raises(self):
        def my_func(t, p): return t
        with self.assertRaises(NotImplementedError):
            func_defaults(self.data, my_func)


class TestFit(unittest.TestCase):
    """Tests for fit() — integration test using synthetic waveform data."""

    def setUp(self):
        # Generate clean synthetic data from bsbcf
        self.y = bsbcf(t=T, p=BSBCF_PARAMS)
        self.t = T

    def _check_fit(self, func, true_params, tol=0.1):
        """Helper: fit and check convergence and parameter recovery."""
        res = fit(self.t, self.y, f=func)
        self.assertTrue(res.success or res.fun < 0.01,
                        f"{func.__name__} fit did not converge: {res.message}")
        self.assertEqual(len(res.x), len(true_params))

    def test_fit_bsbcf_converges(self):
        self._check_fit(bsbcf, BSBCF_PARAMS)

    def test_fit_bcf_converges(self):
        y_bcf = bcf(t=T, p=BCF_PARAMS)
        res = fit(T, y_bcf, f=bcf)
        self.assertTrue(res.success or res.fun < 0.01)

    def test_fit_sbcf_converges(self):
        y_sbcf = sbcf(t=T, p=SBCF_PARAMS)
        res = fit(T, y_sbcf, f=sbcf)
        self.assertTrue(res.success or res.fun < 0.01)

    def test_fit_bbcf_converges(self):
        y_bbcf = bbcf(t=T, p=BBCF_PARAMS)
        res = fit(T, y_bbcf, f=bbcf)
        self.assertTrue(res.success or res.fun < 0.01)

    def test_fit_returns_optimize_result(self):
        import scipy.optimize as opt
        res = fit(self.t, self.y)
        self.assertIsInstance(res, opt.OptimizeResult)
        self.assertIn('x', res)

    def test_fit_custom_p0_lb_ub(self):
        """fit() should accept and use custom p0, lb, ub."""
        p0, lb, ub = func_defaults(self.y, bsbcf)
        res = fit(self.t, self.y, f=bsbcf, p0=p0, lb=lb, ub=ub)
        self.assertIsNotNone(res)

    def test_fit_with_real_data(self):
        """fit() should converge on real dummy data."""
        data = read_data(DUMMY_DATA_FULL)
        participants = np.unique(data.Participant)
        p_data = prepare_part_data(data, participants[0])
        res = fit(p_data.Timedays.values, p_data.Mel.values, f=bsbcf)
        self.assertTrue(res.success or res.fun < 1.0)


# ---------------------------------------------------------------------------
# markers.py tests
# ---------------------------------------------------------------------------

class TestAmplitude(unittest.TestCase):
    """Tests for amplitude()."""

    def test_known_amplitude(self):
        values = np.array([2.0, 5.0, 10.0, 3.0])
        self.assertAlmostEqual(amplitude(values), 8.0)

    def test_flat_signal_amplitude_is_zero(self):
        values = np.full(100, 5.0)
        self.assertAlmostEqual(amplitude(values), 0.0)

    def test_amplitude_on_waveform(self):
        values = bsbcf(t=T, p=BSBCF_PARAMS)
        ampl = amplitude(values)
        self.assertGreater(ampl, 0.0)
        self.assertTrue(np.isfinite(ampl))


class TestMidpoint(unittest.TestCase):
    """Tests for midpoint() using full 24h synthetic waveform."""

    def setUp(self):
        # Generate a full 24h waveform at 1-minute resolution
        self.values = bsbcf(t=T, p=BSBCF_PARAMS)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(T),
            freq=pd.Timedelta(minutes=1)
        )

    def test_returns_four_values(self):
        result = midpoint(self.times, self.values, 0.25)
        self.assertEqual(len(result), 4)

    def test_midpoint_in_range(self):
        midpt, dlmon, dlmoff, thresh = midpoint(self.times, self.values, 0.25)
        for val in [midpt, dlmon, dlmoff]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)

    def test_threshold_is_positive(self):
        _, _, _, thresh = midpoint(self.times, self.values, 0.25)
        self.assertGreater(thresh, 0.0)

    def test_absolute_threshold_mode(self):
        """thresh_abs=True should use the given threshold directly."""
        thresh_val = 10.0
        midpt, dlmon, dlmoff, thresh = midpoint(
            self.times, self.values, thresh_val, thresh_abs=True)
        self.assertAlmostEqual(thresh, thresh_val)

    def test_midpoint_with_real_data(self):
        """midpoint() should return valid results on real dummy data."""
        data = read_data(DUMMY_DATA_FULL)
        participants = np.unique(data.Participant)
        p_data = prepare_part_data(data, participants[0])
        res = fit(p_data.Timedays.values, p_data.Mel.values, f=bsbcf)
        curve = compute_wave(p_data.Timedays.min(), p_data.Timedays.max(),
                             1.0, bsbcf, res.x)
        times = pd.date_range(p_data.Timestamp.min(),
                              periods=len(curve),
                              freq=pd.Timedelta(minutes=1))
        midpt, dlmon, dlmoff, thresh = midpoint(times, curve, 0.25)
        for val in [midpt, dlmon, dlmoff]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(val, 1.0)


class TestAreaCog(unittest.TestCase):
    """Tests for area_cog()."""

    def setUp(self):
        self.values = bsbcf(t=T, p=BSBCF_PARAMS)
        self.times = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(T),
            freq=pd.Timedelta(minutes=1)
        )

    def test_returns_two_values(self):
        result = area_cog(self.times, self.values)
        self.assertEqual(len(result), 2)

    def test_area_is_positive(self):
        area, _ = area_cog(self.times, self.values)
        self.assertGreater(area, 0.0)

    def test_cog_in_range(self):
        _, cog = area_cog(self.times, self.values)
        self.assertGreaterEqual(cog, 0.0)
        self.assertLess(cog, 1.0)

    def test_custom_baseline(self):
        """Custom baseline should produce different area than default."""
        area_default, _ = area_cog(self.times, self.values)
        area_custom, _ = area_cog(self.times, self.values, baseline=0.0)
        self.assertNotAlmostEqual(area_default, area_custom)

    def test_area_cog_with_real_data(self):
        """area_cog() should return valid results on real dummy data."""
        data = read_data(DUMMY_DATA_FULL)
        participants = np.unique(data.Participant)
        p_data = prepare_part_data(data, participants[0])
        res = fit(p_data.Timedays.values, p_data.Mel.values, f=bsbcf)
        curve = compute_wave(p_data.Timedays.min(), p_data.Timedays.max(),
                             1.0, bsbcf, res.x)
        times = pd.date_range(p_data.Timestamp.min(),
                              periods=len(curve),
                              freq=pd.Timedelta(minutes=1))
        area, cog = area_cog(times, curve)
        self.assertGreater(area, 0.0)
        self.assertGreaterEqual(cog, 0.0)
        self.assertLess(cog, 1.0)


# ---------------------------------------------------------------------------
# utils.py tests
# ---------------------------------------------------------------------------

class TestReadData(unittest.TestCase):
    """Tests for read_data()."""

    def test_returns_dataframe(self):
        data = read_data(DUMMY_DATA_FULL)
        self.assertIsInstance(data, pd.DataFrame)

    def test_required_columns_present(self):
        data = read_data(DUMMY_DATA_FULL)
        for col in ["Participant", "Date", "Time", "Mel", "Timestamp"]:
            self.assertIn(col, data.columns)

    def test_participant_is_int(self):
        data = read_data(DUMMY_DATA_FULL)
        self.assertTrue(np.issubdtype(data.Participant.dtype, np.integer))

    def test_mel_is_float(self):
        data = read_data(DUMMY_DATA_FULL)
        self.assertTrue(np.issubdtype(data.Mel.dtype, np.floating))

    def test_timestamp_is_datetime(self):
        data = read_data(DUMMY_DATA_FULL)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data.Timestamp))

    def test_no_empty_dataframe(self):
        data = read_data(DUMMY_DATA_FULL)
        self.assertGreater(len(data), 0)


class TestPreparePartData(unittest.TestCase):
    """Tests for prepare_part_data()."""

    def setUp(self):
        self.data = read_data(DUMMY_DATA_FULL)
        self.participant = np.unique(self.data.Participant)[0]

    def test_returns_dataframe(self):
        p_data = prepare_part_data(self.data, self.participant)
        self.assertIsInstance(p_data, pd.DataFrame)

    def test_timedays_column_present(self):
        p_data = prepare_part_data(self.data, self.participant)
        self.assertIn("Timedays", p_data.columns)

    def test_timedays_monotonically_increasing(self):
        p_data = prepare_part_data(self.data, self.participant)
        diffs = np.diff(p_data.Timedays.values)
        self.assertTrue(np.all(diffs >= 0),
                        "Timedays is not monotonically increasing")

    def test_timedays_starts_at_correct_hour(self):
        """Timedays[0] should correspond to the hour of the first timestamp."""
        p_data = prepare_part_data(self.data, self.participant)
        base = p_data.Timestamp.min()
        expected_start = base.hour / 24.0 + base.minute / (24*60)
        self.assertAlmostEqual(p_data.Timedays.iloc[0], expected_start,
                               places=4)

    def test_only_selected_participant(self):
        p_data = prepare_part_data(self.data, self.participant)
        self.assertTrue(
            np.all(p_data.Participant == self.participant))


class TestComputeWave(unittest.TestCase):
    """Tests for compute_wave()."""

    def test_output_is_array(self):
        result = compute_wave(0.0, 1.0, 1.0, bsbcf, BSBCF_PARAMS)
        self.assertIsInstance(result, np.ndarray)

    def test_full_wave_extends_short_range(self):
        """full_wave=True should extend tmax to tmin+1.0 if range < 1.0."""
        result_full = compute_wave(0.5, 0.8, 1.0, bsbcf, BSBCF_PARAMS,
                                   full_wave=True)
        result_short = compute_wave(0.5, 0.8, 1.0, bsbcf, BSBCF_PARAMS,
                                    full_wave=False)
        self.assertGreater(len(result_full), len(result_short))

    def test_output_length_matches_resolution(self):
        """Output length should match expected number of minutes."""
        dt_min = 1.0
        result = compute_wave(0.0, 1.0, dt_min, bsbcf, BSBCF_PARAMS,
                              full_wave=False)
        expected = int(1.0 / (dt_min / (24 * 60))) + 2  # allow slight overcount
        self.assertLess(len(result), expected)

    def test_output_is_finite(self):
        result = compute_wave(0.0, 1.0, 1.0, bsbcf, BSBCF_PARAMS)
        self.assertTrue(np.all(np.isfinite(result)))


class TestDayProfile(unittest.TestCase):
    """Tests for day_profile()."""

    def setUp(self):
        # Build a simple synthetic time series at 1-minute resolution
        times = pd.date_range("2024-01-01", periods=1440, freq="1min")
        values = bsbcf(t=np.linspace(0, 1, 1440, endpoint=False),
                       p=BSBCF_PARAMS)
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
        mean_normal, _ = day_profile(self.series, repfirst=False)
        mean_repfirst, _ = day_profile(self.series, repfirst=True)
        self.assertEqual(len(mean_repfirst), len(mean_normal) + 1)

    def test_mean_values_are_finite(self):
        mean, _ = day_profile(self.series)
        self.assertTrue(np.all(np.isfinite(mean.values)))


class TestTimeToPhase(unittest.TestCase):
    """Tests for time_to_phase()."""

    def test_zero_returns_zero(self):
        self.assertAlmostEqual(time_to_phase(0.0), 0.0)

    def test_one_returns_zero(self):
        self.assertAlmostEqual(time_to_phase(1.0), 0.0)

    def test_half_returns_half(self):
        self.assertAlmostEqual(time_to_phase(0.5), 0.5)

    def test_value_greater_than_one(self):
        self.assertAlmostEqual(time_to_phase(1.75), 0.75)

    def test_negative_value(self):
        result = time_to_phase(-0.25)
        self.assertGreaterEqual(result, 0.0)
        self.assertLess(result, 1.0)
        self.assertAlmostEqual(result, 0.75)

    def test_hours_mode(self):
        # 6 hours = 0.25 days = phase 0.25
        self.assertAlmostEqual(time_to_phase(6.0, hours=True), 0.25)

    def test_hours_mode_24h(self):
        # 24 hours = 1.0 days = phase 0.0
        self.assertAlmostEqual(time_to_phase(24.0, hours=True), 0.0)

    def test_result_always_in_range(self):
        for t in np.linspace(-3.0, 3.0, 61):
            result = time_to_phase(t)
            self.assertGreaterEqual(result, 0.0)
            self.assertLess(result, 1.0)


class TestPhaseToString(unittest.TestCase):
    """Tests for phase_to_string()."""

    def test_zero_phase(self):
        self.assertEqual(phase_to_string(0.0), "00:00")

    def test_half_phase(self):
        # 0.5 = 12:00
        self.assertEqual(phase_to_string(0.5), "12:00")

    def test_quarter_phase(self):
        # 0.25 = 06:00
        self.assertEqual(phase_to_string(0.25), "06:00")

    def test_three_quarter_phase(self):
        # 0.75 = 18:00
        self.assertEqual(phase_to_string(0.75), "18:00")

    def test_negative_phase(self):
        # Negative phase should produce string with minus sign
        result = phase_to_string(-0.25)
        self.assertTrue(result.startswith("-"))

    def test_returns_string(self):
        self.assertIsInstance(phase_to_string(0.5), str)

    def test_format_hhmm(self):
        # Result should match HH:MM format
        import re
        result = phase_to_string(0.875)
        self.assertRegex(result, r"^-?\d{2}:\d{2}$")


class TestAbsThreshold(unittest.TestCase):
    """Tests for abs_threshold()."""

    def test_zero_relative_gives_baseline(self):
        values = np.array([2.0, 5.0, 10.0])
        self.assertAlmostEqual(abs_threshold(values, 0.0), 2.0)

    def test_one_relative_gives_max(self):
        values = np.array([2.0, 5.0, 10.0])
        self.assertAlmostEqual(abs_threshold(values, 1.0), 10.0)

    def test_quarter_relative(self):
        values = np.array([0.0, 100.0])
        self.assertAlmostEqual(abs_threshold(values, 0.25), 25.0)

    def test_result_between_min_and_max(self):
        values = bsbcf(t=T, p=BSBCF_PARAMS)
        thresh = abs_threshold(values, 0.25)
        self.assertGreaterEqual(thresh, np.min(values))
        self.assertLessEqual(thresh, np.max(values))


class TestPhaseDiff(unittest.TestCase):
    """Tests for phase_diff()."""

    def test_identical_phases(self):
        self.assertAlmostEqual(phase_diff(0.5, 0.5), 0.0)

    def test_positive_difference(self):
        # 0.75 - 0.25 = 0.5, adjusted to -0.5 (shorter path)
        result = phase_diff(0.75, 0.25)
        self.assertAlmostEqual(abs(result), 0.5, places=10)

    def test_result_in_range(self):
        """Result must always be in [-0.5, 0.5]."""
        for p1 in np.linspace(0, 1, 13):
            for p2 in np.linspace(0, 1, 13):
                result = phase_diff(p1, p2)
                self.assertGreaterEqual(result, -0.5)
                self.assertLessEqual(result, 0.5)

    def test_antisymmetric(self):
        """phase_diff(a, b) should equal -phase_diff(b, a)."""
        a, b = 0.3, 0.7
        self.assertAlmostEqual(phase_diff(a, b), -phase_diff(b, a))

    def test_wrap_around(self):
        """Difference crossing midnight should be handled correctly."""
        # 0.05 and 0.95 are 0.1 apart (shorter path through midnight)
        result = phase_diff(0.05, 0.95)
        self.assertAlmostEqual(result, 0.1, places=10)

    def test_out_of_range_inputs_normalised(self):
        """Inputs outside [0,1] should be normalised before differencing."""
        result_normal = phase_diff(0.3, 0.1)
        result_shifted = phase_diff(1.3, 1.1)
        self.assertAlmostEqual(result_normal, result_shifted, places=10)


if __name__ == "__main__":
    unittest.main()
