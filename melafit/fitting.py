"""
melafit.fitting: Melatonin Wave Approximation Models and Fitting Routines

This module provides parametric cosine-based models for approximating melatonin
concentration curves and methods for fitting these models to experimental data.

Models:
-------
- BCF (Baseline Cosine Function)
  * Simple baseline cosine model suitable for symmetric melatonin profiles
  * Parameters: phi (phase), b (baseline), H (height), c (width)
  * Reference: [Ruf '92](https://doi.org/10.1076/brhm.27.2.153.12942)

- SBCF (Skewed Baseline Cosine Function)
  * Baseline cosine model with skewness parameter for asymmetric profiles
  * Parameters: phi, b, H, c, v (skewness)
  * Reference: [Van Someren & Nagtegaal '07](https://doi.org/10.1016/j.sleep.2007.03.012)

- BBCF (Bimodal Baseline Cosine Function)
  * Bimodal baseline cosine model with two peaks for biphasic melatonin
    secretion patterns
  * Parameters: phi, b, H, c, m (bimodality)
  * Reference: [Van Someren & Nagtegaal '07](https://doi.org/10.1016/j.sleep.2007.03.012)

- BSBCF (Bimodal Skewed Baseline Cosine Function)
  * Most complex model combining bimodality and skewness
  * Parameters: phi, b, H, c, v (skewness), m (bimodality)
  * Reference: [Van Someren & Nagtegaal '07](https://doi.org/10.1016/j.sleep.2007.03.012)

Functions:
----------
- bcf, sbcf, bbcf, bsbcf : Waveform model functions
- params_to_array, array_to_params : Convert between dict and array
  parameter formats
- cost : Cost function for curve fitting with trivial solution penalty
- rsquared : R² goodness of fit metric
- func_defaults : Generate default initial conditions and parameter bounds
- fit : Main routine for fitting melatonin data using scipy optimization

Constants:
----------
- BCF_PARAM_NAMES, SBCF_PARAM_NAMES, BBCF_PARAM_NAMES, BSBCF_PARAM_NAMES : 
    Parameter name lists for each model
- BUILTIN_PARAM_NAMES : Mapping of built-in functions to their parameter names
"""

import scipy.optimize as opt
import numpy as np
import pandas as pd
from collections.abc import Mapping
from melafit.results import FitResult

# Parameter names for melatonin wave approximation functions
BCF_PARAM_NAMES   = ["phi", "b", "H", "c"]
SBCF_PARAM_NAMES  = ["phi", "b", "H", "c", "v"]
BBCF_PARAM_NAMES  = ["phi", "b", "H", "c", "m"]
BSBCF_PARAM_NAMES = ["phi", "b", "H", "c", "v", "m"]

def _resolve_params(p: np.ndarray | Mapping) -> np.ndarray:
    """
    Convert parameter mapping to array if needed, pass array through unchanged.
    Accepts plain dicts, FitResult, or any Mapping.
    """
    if isinstance(p, Mapping):
        return np.array(list(p.values()))
    return p

def bcf(t: np.ndarray,
        p: Mapping | np.ndarray) -> np.ndarray:
    """
    Baseline cosine function
    [Ruf '92](https://doi.org/10.1076/brhm.27.2.153.12942)

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the BCF waveform
        p : Dictionary or Numpy array of floats
            BCF parameters phi, b, H, c

    Returns
    -------
        bcf_val : Numpy array of floats
            Values of the BCF function for the respective time points

    See also
    ---------
        :func:`sbcf`, :func:`bbcf`, :func:`bsbcf`: Skewed, bimodal and bimodal
        skewed baseline cosine functions
    """

    p = _resolve_params(p)
    
    phi = p[0]
    b = p[1]
    H = p[2]
    c = p[3]

    phi = 2 * np.pi * phi
    t = 2 * np.pi * t

    bcf_val = b + H / (2 * (1 - c)) * (
        np.cos(t - phi) - c + abs(np.cos(t - phi) - c))
    
    return bcf_val

def sbcf(t: np.ndarray,
         p: Mapping | np.ndarray) -> np.ndarray:
    """
    Skewed baseline cosine function
    [Van Someren & Nagtegaal '07](https://doi.org/10.1016/j.sleep.2007.03.012)

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the SBCF waveform
        p : Dictionary or Numpy array of floats
            SBCF parameters phi, b, H, c, v
    
    Returns
    -------
        sbcf_val : Numpy array of floats
            Values of the SBCF function for the respective time points

    See also
    ---------
        :func:`bcf`, :func:`bbcf`, :func:`bsbcf`: Baseline cosine function and 
        its bimodal and bimodal skewed modifications
    """

    p = _resolve_params(p)
    
    phi = p[0]
    b = p[1]
    H = p[2]
    c = p[3]
    v = p[4]

    phi = 2 * np.pi * phi
    t = 2 * np.pi * t

    sbcf_val = b + H / (2 * (1 - c)) * (
        np.cos(t - phi + v * np.cos(t - phi)) - c + 
        abs(np.cos(t - phi + v * np.cos(t - phi)) - c))
    
    return sbcf_val

def bbcf(t: np.ndarray,
         p: Mapping | np.ndarray) -> np.ndarray:
    """
    Bimodal baseline cosine function
    [Van Someren & Nagtegaal '07](https://doi.org/10.1016/j.sleep.2007.03.012)

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the BBCF waveform
        p : Dictionary or Numpy array of floats
            BBCF parameters phi, b, H, c, m

    Returns
    -------
        bbcf_val : Numpy array of floats
            Values of the BBCF function for the respective time points

    See also
    ---------
        :func:`bcf`, :func:`sbcf`, :func:`bsbcf`: Baseline cosine function and 
        its skewed and bimodal skewed modifications
    """

    p = _resolve_params(p)
    
    phi = p[0]
    b = p[1]
    H = p[2]
    c = p[3]
    m = p[4]

    phi = 2 * np.pi * phi
    t = 2 * np.pi * t

    bbcf_val = b + H / (2 * (1 - c)) * (
        np.cos(t - phi) + m * np.cos(2 * t - 2 * phi - np.pi) - c + 
        abs(np.cos(t - phi) + m * np.cos(2 * t - 2 * phi - np.pi) - c))

    return bbcf_val

def bsbcf(t: np.ndarray,
          p: Mapping | np.ndarray) -> np.ndarray:
    """
    Bimodal skewed baseline cosine function
    [Van Someren & Nagtegaal '07](https://doi.org/10.1016/j.sleep.2007.03.012)

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the bsbcf waveform
        p : Dictionary or Numpy array of floats
            BSBCF parameters phi, b, H, c, v, m

    Returns
    -------
        bsbcf_val : Numpy array of floats
            Values of the BSBCF function for the respective time points

    See also
    ---------
        :func:`bcf`, :func:`sbcf`, :func:`bbcf`: Baseline cosine function and 
        its skewed and bimodal modifications
    """
    
    p = _resolve_params(p)

    phi = p[0]
    b = p[1]
    H = p[2]
    c = p[3]
    v = p[4]
    m = p[5]

    phi = 2 * np.pi * phi
    t = 2 * np.pi * t

    bsbcf_val = b + H / (2 * (1 - c)) * (
        np.cos(t - phi + v * np.cos(t - phi)) + 
        m * np.cos(2 * t - 2 * phi - np.pi) - c + 
        abs(np.cos(t - phi + v * np.cos(t - phi)) + 
            m * np.cos(2 * t - 2 * phi - np.pi) - c))

    return bsbcf_val

# Mapping of functions to built-in parameter names for conversion between dict
# and array representations
BUILTIN_PARAM_NAMES = {
    bcf:   BCF_PARAM_NAMES,
    sbcf:  SBCF_PARAM_NAMES,
    bbcf:  BBCF_PARAM_NAMES,
    bsbcf: BSBCF_PARAM_NAMES,
}

def params_to_array(params: Mapping) -> np.ndarray:
    """
    Convert parameter dictionary to numpy array for scipy.optimize.

    Parameters
    ----------
        params : Mapping
            Dictionary of parameter names and values
    Returns
    -------
        p : Numpy array of floats
            Parameter vector for scipy.optimize

    See also
    ---------
        :func:`array_to_params`: Convert array to parameter dictionary
    """
    return np.array(list(params.values()))

def array_to_params(x: np.ndarray, f: callable) -> dict:
    """
    Convert scipy.optimize result array to named parameter dictionary.

    Parameters
    ----------
        x : Numpy array of floats
            Parameter vector from scipy.optimize
        f : callable
            Melatonin wave approximation function for which the
            parameters were fitted
    Returns
    -------
        params : dict
            Dictionary of parameter names and values for the respective function

    See also
    ---------
        :func:`params_to_array`: Convert parameter dictionary to numpy array
    """

    param_names = BUILTIN_PARAM_NAMES.get(f)
    if param_names is None:
        raise ValueError(
            f"Function {f.__name__} not recognized for parameter conversion.")
    return dict(zip(param_names, x))

def cost(p: np.ndarray,
         t: np.ndarray,
         y: np.ndarray,
         f: callable,
         cost_p : dict | None = None) -> np.float64:
    """
    Cost function for melatonin fitting, penalizes the trivial solution when
    all model values = 0
    [Gabel et al. '17](https://doi.org/10.1038/s41598-017-07060-8)
    NOTE: the order of parameters is pre-defined by the SciPy optimization
    routine

    Parameters
    ----------
        p : Numpy array of floats
            Function parameter vector
        t : Numpy array of floats
            X-values for curve fitting (time)
        y: Numpy array of floats
            Y-values for curve fitting (melatonin levels)
        f : callable
            Melatonin wave approximation function
        cost_p : dict | None
            Cost function parameters (defaults to None) in which case
            {"eps": 1e-8} is used

    Returns
    -------
        val : float
            Value of the cost function
    """

    if cost_p is None:
        cost_p = {}
    eps = cost_p.get("eps", 1e-8)

    y_ = f(t, p)

    return np.nanmean(np.square(y - y_)) / (np.var(y_) + eps)

def rsquared(Y: np.ndarray,
             y: np.ndarray) -> np.float64:
    """
    R2 goodness of fit

    Parameters
    ----------
        Y : Numpy array of floats
            Reference values
        y : Numpy array of floats
            Fitted values

    Returns
    -------
        r2 : float
            R² value
    """

    err = Y - y
    Y_ = Y - np.nanmean(Y)
    r2 = 1 - np.nansum(np.square(err)) / np.nansum(np.square(Y_))

    return r2

def func_defaults(data_fit: np.ndarray,
                  f: callable) -> tuple[dict, dict, dict]:
    """
    Default initial conditions and constraints for melatonin wave approximation
    functions

    Parameters
    ----------
        data_fit : Numpy array of floats
            Y-values for curve fitting (melatonin levels)
        f : callable
            Melatonin wave approximation function

    Returns
    -------
        p0 : Dictionary
            Initial guess for the function parameters
        lb : Dictionary
            Lower bounds for the function parameters
        ub : Dictionary
            Upper bounds for the function parameters
    """

    minx = np.min(data_fit)
    maxx = np.max(data_fit)

    data_range = (maxx - minx)

    if f==bcf:
        # Initial guess for BCF parameters
        p0 = [
            0, # phi
            minx, # b
            (maxx-minx), # H
            0 # c
        ]
            
        # Lower bounds for BCF parameters
        lb = [
            -0.5, # phi
            minx, # b
            0.5 * data_range, # H
            -1 # c
        ]

        # Upper bounds for BCF parameters
        ub = [
            0.5, # phi
            maxx, # b
            2 * data_range, # H
            1 - 1e-6 # c
        ]
    elif f==sbcf:
        # Initial guess for SBCF parameters
        p0 = [
            0, # phi
            minx, # b
            (maxx-minx), # H
            0, # c
            0 # v
        ]
            
        # Lower bounds for SBCF parameters
        lb = [
            -0.5, # phi
            minx, # b
            0.5 * data_range, # H
            -1, # c
            -1 # v
        ]

        # Upper bounds for SBCF parameters
        ub = [
            0.5, # phi
            maxx, # b
            2 * data_range, # H
            1 - 1e-6, # c
            1 # v
        ]
    elif f==bbcf:
        # Initial guess for BBCF parameters
        p0 = [
            0, # phi
            minx, # b
            (maxx-minx), # H
            0, # c
            0 # m
        ]
        
        # Lower bounds for BBCF parameters
        lb = [
            -0.5, # phi
            minx, # b
            0.5 * data_range, # H
            -1, # c
            0 # m
        ]

        # Upper bounds for BBCF parameters
        ub = [
            0.5, # phi
            maxx, # b
            2 * data_range, # H
            1 - 1e-6, # c
            1 - 1e-6 # m
        ]
    elif f==bsbcf:
        # Initial guess for BSBCF parameters
        p0 = [
            0, # phi
            minx, # b
            (maxx-minx), # H
            0, # c
            0, # v
            0 # m
        ]
            
        # Lower bounds for BSBCF parameters
        lb = [
            -0.5, # phi
            minx, # b
            0.5 * data_range, # H
            -1, # c
            -1, # v
            0 # m
        ]

        # Upper bounds for BSBCF parameters
        ub = [
            0.5, # phi
            maxx, # b
            2 * data_range, # H
            1 - 1e-6, # c
            1, # v
            1 - 1e-6 # m
        ]
    else:
        raise NotImplementedError("Constraints and initial conditions for " + 
                                  f"function '{f.__name__}' are not defined!")
    
    return (array_to_params(p0, f),
            array_to_params(lb, f),
            array_to_params(ub, f))

def fit(time_fit: np.ndarray | pd.Series,
        data_fit: np.ndarray,
        f: callable=bsbcf,
        p0: dict | np.ndarray | None = None,
        lb: dict | np.ndarray | None = None,
        ub: dict | np.ndarray | None = None,
        cost_f: callable=cost,
        cost_p: dict | None = None) -> FitResult:
    """
    Melatonin data fitting routine

    Parameters
    ----------
        time_fit : np.ndarray or pd.Series
            X-values for curve fitting (time). Accepts a float array of
            matplotlib date numbers (e.g. from :func:`gen_time_range`) or
            a datetime64 array / pandas Timestamp Series, which is converted
            automatically via matplotlib.dates.date2num
        data_fit : Numpy array of floats
            Y-values for curve fitting (melatonin levels)
        f : callable
            Melatonin wave approximation function (defaults to `bsbcf`)
        p0 : dict, Numpy array of floats, or None
            Non-standard initial values for wave approximation function
            (defaults to 'None')
        lb : dict, Numpy array of floats, or None
            Non-standard lower bounds for wave approximation function
            parameters (defaults to 'None')
        ub : dict, Numpy array of floats, or None
            Non-standard upper bounds for wave approximation function
            parameters (defaults to 'None')
        cost_f : callable
            Cost function for curve fitting (defaults to `cost`)
        cost_p : dict | None
            Cost function parameters as dictionary or None (defaults to None)

    Returns
    -------
        res : FitResult
            Optimization result including parameters of the fitted function
            in the field `x`

    See also
    ---------
        :func:`cost`, :func:`func_defaults`: Cost function, default initial 
        conditions and bounds for fitting
    """

    if hasattr(time_fit, 'dtype') and np.issubdtype(time_fit.dtype, np.datetime64):
        from matplotlib import dates as mdates
        time_fit = mdates.date2num(time_fit)

    # Only try to fetch defaults if we recognize the function
    if f in BUILTIN_PARAM_NAMES.keys():
        _p0, _lb, _ub = func_defaults(data_fit, f)

        if p0 is not None:
            _p0 = p0

        if lb is not None:
            _lb = lb

        if ub is not None:
            _ub = ub
    else:
        # For custom functions, require the user to have provided p0/lb/ub
        if p0 is None or lb is None or ub is None:
            raise ValueError(
                f"Function '{f.__name__}' is not a built-in model. "
                "You must provide p0, lb, and ub manually.")
        _p0, _lb, _ub = p0, lb, ub

    bounds = opt.Bounds(_resolve_params(_lb), _resolve_params(_ub))
    res = opt.minimize(fun=cost_f,
                       args=(time_fit, data_fit, f, cost_p),
                       x0=_resolve_params(_p0),
                       bounds=bounds)
    
    if f in BUILTIN_PARAM_NAMES:
        param_names = BUILTIN_PARAM_NAMES[f]
    else:
        if isinstance(_p0, dict):
            param_names = list(_p0.keys())
        elif isinstance(_lb, dict):
            param_names = list(_lb.keys())
        elif isinstance(_ub, dict):
            param_names = list(_ub.keys())
        else:
            param_names = None

    r2 = rsquared(data_fit, f(t=time_fit, p=res.x))

    return FitResult(result=res, wave_func=f, param_names=param_names, r2=r2)
