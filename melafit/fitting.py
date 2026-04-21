import scipy.optimize as opt
import numpy as np

def bsbcf(t: np.ndarray[np.float64],
          phi: np.float64,
          b: np.float64,
          H: np.float64,
          c: np.float64,
          v: np.float64,
          m: np.float64) -> np.ndarray[np.float64]:
    """
    Bimodal skewed baseline cosine function (Van Someren & Nagtegaal, 2007)

    Parameters
    ----------
        t : Numpy array of floats
            Cumulative time in days
        phi : float
            phase [-0.5 0.5], 0.5 = 12h
        b : float
            Baseline >=0
        H : float
            Height >0
        c : float
            Width [-1,1)
        v : float
            Skewness [-0.5 0.5]
        m : float
            Bimodality [0 1)

    Returns
    -------
        bsbcf_val : Numpy array of floats
            Values of the BSBCF function for the respective time points
    """
    
    phi = 2 * np.pi * phi
    t = 2 * np.pi * t

    bsbcf_val = b + H / (2 * (1 - c)) * (
        np.cos(t - phi + v * np.cos(t - phi)) + 
        m * np.cos(2 * t - 2 * phi - np.pi) - c + 
        abs(np.cos(t - phi + v * np.cos(t - phi)) + 
            m * np.cos(2 * t - 2 * phi - np.pi) - c))

    return bsbcf_val

def cost(p: np.ndarray[np.float64],
         t: np.ndarray[np.float64],
         y: np.ndarray[np.float64]) -> np.float64:
    """
    Cost function for melatonin fitting, penalizes the trivial solution when
    all model values = 0 (Gabel et al., 2017)

    Parameters
    ----------
        p : Numpy array of floats
            Function parameter vector
        t : Numpy array of floats
            X-values for curve fitting (time)
        y: Numpy array of floats
            Y-values for curve fitting (melatonin levels)

    Returns
    -------
        val : float
            Value of the cost function
    """

    phi = p[0]
    b = p[1]
    H = p[2]
    c = p[3]
    v = p[4]
    m = p[5]

    y_ = bsbcf(t, phi, b, H, c, v, m)

    return np.nanmean(np.square(y - y_)) / np.var(y_)

def rsquared(Y: np.ndarray[np.float64],
             y: np.ndarray[np.float64]) -> np.float64:
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
            R2 value
    """

    err = Y - y
    Y_ = Y - np.nanmean(Y)
    r2 = 1 - np.nansum(np.square(err)) / np.nansum(np.square(Y_))

    return r2

def fit(time_fit: np.ndarray[np.float64],
        data_fit: np.ndarray[np.float64],
        cost: callable=cost) -> np.ndarray[np.float64]:
    """
    Melatonin data fitting routine

    Parameters
    ----------
        time_fit : Numpy array of floats
            X-values for curve fitting (time)
        data_fit : Numpy array of floats
            Y-values for curve fitting (melatonin levels)
        cost : callable
            Cost function for curve fitting (defaults to `cost`)

    Returns
    -------
        res : Numpy array of floats
            Parameters of the fitted function
    """

    minx = data_fit.min()
    maxx = data_fit.max()
    range = (maxx - minx)

    # Initial guess for BSBCF parameters
    param0 = [
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
        0.5 * range, # H
        -1, # c
        -1, # v
        0 # m
    ]

    # Upper bounds for BSBCF parameters
    ub = [
        0.5, # phi
        maxx, # b
        2 * range, # H
        1 - 1e-6, # c
        1, # v
        1 - 1e-6 # m
    ]

    bounds = opt.Bounds(lb, ub)
    res = opt.minimize(fun=cost,
                       args=(time_fit, data_fit),
                       x0=param0,
                       bounds=bounds)

    return res