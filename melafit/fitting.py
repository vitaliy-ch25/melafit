import scipy.optimize as opt
import numpy as np

def bcf(t: np.ndarray,
        p: np.ndarray) -> np.ndarray:
    """
    Baseline cosine function (Ruf, 1992)
    [https://doi.org/10.1076/brhm.27.2.153.12942]

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the BCF waveform
        p : Numpy array of floats
            BCF parameters phi, b, H, c

    Returns
    -------
        bcf_val : Numpy array of floats
            Values of the BCF function for the respective time points
    """

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
         p: np.ndarray) -> np.ndarray:
    """
    Skewed baseline cosine function (Van Someren & Nagtegaal, 2007)
    [https://doi.org/10.1016/j.sleep.2007.03.012]

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the SBCF waveform
        p : Numpy array of floats
            SBCF parameters phi, b, H, c, v
    
    Returns
    -------
        sbcf_val : Numpy array of floats
            Values of the SBCF function for the respective time points
    """

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
         p: np.ndarray) -> np.ndarray:
    """
    Bimodal baseline cosine function (Van Someren & Nagtegaal, 2007)
    [https://doi.org/10.1016/j.sleep.2007.03.012]

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the BBCF waveform
        p : Numpy array of floats
            BBCF parameters phi, b, H, c, m

    Returns
    -------
        bbcf_val : Numpy array of floats
            Values of the BBCF function for the respective time points
    """

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
          p: np.ndarray) -> np.ndarray:
    """
    Bimodal skewed baseline cosine function (Van Someren & Nagtegaal, 2007)
    [https://doi.org/10.1016/j.sleep.2007.03.012]

    Parameters
    ----------
        t : Numpy array of floats
            Time values for the bsbcf waveform
        p : Numpy array of floats
            BSBCF parameters phi, b, H, c, v, m

    Returns
    -------
        bsbcf_val : Numpy array of floats
            Values of the BSBCF function for the respective time points
    """
    
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

def cost(p: np.ndarray,
         t: np.ndarray,
         y: np.ndarray,
         f: callable) -> np.float64:
    """
    Cost function for melatonin fitting, penalizes the trivial solution when
    all model values = 0 (Gabel et al., 2017)
    [https://doi.org/10.1038/s41598-017-07060-8]. NOTE: the order of
    parameters is pre-defined by the SciPy optimization routine

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

    Returns
    -------
        val : float
            Value of the cost function
    """

    y_ = f(t, p)

    return np.nanmean(np.square(y - y_)) / np.var(y_)

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
            R2 value
    """

    err = Y - y
    Y_ = Y - np.nanmean(Y)
    r2 = 1 - np.nansum(np.square(err)) / np.nansum(np.square(Y_))

    return r2

def fit(time_fit: np.ndarray,
        data_fit: np.ndarray,
        f: callable=bsbcf,
        cost_f: callable=cost,
        p0: np.ndarray = None,
        lb: np.ndarray = None,
        ub: np.ndarray = None) -> opt.OptimizeResult:
    """
    Melatonin data fitting routine

    Parameters
    ----------
        time_fit : Numpy array of floats
            X-values for curve fitting (time)
        data_fit : Numpy array of floats
            Y-values for curve fitting (melatonin levels)
        f : callable
            Melatonin wave approximation function (defaults to `bsbcf`)
        cost_f : callable
            Cost function for curve fitting (defaults to `cost`)
        p0 : Numpy array of floats
            Non-standard initial values for wave approximation function
            (defaults to 'None')
        lb : Numpy array of floats
            Non-standard lower bounds for wave approximation function
            parameters (defaults to 'None')
        ub : Numpy array of floats
            Non-standard upper bounds for wave approximation function
            parameters (defaults to 'None')

    Returns
    -------
        res : OptimizeResult
            Optimization result including parameters of the fitted function
            in the field `x`
    """

    minx = data_fit.min()
    maxx = data_fit.max()
    data_range = (maxx - minx)

    if f==bcf:
        if p0 is None:
            # Initial guess for BCF parameters
            p0 = [
                0, # phi
                minx, # b
                (maxx-minx), # H
                0 # c
                ]
            
        if lb is None:
            # Lower bounds for BCF parameters
            lb = [
                -0.5, # phi
                minx, # b
                0.5 * data_range, # H
                -1 # c
            ]

        if ub is None:
            # Upper bounds for BCF parameters
            ub = [
                0.5, # phi
                maxx, # b
                2 * data_range, # H
                1 - 1e-6 # c
            ]
    elif f==sbcf:
        if p0 is None:
            # Initial guess for SBCF parameters
            p0 = [
                0, # phi
                minx, # b
                (maxx-minx), # H
                0, # c
                0 # v
                ]
            
        if lb is None:
            # Lower bounds for SBCF parameters
            lb = [
                -0.5, # phi
                minx, # b
                0.5 * data_range, # H
                -1, # c
                -1 # v
            ]

        if ub is None:
            # Upper bounds for SBCF parameters
            ub = [
                0.5, # phi
                maxx, # b
                2 * data_range, # H
                1 - 1e-6, # c
                1 # v
            ]
    elif f==bbcf:
        if p0 is None:
            # Initial guess for BBCF parameters
            p0 = [
                0, # phi
                minx, # b
                (maxx-minx), # H
                0, # c
                0 # m
                ]
            
        if lb is None:
            # Lower bounds for BBCF parameters
            lb = [
                -0.5, # phi
                minx, # b
                0.5 * data_range, # H
                -1, # c
                0 # m
            ]

        if ub is None:
            # Upper bounds for BBCF parameters
            ub = [
                0.5, # phi
                maxx, # b
                2 * data_range, # H
                1 - 1e-6, # c
                1 - 1e-6 # m
            ]
    elif f==bsbcf:
        if p0 is None:
            # Initial guess for BSBCF parameters
            p0 = [
                0, # phi
                minx, # b
                (maxx-minx), # H
                0, # c
                0, # v
                0 # m
                ]
            
        if lb is None:
            # Lower bounds for BSBCF parameters
            lb = [
                -0.5, # phi
                minx, # b
                0.5 * data_range, # H
                -1, # c
                -1, # v
                0 # m
            ]

        if ub is None:
            # Upper bounds for BSBCF parameters
            ub = [
                0.5, # phi
                maxx, # b
                2 * data_range, # H
                1 - 1e-6, # c
                1, # v
                1 - 1e-6 # m
            ]
    elif (p0 is None) or (lb is None) or (ub is None):
        raise NotImplementedError(f"Constraints or initial conditions for " + 
                                  f"function '{f.__name__}' are not defined!")

    bounds = opt.Bounds(lb, ub)
    res = opt.minimize(fun=cost_f,
                       args=(time_fit, data_fit, f),
                       x0=p0,
                       bounds=bounds)

    return res