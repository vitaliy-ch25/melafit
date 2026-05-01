import numpy as np
import pandas as pd

def amplitude(wave: np.array) -> np.float64:
    """
    Peak-to-baseline amplitude of fitted waveform

    Parameters
    ----------
        wave : Numpy array of floats
            Waveform values

    Returns
    -------
        val : float
            Peak-to-baseline amplitude
    """

    return np.max(wave) - np.min(wave)