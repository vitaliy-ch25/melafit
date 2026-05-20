"""
melafit: Melatonin Data Fitting and Circadian Rhythm Analysis

This is a Python package for fitting circadian rhythm melatonin data using
parametric wave approximation models and computing circadian phase markers.

The package provides functionality for:
- Fitting melatonin concentration data using four cosine-based models:
  * BCF (Baseline Cosine Function) - Simple cosine model
  * SBCF (Skewed Baseline Cosine Function) - Cosine model with skewness
  * BBCF (Bimodal Baseline Cosine Function) - Bimodal cosine model
  * BSBCF (Bimodal Skewed BCF) - Bimodal model with skewness
- Computing circadian phase markers including:
  * DLMOn and DLMOff (Dim Light Melatonin Onset/Offset) times
  * DLMO (melatonin midpoint)
  * Amplitude and area under the curve
  * Center of gravity of melatonin secretion
- Data I/O, preprocessing, and utility functions for circadian analysis

Submodules:
-----------
fitting : Melatonin wave approximation functions and curve fitting
markers : Circadian phase marker computation (DLMOn/Off, amplitude, etc.)
utils : Data I/O, preprocessing, and utility functions

References:
-----------
- Ruf et al. (1992). "The baseline cosinus function: a periodic regression 
  model for biological rhythms" https://doi.org/10.1076/brhm.27.2.153.12942
- Van Someren & Nagtegaal (2007). "Improving melatonin circadian phase 
  estimates" https://doi.org/10.1016/j.sleep.2007.03.012
- Gabel et al. (2017). "Differential impact in young and older individuals of 
  blue-enriched white light on circadian physiology and alertness during 
  sustained wakefulness" https://doi.org/10.1038/s41598-017-07060-8
"""

from .fitting import *
from .markers import *
from .utils import *