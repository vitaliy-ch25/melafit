# melafit

**High-precision 24h melatonin profile analysis.** Features bimodal skewed baseline cosine fitting (Van Someren & Nagtegaal, 2007) and a robust cost function for superior convergence, even with sparse data (Gabel et al., 2017).

## Overview

`melafit` is a Python library designed for high-precision modeling of 24-hour melatonin secretion. While standard cosinor analysis often fails to capture the physiological nuances of the melatonin "wave," `melafit` implements the **bimodal skewed baseline cosine function**. This approach accounts for the characteristic asymmetry and dual peaks often seen in high-resolution endocrine data.

Furthermore, the library utilizes a **specialized cost function** developed to overcome common optimization hurdles (local minima), ensuring stable convergence even when working with sparse or incomplete time series.

## Installation

### Using Conda or Miniconda (Recommended)

To ensure all dependencies (Python 3.12, NumPy, SciPy, Pandas, etc.) are correctly configured, you can create a dedicated environment using the provided `melafit.yml` file:

```bash
# Create the environment from the yaml file
conda env create -f melafit.yml

# Activate the environment
conda activate melafit
```

Note: A full-fledged Python package will be available soon.

## Key Features

* **Bimodal Waveform Fitting:** Implementation of the Nagtegaal & Van Someren model for superior physiological accuracy.
* **Optimized Convergence:** Leverages the robust cost function described in Gabel et al. (2017) to ensure reliable fits across diverse datasets.
* **Sparse Data Support:** Capable of reconstructing full profiles and estimating circadian phase from limited data points.
* **Research-Ready:** Direct derivation of phase markers from continuous, fitted waveforms.

## Scientific Foundations

If you use `melafit` in your research, please cite the following foundational publications:

### Human-Readable
1. **Van Someren, E. J., & Nagtegaal, E. (2007).** Improving melatonin circadian phase estimates. *Sleep Medicine*, 8(6), 590-601. [https://doi.org/10.1016/j.sleep.2007.03.012]
2. **Gabel, V., et al. (2017).** Differential impact in young and older individuals of blue-enriched white light on circadian physiology and alertness during sustained wakefulness. *Scientific Reports*, 7(1), 7620. [https://doi.org/10.1038/s41598-017-07060-8]

### BibTeX
```bibtex
@article{vansomeren2007,
  title={Improving melatonin circadian phase estimates},
  author={Van Someren, Eus JW and Nagtegaal, Elsbeth},
  journal={Sleep Medicine},
  volume={8},
  number={6},
  pages={590--601},
  year={2007},
  publisher={Elsevier}
}

@article{gabel2017,
  title={Differential impact in young and older individuals of blue-enriched white light on circadian physiology and alertness during sustained wakefulness},
  author={Gabel, Virginie and Reichert, Carolin F and Maire, Micheline and Schmidt, Christina and Schlangen, Luc JM and Kolodyazhniy, Vitaliy and Garbazza, Corrado and Cajochen, Christian and Viola, Antoine U},
  journal={Scientific Reports},
  volume={7},
  number={1},
  pages={7620},
  year={2017},
  publisher={Nature Publishing Group}
}