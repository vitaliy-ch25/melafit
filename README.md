# melafit

Python package for **high-precision 24h melatonin profile analysis.** Features a variety of baseline cosine functions for curve fitting (Van Someren & Nagtegaal, 2007) and a robust cost function for superior convergence, even with sparse data (Gabel et al., 2017).

## Overview

`melafit` is a Python package designed for high-precision modeling of 24-hour melatonin secretion. While standard cosinor or harmonic analysis fail to capture the physiological nuances of the melatonin "wave," `melafit` implements several **baseline cosine functions** including bimodal, skewed and bimodal-skewed modifications. This approach accounts for the characteristic baseline, asymmetry and dual peaks often seen in high-resolution melatonin data.

Furthermore, the library utilizes a **specialized cost function** developed to overcome common optimization hurdles (trivial all-zero solutions), ensuring stable convergence even when working with sparse or incomplete time series.

## Installation

### Using Conda or Miniconda (Recommended)

To ensure all dependencies (Python 3.12, NumPy, SciPy, Pandas, etc.) are correctly configured, you can create a dedicated Python virtual environment using the provided [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml) file. Download the file [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml), navigate to the directory you downloaded it to, and execute the following commands in your terminal window:

```bash
conda env create -f melafit.yml

conda activate melafit
```

This will create a fully functional analysis environment, including a number of supporting data manipulation and analysis packages (`numpy`, `scipy`, `pandas`, `openpyxl` and `matplotlib`).

## Updating

Download the latest file [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml). In your terminal prompt, navigate to the directory where your `melafit.yml` file resides, and run the following command:

```bash
conda env update -f melafit.yml --prune
```

This updates both the dependencies and the `melafit` package itself to the latest version.

## Getting Started

Code example and some dummy data demonstrating melatonin profile curve fitting with this package are included in [./examples/](https://github.com/vitaliy-ch25/melafit/blob/main/examples/) and [./data/](https://github.com/vitaliy-ch25/melafit/blob/main/data/). Copy sample scripts and datasets to your working directory and start from there. If you have performed the steps above as described, your script will 'see' all the required packages from any location. Simply make sure to use the virtual environment `melafit` you created.

## Data preparation
Follow the Excel table format and column naming conventions as in [./data/](https://github.com/vitaliy-ch25/melafit/blob/main/data/):
* *Participant* for study participant ID
* *Date* for dates of the respective samples
* *Time* for sample timestamps 
* *Mel* for melatonin level values

## Key Features

* **Bimodal Waveform Fitting:** Implementation of the Nagtegaal & Van Someren model for superior physiological accuracy.
* **Optimized Convergence:** Leverages the robust cost function described in Gabel et al. (2017) to ensure reliable fits across diverse datasets.
* **Sparse Data Support:** Capable of reconstructing full profiles and estimating circadian phase from limited data points, as well as determing dim light melatonin onset (DLMO) with partial data.
* **Research-Ready:** Direct derivation of phase markers from continuous, fitted waveforms.

## Scientific Foundations

If you use `melafit` in your research, please cite the following foundational publications:

### Human-Readable
1. [Van Someren, E. J., & Nagtegaal, E. (2007). Improving melatonin circadian phase estimates. Sleep Medicine, 8(6), 590-601.](https://doi.org/10.1016/j.sleep.2007.03.012)
2. [Gabel, V., et al. (2017). Differential impact in young and older individuals of blue-enriched white light on circadian physiology and alertness during sustained wakefulness. Scientific Reports, 7, 7620.](https://doi.org/10.1038/s41598-017-07060-8)

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
  pages={7620},
  year={2017},
  publisher={Nature Publishing Group}
}
```

## Authors
* Vitaliy Kolodyazhniy – Lead Developer
* Christian Cajochen – Scientific Lead

## Revision History

### v0.0.9
- New function `func_defaults()` in `fitting.py` for standalone access to
  default initial conditions and constraints for all waveform functions
- Improved cost function: `eps` parameter for more robust fitting
- Optional `thresh_abs` parameter in `markers.midpoint()` for absolute
  threshold support
- New example script `example_dlmo.py` and dataset for DLMO detection
  from partial data
- Previous example renamed to `example_full_profile.py`
- Improved type hints, docstrings and README

### Previous revisions (v0.0.1 – v0.0.8)
- Full implementation of melatonin profile analysis as described in
  [Gabel et al. (2017)](https://doi.org/10.1038/s41598-017-07060-8)
- Waveform functions: `bcf`, `sbcf`, `bbcf`, `bsbcf`
- Markers: `amplitude`, `midpoint`, `area_cog`
- Utilities: `read_data`, `prepare_part_data`, `compute_wave`,
  `day_profile`, `abs_threshold`, `time_to_phase`, `phase_to_string`,
  `phase_diff`
- MIT license, packaging metadata and README

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.