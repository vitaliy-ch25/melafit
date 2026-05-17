# melafit

Python package for **high-precision circadian melatonin profile analysis.** Features a variety of baseline cosine functions for curve fitting ([Van Someren & Nagtegaal, 2007](https://doi.org/10.1016/j.sleep.2007.03.012)) and a robust cost function for superior convergence, even with sparse data ([Gabel et al., 2017](https://doi.org/10.1038/s41598-017-07060-8)).

## Overview

[melafit](https://github.com/vitaliy-ch25/melafit) is a Python package designed for high-precision modeling of 24-hour melatonin secretion. While standard cosinor or harmonic analyses fail to capture the physiological nuances of the melatonin "wave," [melafit](https://github.com/vitaliy-ch25/melafit) implements several **baseline cosine functions** including bimodal, skewed and bimodal-skewed modifications. This approach accounts for the characteristic baseline, asymmetry and dual peaks often seen in high-resolution circadian melatonin data.

Furthermore, the library utilizes a **specialized cost function** developed to overcome common optimization hurdles (trivial all-zero solutions), ensuring stable convergence even when working with sparse or incomplete time series.

## Key Features

* **Bimodal Waveform Fitting:** Implementation of the [Van Someren & Nagtegaal (2007)](https://doi.org/10.1016/j.sleep.2007.03.012) model for superior physiological accuracy.
* **Optimized Convergence:** Leverages the robust cost function described in [Gabel et al. (2017)](https://doi.org/10.1038/s41598-017-07060-8) to ensure reliable fits across diverse datasets.
* **Sparse Data Support:** Capable of reconstructing full profiles and estimating circadian phase from limited data points, as well as determining dim light melatonin onset (DLMO) with partial data.
* **Research-Ready:** Direct derivation of `Amplitude`, `DLMOn`, `DLMOff`, `Midpoint`, `Area` and `COG` markers from continuous, fitted waveforms.

## Installation

`melafit` is available on [PyPI](https://pypi.org/project/melafit/) and can be installed with `pip`. However, installing directly into your system Python environment without a virtual environment is strongly discouraged, as it may cause conflicts with other packages. The recommended approach is to use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) as the package and environment manager to create a dedicated virtual environment, as described below.

### Standard installation

Download file [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml) to a directory of your choice (`<YOUR-DIRECTORY>`). Navigate to the directory, create and activate the conda environment:

```bash
cd <YOUR-DIRECTORY>
conda env create -f melafit.yml
conda activate melafit
```

The environment configuration file [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml) uses `conda-forge` as the sole package channel, ensuring reproducibility and avoiding potential conflicts between packages from different channels. `melafit` itself is installed from [PyPI](https://pypi.org/project/melafit/) via `pip` as part of the environment setup. This will create a fully functional analysis environment, including all supporting packages (`numpy`, `scipy`, `pandas`, `openpyxl` and `matplotlib`).

### Developer installation

If you intend to follow the development closely or contribute to the package, clone the repository first to a dedicated directory `<YOUR-DIRECTORY>`. Navigate to it and clone the repository as follows:

```bash
cd <YOUR-DIRECTORY>
git clone https://github.com/vitaliy-ch25/melafit.git
cd melafit
```

Then create and activate the conda environment using the developer configuration file [`melafit-dev.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit-dev.yml), which installs `melafit` directly from the cloned directory in editable mode:

```bash
conda env create -f melafit-dev.yml
conda activate melafit
```

With an editable install, any changes to the source code in the cloned directory take effect immediately without reinstalling the package.

## Updating

### Standard update

Download the latest [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml) to `<YOUR-DIRECTORY>`. Navigate to it and run the update command as follows:

```bash
cd <YOUR-DIRECTORY>
conda env update -f melafit.yml --prune
```

This updates both the dependencies and `melafit` itself to the latest released version.

### Developer update

Navigate to the cloned repository directory and pull the latest version from the main branch:

```bash
cd <YOUR-DIRECTORY>/melafit
git pull
```

The editable install picks up the changes immediately. If dependencies in [`melafit-dev.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit-dev.yml) have changed, also run:

```bash
conda env update -f melafit-dev.yml --prune
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

## Scientific Foundations

If you use [melafit](https://github.com/vitaliy-ch25/melafit) in your research, please cite the following foundational publications:

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

If there is no associated publication on `melafit` yet, please cite the package directly using the following reference:

[Kolodyazhniy, V., & Cajochen, C. (2026). melafit: High-precision circadian melatonin profile analysis. Retrieved from https://github.com/vitaliy-ch25/melafit](https://github.com/vitaliy-ch25/melafit).

## Authors
* Vitaliy Kolodyazhniy – Lead Developer
* Christian Cajochen – Scientific Lead

## Revision History

### [v0.2.0]

### [v0.1.3](https://github.com/vitaliy-ch25/melafit/releases/tag/v0.1.3)
- Improved documentation
- Developer installation option

### [v0.1.2](https://github.com/vitaliy-ch25/melafit/releases/tag/v0.1.2)
- Improved documentation

### [v0.1.1](https://github.com/vitaliy-ch25/melafit/releases/tag/v0.1.1) - First PyPI release
- Enhanced function `fit()` to support custom waveform functions with user-defined initial parameters and bounds
- Changed named parameter order in `fit()`: `cost_f` and `cost_p` are now the last two parameters
- Fixed returned type hints in `func_defaults()`
- Additional unit tests for new functionality
- Improved README
- Package registered in Python Package Index PyPI

### [v0.1.0](https://github.com/vitaliy-ch25/melafit/releases/tag/v0.1.0) — First public release
- Dictionary support for waveform function parameters throughout the package: all functions accept both `dict` and `np.ndarray` for parameter input
- Named parameter constants: `BCF_PARAM_NAMES`, `SBCF_PARAM_NAMES`, `BBCF_PARAM_NAMES`, `BSBCF_PARAM_NAMES` and `PARAM_NAMES` lookup
- New utility functions `params_to_array()` and `array_to_params()` for conversion between array and named dictionary representations
- `fit()` now returns named parameter dictionary as `res.p` in addition to the standard scipy `res.x` array
- `fit()` now accepts `cost_p` dictionary for passing parameters to the cost function (e.g. `{"eps": 1e-6}`)
- New utility function `params_to_string()` for human-readable parameter output
- Fixed `area_cog()`: baseline subtraction and bin size normalization
- Unit tests for all public functions in `fitting`, `markers` and `utils`

### v0.0.9
- New function `func_defaults()` in `fitting.py` for standalone access to default initial conditions and constraints for all waveform functions
- Improved cost function: `eps` parameter for more robust fitting
- Optional `thresh_abs` parameter in `markers.midpoint()` for absolute threshold support
- New example script `example_dlmo.py` and dataset for DLMO detection from partial data
- Previous example renamed to `example_full_profile.py`
- Improved type hints, docstrings and README

### Initial revisions (v0.0.1 – v0.0.8)
- Full implementation of melatonin profile analysis as described in [Gabel et al. (2017)](https://doi.org/10.1038/s41598-017-07060-8)
- Waveform functions: `bcf`, `sbcf`, `bbcf`, `bsbcf`
- Markers: `amplitude`, `midpoint`, `DLMOn`, `DLMOff`, `area`, `cog`
- Utilities: `read_data`, `prepare_part_data`, `compute_wave`, `day_profile`, `abs_threshold`, `time_to_phase`, `phase_to_string`, `phase_diff`
- MIT license, packaging metadata and README

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/vitaliy-ch25/melafit/blob/main/LICENSE) file for details.