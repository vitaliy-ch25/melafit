# melafit

Python package for **high-precision 24h melatonin profile analysis.** Features a variety of baseline cosine functions for curve fitting (Van Someren & Nagtegaal, 2007) and a robust cost function for superior convergence, even with sparse data (Gabel et al., 2017).

## Overview

`melafit` is a Python package designed for high-precision modeling of 24-hour melatonin secretion. While standard cosinor or harmonic analysis fail to capture the physiological nuances of the melatonin "wave," `melafit` implements several **baseline cosine functions** including bimodal, skewed and biomdal-skewed modifications. This approach accounts for the characteristic baseline, asymmetry and dual peaks often seen in high-resolution melatonin data.

Furthermore, the library utilizes a **specialized cost function** developed to overcome common optimization hurdles (trivial all-zero solutions), ensuring stable convergence even when working with sparse or incomplete time series.

## Installation

### Using Conda or Miniconda (Recommended)

To ensure all dependencies (Python 3.12, NumPy, SciPy, Pandas, etc.) are correctly configured, you can create a dedicated Python virtual environment using the provided [`melafit.yml`](https://github.com/vitaliy-ch25/melafit/blob/main/melafit.yml) file:

```bash
# Create the environment from the yaml file
conda env create -f melafit.yml

# Activate the environment
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

A code example and some dummy data demonstrating melatonin profile curve fitting with this package are included in [./examples/example.py](https://github.com/vitaliy-ch25/melafit/blob/main/examples/example.py) and [./data/dummy_data.xlsx](https://github.com/vitaliy-ch25/melafit/blob/main/data/dummy_data.xlsx). Copy the sample script and data to your working directory and start from there. If you have performed the steps above as described, your script will 'see' all the required packages from any location. Simply make sure to use the virtual environment `melafit` you created.

## Data preparation
Follow the Excel table format and column naming conventions in [./data/dummy_data.xlsx](https://github.com/vitaliy-ch25/melafit/blob/main/data/dummy_data.xlsx):
* *Participant* for study participant ID
* *Date* for dates of the respective samples
* *Time* for sample timestamps 
* *Mel* for melatonin level values

## Key Features

* **Bimodal Waveform Fitting:** Implementation of the Nagtegaal & Van Someren model for superior physiological accuracy.
* **Optimized Convergence:** Leverages the robust cost function described in Gabel et al. (2017) to ensure reliable fits across diverse datasets.
* **Sparse Data Support:** Capable of reconstructing full profiles and estimating circadian phase from limited data points.
* **Research-Ready:** Direct derivation of phase markers from continuous, fitted waveforms.

## Scientific Foundations

If you use `melafit` in your research, please cite the following foundational publications:

### Human-Readable
1. **Van Someren, E. J., & Nagtegaal, E. (2007).** Improving melatonin circadian phase estimates. *Sleep Medicine*, 8(6), 590-601. [https://doi.org/10.1016/j.sleep.2007.03.012]
2. **Gabel, V., et al. (2017).** Differential impact in young and older individuals of blue-enriched white light on circadian physiology and alertness during sustained wakefulness. *Scientific Reports*, 7, 7620. [https://doi.org/10.1038/s41598-017-07060-8]

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

### v0.0.6
- New markers: melatonin area and centre of gravity (COG)
- New utility function `time_to_phase()`
- Improved docstrings and example script

### v0.0.5
- Additional waveform functions: `bcf`, `sbcf` and `bbcf`
- Improved example script and docstrings
- Revision history added to README

### v0.0.4
- MIT license added
- Improved and corrected type hints throughout
- Fixed bug in `example.py`: participant start time now correctly derived
  from participant's own data
- Added updating instructions to README

### v0.0.3
- Flexible framework for fitting of arbitrary waveform functions with
  customizable initial parameters and constraints
- New module `markers`: melatonin midpoint, DLMOn and DLMOff computation
- New module `utils`: `day_profile`, `phase_to_string`, `abs_threshold`,
  `read_data`, `prepare_part_data`, `compute_wave`
- Data preparation section added to README

### v0.0.2
- Improved input/output data format and column naming conventions
- Dependencies added to `pyproject.toml`
- Unified parameter vector `p` interface across all fitting routines

### v0.0.1
- Initial release: bimodal skewed baseline cosine function (`bsbcf`) and
  robust cost function (Gabel et al., 2017)
- Sample analysis script and dummy data included

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.