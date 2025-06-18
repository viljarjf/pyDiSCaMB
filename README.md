# pyDiSCaMB

![Tests](https://github.com/viljarjf/pyDiSCaMB/actions/workflows/test.yaml/badge.svg?event=push&branch=main)

Simple pybind11 wrapper to communicate with DiSCaMB from cctbx, built with scikit-build and cmake.

**LICENSE NOTE**

Recursive cloning installs the [MATTS databank](https://www.github.com/discamb-project/MATTS), which uses a license restricting commercial use.
Building the project with the databank will **include the databank in the built module**, which should be taken into account if the build is distributed.

# Installation

Tested on ubuntu, macOS, and windows, on python 3.9-3.12.
Currently (June 2025) not available as pre-built packages, manual installation instructions follow:

## Using Conda
A environment file is provided. You also need a C++ compiler, which is in a seperate file depending on your operating system.
Replace the brackets below with the appropriate choice (e.g. `compiler_macOS-latest.yml`).
```bash
git clone --recursive git@github.com:viljarjf/pyDiSCaMB.git
conda create -f pyDiSCaMB/conda/dev_env.yml
conda env update --file pyDiSCaMB/conda/compiler_[any of "macOS", "ubuntu", "windows"]-latest.yml
conda activate pydiscamb-dev
pip install pyDiSCaMB/
```

## Using Phenix
With [Phenix](https://phenix-online.org/), installation is even simpler:
```bash
git clone --recursive git@github.com:viljarjf/pyDiSCaMB.git
phenix.python -m pip install pyDiSCaMB/
```
Only tested on versions newer than around 2.0rc1-5500.

# Example usage

```python
from pydiscamb import DiscambWrapper

wrapper = DiscambWrapper.from_file("your_structure.cif")
wrapper.set_d_min(2.0)
f_calc = wrapper.f_calc()

# Now do what you want with the list of structure factors
```

# Development

Feel free to open pull requests. Currently under development by Viljar Femoen.

## Installation

Make a fork of [https://github.com/viljarjf/pyDiSCaMB](https://github.com/viljarjf/pyDiSCaMB) and use your new URL for cloning instead.
Otherwise the instructions are like before.

To debug, install in an editable state:
```bash
pip install -e pyDiSCaMB/
```
This sets the `Debug` flag in cmake, allowing you to use e.g. gdb.
Confirmed to work with both C++ and python breakpoints in WSL Ubuntu, using gdb and [Python C++ Debugger](https://marketplace.visualstudio.com/items/?itemName=benjamin-simmonds.pythoncpp-debug) for VS Code.

## Testing

Performed automatically on multiple OSs and Python version for all pull requests.
Can be done manually as follows:

```bash
pip install pytest
pytest pyDiSCaMB/
```

To avoid slow or very slow tests, instead run 
```bash
pytest -m "not veryslow" pyDiSCaMB/
```
change "veryslow" for "slow" to avoid even more tests.
