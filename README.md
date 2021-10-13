# shrad-pro
Post-processing algorithm for the GUVis-3511 shadow-band radiometer an a ship platform.
  * Calibration
  * cosine response error correction
  * misalignment correction
  * identification of spectral irradiance components (DNI, GHI, DHI)
  * calculation of spectral aerosol optical depth

## Requirements:
This code runs on python 3.x with a reasonable up to date versions of numpy, scipy, xarray, and pandas.
Also the [trosat](https://github.com/hdeneke/trosat-base) package is used to calculate the sun position:
```
pip install git+https://github.com/hdeneke/trosat-base.git#egg=trosat-base
```

## References:
Witthuhn, J., Deneke, H., Macke, A., and Bernhard, G.: Algorithms and uncertainties for the determination of multispectral irradiance components and aerosol optical depth from a shipborne rotating shadowband radiometer, Atmos. Meas. Tech., 10, 709–730, https://doi.org/10.5194/amt-10-709-2017, 2017

## Todo:
  * l1b -> l1c
  * l1c -> l2aod

## Usage:
Clone this repository, edit ConfigFile.ini, and run

```
python ./shrad.py --help
```
```
usage: shrad.py [-h] {utils,process} ...

optional arguments:
  -h, --help       show this help message and exit

ShRad jobs:
  {utils,process}  Choose:
                       *utils: collection of utility scripts
                       *process: advance input GUVis files in processing level

```
---
```
usage: shrad.py utils [-h] {dangle} ...

Run utility scripts.

optional arguments:
  -h, --help  show this help message and exit

utility scripts:
  {dangle}    Choose:
                  dangle: Calculate offset angles between
                          INS platform and GUV instrument
```
---
```
usage: shrad.py process [-h] {l1a,l1b,l1c,l2aod} ...

Advance input files in processing level.

optional arguments:
  -h, --help           show this help message and exit

Output data level:
  {l1a,l1b,l1c,l2aod}  Choose:
                           l1a: calibrated GUVis data
                           l1b: calibrated and corrected GUVis data
                           l1c: processed irradiance components
                           l2_aod: spectral aerosol optical depth

```
