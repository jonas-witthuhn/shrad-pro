# How to ...

(sec-howto-ancillary)=
## ... prepare ancillary data.
For functionality a set of ancillary data (see {ref}`sec-intro-ancillary`) is needed in order to:
 
   * correct measured irradiance due to alignment of the instrument (ship movement)
   * estimate trace-gas absorption and Rayleigh-scattering optical depths in order to calculate the aerosol optical depth

To prepare the required ancillary data one can use `modules/ncutils.create_nc_dataset`.
This module requires the output filename, the Data in a dictionary with appropriate variable naming according to the `cfconv.json` configuration file.
In the shrad-pro root directory one can find `trosat_cfmeta_ins.json` and `trosat_cfmeta_met.json` as a blueprint for the INS and MET data, respectively.

### Required Variables
```python
# INS
variables = [
    "pitch", # degrees, positive if fore (bow) goes up
    "roll", # degrees, positive if portside goes up
    "yaw", # degrees, positive clockwise from North
    "lat", # degrees North
    "lon", # degrees East
    "time" # numpy datetime64
]
```

```python
# MET
variables = [
    "T", # Kelvin, air temperature
    "P", # Pa, air pressure
    "RH", # 0-1, relative humidity
    "lat", # degrees North
    "lon", # degrees East
    "time" # numpy datetime64
]
```

### Example usage:
   ```python
   import numpy as np
   import modules.ncutils as ncu
   # example for INS data
   data = {
        'roll': np.array(...),
        'pitch': np.array(...),
        'yaw': np.array(...),
        'lat': np.array(...),
        'lon': np.array(...),
        'time': np.array(..., dtype=np.datetime64),
          } 
   fname = "test.nc"
   cfjson = "<path>/shrad-pro/trosat_cfmeta_ins.json"
   attrs = {"note": "this note is added additionally as global attribute"}
   # store the data to shrad-pro compatible netCDF4 file
   ncu.create_cf_dataset(fname, data, cfjson, attrs)
   ```

The output files should be named and stored according to the settings in `ConfigFile.ini`

(sec-howto-processraw)=
## ... start processing raw data.
Raw data is processed using `python shrad.py process l1a`
* First things first, the raw data (uLogger .csv output data) should be stored in files <= 1 day. There can be multiple files for one day, collecting of daily data and merging will be done in 
   *shrad* 
(*modules.utils.load_rawdata_and_combine*), but multiple days in one file have not been implemented/tested yet.
* Check and modify `ConfigFile.ini`.
  * update Meta data for campaign and contacts
  * ensure correct paths
  * modify output filenames to your liking
  * 
* Ensure consistent file naming (As from uLogger output, the filenames will be already usable) 
  * All filenames should start with a prefix separated by an '_'. This will be used to identify the files, define filenames of shrad-output and will be stored in netCDF metadata. See also 
    ConfigFile.ini -> Meta -> pfx.
  * All filenames should include a date/time string indicating, when the file was created.
  ```tip
    shrad default settings require the date/time string in the form of YYMMDD_HHMM.
    This can be changed using --datetimepattern. See `shrad.py process l1a --help`
  ```
* Setup a correct and up to date calibration file. The default calibration file is data/GUVis_calibrations.json. Use this file as a blueprint, or update accordingly. The calibration file location 
  can be specified with `--calibration-file`.
* Setup ancillary data. For raw -> l1a, its not mandatory but recommended to provide the inertial-navigation-system (ins) data. See {ref}`sec-howto-ancillary`) for preparation of these datasets.
```tip
If no INS data is available and/or measurements where conducted on land (fixed location),
use --disable-ancillary-ins to run shrad without requesting ins data.
In addition, specify location coordinates with --coordinates, if BioGPS data is not available.
```

## .. process data level l1a to l1b.
Data which is already calibrated, sorted into daily netCDF files and enhanced with ancillary data such as sun position and platform movement angles is described as level **l1a**.
Level **l1a** data is directly processed from **raw** data (.csv output from uLogger software, see {ref}(`sec-howto-processraw`)).
Data which is, in addition, corrected for platform movement (only applies for *spectral_flux* and *broadband_flux*) is described as level **l1b**. 

Processing **l1a** to **l1b** data requires the calculation of the *apparent solar zenith angle*, which is the angle between the normal vector of GUVis and the sun position vector.
To calculate the *apparent solar zenith angle*, information about the platform angles and GUVis setup angles relative to the platform is needed. For definition of coordinate systems, angles and 
rotations, see {ref}(`sec-def-coords`).

Use `python shrad.py process l1b` to process **l1a** data to level **l1b** (for measurements on the ship):
1. (optional) If this step is skipped, this script will be triggered in the normal processing, but as it takes a while, it is recommended to do this separately.
  Find out offset roll and pitch angles of GUVis to the ships INS by using `python shrad.py utils dangle [l1a-filenames]`.
  This script compares the GUVis measured roll and pitch angles to the 
   ship angles.
  The GUVis roll and pitch angles are transformed to the **heading-aligned-coordinate-system** by a minimization method estimating GUVis yaw angle relative to the ship.
  If the relative yaw angle is known, one can give this script a first guess with the `--dyaw` option.
  Be careful of angle definitions: {ref}(`sec-def-coords`).
  As a result, the mean offset roll and pitch angles in **heading-aligned-coordinate-system** are printed, save them for the next step.

2. 

3. 




