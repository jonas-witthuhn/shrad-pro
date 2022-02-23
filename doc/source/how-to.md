# How to ...
## ... prepare ancillary data.
For functionality a set of ancillary data (see {ref}`sec-intro-ancillary`) is needed in order to:
 
   * correct measured irradiance due to alignment of the instrument (ship movement)
   * estimate trace-gas absorption and Rayleigh-scattering optical depths in order to calculate the aerosol optical depth

To prepare the required ancillary data one can use `modules/ncutils.create_nc_dataset`.
This module requires the output filename, the Data in a dictionary with appropriate variable naming according to the `cfconv.json` configuration file.
In the shrad-pro root directory one can find `trosat_cfmeta_ins.json` and `trosat_cfmeta_met.json` as a blueprint for the INS and MET data, respectively.

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
   ncu.create_nc_dataset(fname, data, cfjson, attrs)
   ```

The output files should be named and stored according to the settings in `ConfigFile.ini`

