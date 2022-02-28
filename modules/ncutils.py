import numpy as np
import pandas as pd

from trosat import cfconv as cf


def create_cf_dataset(fname, data, cfjson, attrs={}):
    """
    Create cf-compliant netCDF file using trosat.cfconv
    
    Parameters
    ----------
    fname: string
        Output netCDF filename
    data: dict or xarray.Dataset
        Data to store. Variable names have to match definitions in the cfjson file.
    cfjson: string
        Filename of the json configuration file required for trosat.cfconv
    attrs: dict
        Additional global attributes for output file. This will update the
        cfjson["attributes"] dict.
    Returns
    -------
    """
    # read cfdict
    cfdict = cf.read_cfjson(cfjson)

    # update global attributes
    cfdict["attributes"].update(attrs)

    # set appropriate time variable units to "<time> since <data-start-time>"
    timedata = np.array(data["time"]).astype("datetime64[ms]")
    starttime = pd.to_datetime(timedata[0])
    timeunits = f"milliseconds since {starttime:%Y-%m-%d %H:%M:%S}"
    cfdict["variables"]["time"]["attributes"]["units"] = timeunits

    # the scale_factor attributes of variables in the cfjson file define the
    # precision of the stored data. E.g. variables with a scale_factor of 0.1
    # will be accurate to one decimal. The following loop adds the "add_offset"
    # attribute to offset the values to be always positive.
    for var in cfdict["variables"].keys():
        if var == "time":
            continue
        scalefactor = cfdict["variables"][var]["attributes"]["scale_factor"]
        # maximum precision of stored integer values
        prec = np.int(np.ceil(np.abs(np.log10(scalefactor))))
        # calculate and set offset based on precision and values
        addoffset = np.round(np.min(data[var]), prec)
        cfdict["variables"][var]["attributes"]["add_offset"] = addoffset

    # create the file
    f = cf.create_file(fname, cfdict=cfdict)
    f.set_auto_scale(True)

    # fill the data
    # apply offset and convert time manually to milliseconds from start-time
    convtime = (timedata-timedata[0]).astype(int)
    f["time"][:] = convtime

    for var in cfdict["variables"].keys():
        if var == "time":
            continue
        f[var][:] = np.array(data[var])

    # close and save the file
    f.close()
    return 0
