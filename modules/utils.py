import configparser
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
import trosat.sunpos as sp
import xarray as xr
from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints
from modules.helpers import print_warning as printw
from scipy.interpolate import griddata

CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
CONFIG.read("ConfigFile.ini")


def get_pfx_time_from_input(pattern,
                            input_files,
                            VERBOSE=True, DEBUG=False, lvl=0):
    """Parsing Date and Time according to --datetimepattern from raw GUVis files.
    In addition, the file prefix is identified for later use for for all dataset files and to identify ancillary data.
    """
    datetime_parser = re.compile(pattern)
    # identify input file dates
    input_dates = np.array([], dtype=np.datetime64)
    pfxs = []
    for input_file in input_files:
        _, filename = os.path.split(input_file)
        pfx = filename.split('_')[0]
        m = datetime_parser.match(filename)
        if DEBUG:
            printd(f"Parse: {input_file}")
            printd(f"    -- Matches: {m.groupdict()}")
        m = m.groupdict()
        for key in ['year', 'month', 'day']:
            if key not in m.keys():
                raise ValueError(f"Cannot Identify Date with given option --datetimepattern: {pattern}."
                                 " Missing year, month or day searchgroup?")
        # string to int conversion
        if len(m['year']) == 2:
            m.update({'year': 2000 + int(m['year'])})
        m = dict([a, int(x)] for a, x in m.items())
        # convert to numpy datetime64
        input_datetime = np.datetime64(dt.datetime(**m))
        input_dates = np.append(input_dates, input_datetime)
        pfxs.append(pfx)
    pfx = np.unique(pfxs)
    if len(pfx) != 1:
        raise ValueError("Input files should always have the same prefix")
    pfx = str(pfx[0])
    return pfx, input_dates


def load_rawdata_and_combine(files,
                             calib_file="",
                             verbose=True, debug=False, lvl=0):
    """ Reading raw GUVis files and combine to one dataframe
    """

    if verbose:
        prints(f"Load raw data from {len(files)} file(s)...",
               lvl=lvl)
    if debug:
        printd(f"Files for to process:")
        printd(str(files))

    complete_df = []
    for i, fname in enumerate(files):
        fsize = os.path.getsize(fname) / 1024 ** 2
        if fsize < 0.1:
            printw(str(f'Skip file {fname} of size {fsize:.2f}MB.' +
                       ' ... less or no data.'))
            continue
        if verbose:
            prints(f"Read file {i + 1}/{len(files)} with {fsize:.2f}MB.",
                   lvl=lvl + 1)

        df = pd.read_csv(fname, sep=',', encoding=None)
        if type(complete_df) == list:
            complete_df = df.copy()
        else:
            complete_df = complete_df.append(df, ignore_index=True)

    if type(complete_df) == list:
        # no files, or all files are empty
        if verbose:
            prints(str(f"No data - skipping.."), lvl=lvl)
        return False

    # homogenize  dataframe
    complete_df.drop_duplicates(subset='DateTimeUTCISO', keep='first', inplace=True)
    complete_df.reset_index(drop=True, inplace=True)
    # remove the unit appendix in standard raw csv data of GUVis        
    keys = {}
    units = []
    variables = []
    for k in complete_df.keys():
        ksplit = k.split(' ', 1)
        ksplit += [''] * (2 - len(ksplit))
        units.append(ksplit[1])
        variables.append(ksplit[0])
        keys.update({k: ksplit[0]})
    complete_df = complete_df.rename(keys, axis='columns')

    # to xarray dataset
    ds = xr.Dataset.from_dataframe(complete_df)
    # parse datetime for date objects
    for key in ds.keys():
        if key[:8] == 'DateTime':
            datetime = pd.to_datetime(ds[key].values,
                                      infer_datetime_format=True)
            ds[key].values = datetime

    ds = ds.rename_vars({'DateTimeUTCISO': 'time'})
    ds = ds.swap_dims({'index': 'time'})
    ds = ds.reset_coords(names=['index'], drop=True)

    # Bug correction for uLogger version < 1.0.24
    if ("BioGpsTime" in list(ds.keys())) and (ds.time.values[0] < np.datetime64("2016-04-01")):
        lat = ds.BioGpsLatitude.values
        ni = lat < 0
        lat[ni] = np.floor(lat[ni]) + 1. - (lat[ni] - np.floor(lat[ni]))
        ds.BioGpsLatitude.values = lat

    # detect uncalibrated files
    # index of first radiation data
    idx = [i for i, var in enumerate(variables) if re.match("Es\d", var)][0]
    radunit = units[idx]

    if calib_file:
        if verbose:
            prints("Calibrate Radiation data ...", lvl=lvl + 1)
        # get the calibration
        calib_ds = get_calibration_factor(date=ds.time.values[0],
                                          file=calib_file)
        if not re.match(".*V", radunit):
            if debug:
                printd(" Files are precalibrated with from the GUVis internal storage, now uncalibrate...")
            # data is already calibrated with stored calibration,
            # thus, remove stored calibration befor calibration
            # with drift corrected calibration factor
            for ch in calib_ds.channel.values:
                if ch == 0:
                    # broadband channel calibration is never stored
                    continue
                cds = calib_ds.sel(channel=ch)
                cs = cds.calibration_factor_stored
                ds[f"Es{ch}"] = ds[f"Es{ch}"] * cs

                # apply calibration
        if debug:
            printd("Now Calibrate the radiation data ...")

        for ch in calib_ds.channel.values:
            cds = calib_ds.sel(channel=ch)
            ca = cds.calibration_factor
            ds[f"Es{ch}"] = ds[f"Es{ch}"] / ca
        if verbose:
            prints("... done", lvl=lvl + 1)
    else:
        if debug:
            printd("No --calibration-file is specified. Assuming precalibrated files.")
        # no Calibratiofile is given,
        # assuming precalibrated files...
        if re.match(".*V", radunit):
            # Files are not calibrated
            raise ValueError("Files are not calibrated, at least the ASCII Header tells so."
                             " Please specify a --calibration-file.")

    ####################################################################################
    # Make nice dataset with attributes
    ###################################
    # First combine all radiation channels

    if verbose:
        prints("... done", lvl=lvl)
    return ds


def get_calibration_factor(date, file):
    """
    Retrieve the corrected calibration factor for GUVis from 
    GUVis_calibrations.json
    
    Parameters
    ----------
    date: numpy.datetime64
        Day of the data
    file: str
        Path to the calibration file (.json)
    
    Returns
    -------
    calib_ds: xarray.Dataset
        variables: 
            centroid_wvl: nm
                the centre wavelength of the spectral response
            calibration_factor: V / (uW cm-2 nm-1)
                drift corrected calibration factor
            calibration_factor_stored: V / (uW cm-2 nm-1)
                calibration factor stored by Biospherical calibration procedure in the instrument storage
            signal_noise_ratio: 
                Signal/Noise ration retrieved from the Biospherical calibration certificate.
        coords:
            channel: nm
                Name of the spectral channel of the GUVis    
    """
    date = date.astype('datetime64[D]').astype(int)
    with open(file, 'r') as f:
        calibrations = json.load(f)

    channel = calibrations['_CHANNEL']
    cwvl = calibrations['_CENTROID_WVL']
    cdates = list(calibrations['calibration'].keys())
    cdates = np.array(cdates, dtype='datetime64[D]').astype(int)
    values = []
    snrs = []
    stored = []
    for c in calibrations['calibration'].keys():
        val = np.array(calibrations['calibration'][c]['calibF'])
        val[val is None] = np.nan
        snr = np.array(calibrations['calibration'][c]['SNR'])
        snr[snr is None] = np.nan
        stored.append(calibrations['calibration'][c]['stored'])
        if len(values) == 0:
            values = np.array(val)
            snrs = np.array(snr)
        else:
            values = np.vstack((values, val))
            snrs = np.vstack((snrs, snr))
    stored = np.array(stored, dtype=bool)

    si = np.argsort(cdates)
    cdates = cdates[si]
    values = np.array(values[si, :], dtype=np.float)
    snrs = np.array(snrs[si, :], dtype=np.float)
    stored = stored[si]

    # fill nan values with interpolatet values
    for i in range(len(values[0, :])):
        mask = np.isnan(values[:, i])
        values[mask, i] = np.interp(np.flatnonzero(mask),
                                    np.flatnonzero(~mask),
                                    values[~mask, i])
        mask = np.isnan(snrs[:, i])
        snrs[mask, i] = np.interp(np.flatnonzero(mask),
                                  np.flatnonzero(~mask),
                                  snrs[~mask, i])

    # interpolation linear between the closest two calibrations
    # to correct calibration assuming a linear drift
    ca = griddata(cdates, values, date, method='linear')
    snr = griddata(cdates, snrs, date, method='linear')
    if np.all(np.isnan(ca)):
        ca = griddata(cdates, values, date, method='nearest')
        snr = griddata(cdates, snrs, date, method='nearest')

    # stored calibration in GUVis uLogger
    cs = values[stored, :][np.searchsorted(cdates[stored], date) - 1, :]

    calib_ds = xr.Dataset({'centroid_wvl': ('channel', cwvl),
                           'calibration_factor': ('channel', ca),
                           'calibration_factor_stored': ('channel', cs),
                           'signal_noise_ratio': ('channel', snr)},
                          coords={'channel': ('channel', channel)})

    return calib_ds


def add_ins_data(ds,
                 verbose=True,
                 debug=False,
                 lvl=0):
    day = pd.to_datetime(ds.time.values[0].astype("datetime64[D]"))
    fname = CONFIG['FNAMES']['ancillary'].format(pfx=ds.pfx,
                                                 AncillaryType='INS',
                                                 date=day)
    if debug:
        printd(f"INS file: {fname}")
    if verbose:
        prints("Assigning INS data ...", lvl=lvl)
    if not os.path.exists(fname):
        printw("There is no INS data available for this day. I try skipping this day for now. "
               "Consider to disable the usage of ancillary INS data with --disable-ancillary-ins, "
               "or provide data for this day. Continuing...")
        return False
    ds_ins = xr.open_dataset(fname)

    # check variables
    required_vars = ['pitch', 'roll', 'yaw', 'lat', 'lon']
    available_vars = [var for var in ds_ins.keys()]
    if not all(var in available_vars for var in required_vars):
        raise ValueError(f"The INS data given did not have all required variables {required_vars}!")

    # interpolate to ds
    ds_ins = ds_ins.interp_like(ds,
                                method='linear',
                                kwargs=dict(bounds_error=False,
                                            fill_value=np.nan))

    # Assign to ds
    ds = ds.assign({'InsLatitude': ('time', ds_ins['lat'].data),
                    'InsLongitude': ('time', ds_ins['lon'].data),
                    'InsPitch': ('time', ds_ins['pitch'].data),
                    'InsRoll': ('time', ds_ins['roll'].data),
                    'InsYaw': ('time', ds_ins['yaw'].data)})

    if 'Instrument' in ds_ins.attrs.keys():
        insinstrument = ds_ins.Instrument
    else:
        if debug:
            printd("INS data has no global attribute 'Instrument', description will be set to 'n/a'")
        insinstrument = 'n/a'

    for key in ['InsLatitude',
                'InsLongitude',
                'InsPitch',
                'InsRoll',
                'InsYaw']:
        if key in CONFIG['NC Variables Map'].keys():
            cfgkey = CONFIG['NC Variables Map'][key]
        else:
            cfgkey = key
        ds[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][cfgkey],
                              'units': CONFIG['CF Units'][cfgkey],
                              'notes': f'Derived from ancillary INS data {fname} of instrument: {insinstrument}'})

    if verbose:
        prints("... done", lvl=lvl)

    return ds


def add_met_data(ds,
                 verbose=True,
                 debug=False,
                 lvl=0):
    day = pd.to_datetime(ds.time.values[0].astype("datetime64[D]"))
    fname = CONFIG['FNAMES']['ancillary'].format(pfx=ds.pfx,
                                                 AncillaryType='MET',
                                                 date=day)
    if debug:
        printd(f"MET file: {fname}")
    if verbose:
        prints("Merging MET data ...", lvl=lvl)
    if not os.path.exists(fname):
        printw("There is no MET data available for this day. I try skipping this day for now."
               " Consider to disable the usage of ancillary MET data with --disable-ancillary-met,"
               " or provide data for this day. Continuing...")
        return False
    ds_met = xr.open_dataset(fname)

    # check variables
    required_vars = ['T', 'P', 'RH']
    available_vars = [var for var in ds_met.keys()]
    if not all(var in available_vars for var in required_vars):
        raise ValueError(f"The MET data given did not have all required variables {required_vars}!")

    # interpolate to ds
    ds_met = ds_met.interp_like(ds,
                                method='linear',
                                kwargs=dict(bounds_error=False,
                                            fill_value=np.nan))

    # Assign to ds
    ds = ds.assign({'T': ('time', ds_met['T'].data),
                    'RH': ('time', ds_met['RH'].data),
                    'P': ('time', ds_met['P'].data)})

    if 'Instrument' in ds_met.attrs.keys():
        metinstrument = ds_met.Instrument
    else:
        if debug:
            printd("MET data has no global attribute 'Instrument', description will be set to 'n/a'")
        metinstrument = 'n/a'
    for key in ['T', 'P', 'RH']:
        if key in CONFIG['NC Variables Map'].keys():
            cfgkey = CONFIG['NC Variables Map'][key]
        else:
            cfgkey = key
        ds[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][cfgkey],
                              'units': CONFIG['CF Units'][cfgkey],
                              'notes': f'Derived from ancillary MET data {fname} of instrument: {metinstrument}'})
    if verbose:
        prints("... done", lvl=lvl)

    return ds


def add_sun_position(ds,
                     coords=None,
                     verbose=True,
                     debug=False,
                     lvl=0):
    # get latitude/longitude from preferred sources
    # Sources: BioGPS > Ships INS > provided stationary coords
    if coords is None:
        coords = [False, False]
    latkeys = ['BioGpsLatitude', 'InsLatitude']
    lonkeys = ['BioGpsLongitude', 'InsLongitude']
    # look for latitude information
    lat = False
    for key in latkeys:
        if key in ds.keys():
            lat = ds[key].values
            latkey = key
            break
    if type(lat) == bool and not lat:
        lat = coords[0]
    if type(lat) == bool and not lat:
        raise ValueError("No positional information (latitude) is found! E.g., no information from BioGPS,"
                         " INS data and --coordinates is not set (for stationary observations)")

    lon = False
    for key in lonkeys:
        if key in ds.keys():
            lon = ds[key].values
            lonkey = key
            break
    if type(lon) == bool and not lon:
        lon = coords[1]
    if type(lon) == bool and not lon:
        raise ValueError("No positional information (longitude) is found! E.g., no information from BioGPS, "
                         " INS data and --coordinates is not set (for stationary observations)")

    szen, sazi = sp.sun_angles(time=ds.time.values,
                               lat=lat,
                               lon=lon,
                               units=sp.DEG)

    ds = ds.assign({'SolarZenithAngle': ('time', szen),
                    'SolarAzimuthAngle': ('time', sazi)})
    for key in ['SolarZenithAngle', 'SolarAzimuthAngle']:
        if key in CONFIG['NC Variables Map'].keys():
            cfgkey = CONFIG['NC Variables Map'][key]
        else:
            cfgkey = key
        ds[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][cfgkey],
                              'units': CONFIG['CF Units'][cfgkey],
                              'notes': (f"Calculated using trosat.sunpos with data of the variables:"
                                        " {latkey}, {lonkey}, and time")})

    esd = sp.earth_sun_distance(time=ds.time.values[0])
    ds = ds.assign({'EarthSunDistance': ('', [np.mean(esd)])})
    ds['EarthSunDistance'].attrs.update({'long_name': 'Distance from Earth centre to the sun',
                                         'standard_name': "distance_from_sun",
                                         'units': 'AU'})
    return ds


def correct_uv_cosine_response(ds,
                               channels,
                               file,
                               VERBOSE=True,
                               DEBUG=False,
                               lvl=0):
    """ Based on the diffuser material and inlet optic, the UV channels, e.g., Es305 and Es313
    need additional cosine response correction. The correction factor is provided by Biospherical Inc.
    based on solar zenith angle.
    """
    # check if file is there:
    if not os.path.exists(file):
        raise ValueError(f"Cosine Correction of UVchannels is switched on, but can't find the file specified by"
                         " --uvcosine-correction-file: {File}.")

    if ds.time.values[0] < np.datetime64("2016-02-29"):
        printw("For TROPOS GUVis-3511 SN:000350, the UVchannel cosine response correction is required only after"
               " Diffuser replacement on 2016-02-29. Consider switch of by setting --uvcosine-correction-disable")

    if VERBOSE:
        prints("Apply UV channel cosine response correction ...", lvl=lvl)
    corr_ds = pd.read_csv(file, sep=',')
    channels = np.unique(channels)
    for chan in channels:
        if DEBUG:
            printd(f"Processing Es{chan} / Es{chan}_corr.")
        # delete data which is corrected by the uLogger software,
        # as this correction is applied to global irradiance only,
        # e.g., when the BioSHADE is in Z or P position.
        if f"Es{chan}_corr" in ds.keys():
            ds = ds.drop_vars(f"Es{chan}_corr")
        if f"Es{chan}" in ds.keys():
            c = griddata(corr_ds['SZA'],
                         corr_ds[f'Es{chan}'],
                         ds.SensorZenithAngle.values)
            ds[f'Es{chan}'] = ds[f'Es{chan}'] / c
    if VERBOSE:
        prints("... done", lvl=lvl)
    return ds
