import configparser
import json
import os
import re
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

import trosat.sunpos as sp

import modules.shcalc as shcalc
from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints
from modules.helpers import print_warning as printw

CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
CONFIG.read("ConfigFile.ini")


def get_pfx_time_from_raw_input(pattern,
                                input_files,
                                verbose=True, debug=False, lvl=0):
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
        if debug:
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

        df = pd.read_csv(fname, sep=',', encoding="ISO-8859-1")
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
                printd(" Files are pre-calibrated with from the GUVis internal storage, now un-calibrate...")
            # data is already calibrated with stored calibration,
            # thus, remove stored calibration before calibration
            # with drift corrected calibration factor
            for ch in calib_ds.channel.values:
                if int(ch) == 0:
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
            printd("No --calibration-file is specified. Assuming pre-calibrated files.")
        # no calibration file is given,
        # assuming precalibrated files...
        if re.match(".*V", radunit):
            # Files are not calibrated
            raise ValueError("Files are not calibrated, at least the ASCII Header tells so."
                             " Please specify a --calibration-file.")

    # Make nice dataset with attributes
    dsa = xr.Dataset()
    dsa = dsa.assign_coords({'time': ('time', ds.time.data)})

    channels = calib_ds.channel.data
    channel_idx = np.where(channels != 0)  # skipping broadband for now
    # add channels as dimension
    key = 'wavelength'
    dsa = dsa.assign_coords({'wavelength': ('ch', channels[channel_idx])})
    dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                           'units': CONFIG['CF Units'][key]})

    # add specified centroid wavelength of each spectral channel
    key = 'centroid_wavelength'
    dsa = dsa.assign({key: ('ch', calib_ds.centroid_wvl.data[channel_idx])})
    dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                           'units': CONFIG['CF Units'][key]})

    # add measurements of spectral flux of all spectral channels
    for i, ch in enumerate(dsa.wavelength.values):
        if i == 0:
            rad = ds[f'Es{ch}'].data
        else:
            rad = np.vstack((rad, ds[f'Es{ch}'].data))
    rad = rad.T
    # convert units: uW cm-2 nm-1 -> W m-2 nm
    rad *= 1e-2

    key = 'spectral_flux'
    dsa = dsa.assign({key: (('time', 'ch'), rad)})
    dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                           'units': CONFIG['CF Units'][key]})

    # if available, add the broadband measurements too
    if 'Es0' in ds.keys():
        key = 'broadband_flux'
        bb_flux = ds['Es0'].data
        bb_flux *= 1e-2 # uW cm-2 -> W m-2
        dsa = dsa.assign({key: ('time', bb_flux)})
        dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][key],
                               'units': CONFIG['CF Units'][key],
                               'notes': 'Measured with the GUVis Radiometer'})

    for key in ['EsRoll',
                'EsPitch',
                'BioShadeAngle',
                'BioShadeMode',
                'BioGpsLongitude',
                'BioGpsLatitude',
                'EsTemp',
                'SolarAzimuthAngle',
                'SolarZenithAngle']:
        if key in ds.keys():
            dsa = dsa.assign({key: ('time', ds[key].data)})
            if key in CONFIG['NC Variables Map'].keys():
                cfgkey = CONFIG['NC Variables Map'][key]
            else:
                cfgkey = key
            dsa[key].attrs.update({'standard_name': CONFIG['CF Standard Names'][cfgkey],
                                   'units': CONFIG['CF Units'][cfgkey],
                                   'notes': f'Obtained from GUVis raw data'})

    if verbose:
        prints("... done", lvl=lvl)
    return dsa


def store_nc(ds, output_filename, overwrite=False,
             verbose=True, debug=False, lvl=0):
    """
    Stores xarray Dataset to given path, and adds zlib encoding to all variables

    Parameters
    ----------
    ds: xarray.Dataset
    overwrite: bool
        overwrite output files if True, the default is False
    verbose: bool
        enable verbose mode, the default is True
    debug: bool
        enable debug (more verbose) mode, the default is False
    lvl: int
        intend level of verbose messages, the default is 0
    output_filename: str
        path of resulting output file
    """
    if debug:
        printd(f"Add 'zlib' encoding to variables: {[key for key in ds.keys()][:]}")
    encoding = {}
    for key in ds.keys():
        encoding.update({key: {'zlib': True}})
    if verbose:
        prints(f"Storing to {output_filename} ...", lvl=lvl)
    os.makedirs(os.path.dirname(output_filename),
                exist_ok=True)
    if os.path.exists(output_filename):
        if overwrite:
            os.remove(output_filename)
        else:
            raise ValueError("Output file already exists and --overwrite was not set.")
    ds.to_netcdf(output_filename,
                 encoding=encoding)
    if verbose:
        prints("... done", lvl=lvl)
    return 0


def add_nc_global_attrs(ds, system_meta):
    """
    Reads and stores global attributes from ConfigFile.ini
    Parameters
    ----------
    ds: xarray.Dataset
    system_meta: dict
        Dict storing all variables necessary for string formatting of attributes from the config file

    Returns
    -------
    ds: xarray.Dataset
        The same Dataset as input, but with updated global variables
    """
    for key in CONFIG['META'].keys():
        ds = ds.assign_attrs({key: CONFIG['META'][key].format(**system_meta)})
    return ds


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
    # remove nan values
    ds_ins = ds_ins.dropna('time')

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
    # remove nan values
    ds_met = ds_met.dropna('time')
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
    """
    Adds the sun position calculated based on latitude, longitude and time of the input dataset with trosat.sunpos
    (https://github.com/hdeneke/trosat-base).

    Parameters
    ----------
    ds: xarray.Dataset
        The input Dataset have to contain 'time' as a Variable, storing numpy.datetime64 data. Optionally, latitude and
        longitude will be acquired from 'BioGpsLatitude', 'InsLatitude', 'BioGpsLongitude', or 'InsLongitude'.
    coords: (float, float) or None
        Fallback (Latitude, Longitude) in degrees North and degrees East, respectively, if ds has no latitude or
        longitude information. The default is None.
    verbose: bool
        Enables verbose output. The default is True.
    debug: bool
        Enables debug messages. The default is False.
    lvl: int
        Sets the intention level of verbose messages. the default is 0.

    Returns
    -------
    ds: xarray.Dataset
        Same as the input ds, but added the variables 'SolarZenithAngle', 'SolarAzimuthAngle', and 'SunEarthDistance'.
    """
    if verbose:
        prints("Calculate and assign sun position data ... ", lvl=lvl)
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
        lat = np.array([float(coords[0])])
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
        lon = np.array([float(coords[1])])
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
                                        f" {latkey}, {lonkey}, and time")})

    esd = sp.earth_sun_distance(time=ds.time.values[0])
    ds = ds.assign({'EarthSunDistance': ('scalar', [np.mean(esd)])})
    ds['EarthSunDistance'].attrs.update({'long_name': 'Distance from Earth centre to the sun',
                                         'standard_name': "distance_from_sun",
                                         'units': 'AU'})
    if verbose:
        prints("... done", lvl=lvl)
    return ds


def add_offset_angles(ds, drdp, dy):
    """add offset angles to the dataset
    """
    delta_roll, delta_pitch = drdp
    yaw_guvis = dy

    # apply to dataset
    ds = ds.assign({'OffsetRoll': ('scalar', [delta_roll]),
                    'OffsetPitch': ('scalar', [delta_pitch])})
    ds = ds.assign(EsYaw=lambda ds: ds.InsYaw + yaw_guvis)

    ds.OffsetRoll.attrs.update({'long_name': 'Offset_platform_to_guvis_roll_angle_starboard_down',
                                'standard_name': 'platform_roll_starboard_down',
                                'units': 'degrees'})
    ds.OffsetPitch.attrs.update({'long_name': 'Offset_platform_to_guvis_pitch_angle_fore_up',
                                 'standard_name': 'platform_pitch_fore_up',
                                 'units': 'degrees'})
    ds.EsYaw.attrs.update({'long_name': 'guvis_yaw_clockwise_from_north',
                           'standard_name': 'platform_yaw_north_east',
                           'units': 'degrees'})
    return ds


def add_apparent_zenith_angle(ds, verbose=True, debug=False, lvl=0):
    if verbose:
        prints("Calculate and assign sun position data ... ", lvl=lvl)
    # prefer Ins data
    if "InsRoll" in ds.keys():
        if debug:
            printd("Using INS data")
        rpy = np.vstack((ds.InsRoll.data,
                         ds.InsPitch.data,
                         ds.InsYaw.data)).T
        drdpdy = np.vstack((ds.OffsetRoll.data,
                            ds.OffsetPitch.data,
                            np.zeros(ds.OffsetRoll.data.shape))).T
    else:
        if debug:
            printd("Fallback to GUVis accelerometer")
        rpy = np.vstack((ds.EsRoll.data,
                         ds.EsPitch.data,
                         ds.EsYaw.data)).T
        drdpdy = np.zeros(rpy.shape)

    sun_angles = np.vstack((ds.SolarZenithAngle.data,
                            ds.SolarAzimuthAngle.data)).T

    apparent_zen = shcalc.calc_apparent_szen(rpy, sun_angles, drdpdy)

    ds = ds.assign({'ApparentSolarZenithAngle': ('time', apparent_zen)})
    ds.OffsetRoll.attrs.update({'long_name': 'apparent_solar_zenith_angle_from_sensor_normal',
                                'standard_name': CONFIG['CF Standard Names']['sensor_zenith_angle'],
                                'units': CONFIG['CF Units']['sensor_zenith_angle']})
    if verbose:
        prints("... done", lvl=lvl)
    return ds


def correct_uv_cosine_response(ds,
                               channels,
                               correction_file,
                               verbose=True,
                               debug=False,
                               lvl=0):
    """ Based on the diffuser material and inlet optic, the UV channels, e.g., Es305 and Es313
    need additional cosine response correction. The correction factor is provided by Biospherical Inc.
    based on solar zenith angle.
    """
    # check if file is there:
    if not os.path.exists(correction_file):
        raise ValueError(f"Cosine Correction of UV-channels is switched on, but can't find the file specified by"
                         f" --uvcosine-correction-file: {correction_file}.")

    if ds.time.values[0] < np.datetime64("2016-02-29"):
        printw("For TROPOS GUVis-3511 SN:000350, the UVchannel cosine response correction is required only after"
               " Diffuser replacement on 2016-02-29. Consider switch of by setting --uvcosine-correction-disable")

    if verbose:
        prints("Apply UV channel cosine response correction ...", lvl=lvl)
    corr_ds = pd.read_csv(correction_file, sep=',')
    channels = np.unique(channels)
    for chan in channels:
        if int(chan) in ds.wavelength.data:
            idx = int(np.where(int(chan) == ds.wavelength.data)[0])
            if debug:
                printd(f"Processing Es{chan}.")
            c = griddata(corr_ds['SZA'],
                         corr_ds[f'Es{chan}'],
                         ds.ApparentSolarZenithAngle.values)
            C = np.ones(ds.spectral_flux.data.shape)
            C[:, idx] = 1. / c
            ds.spectral_flux.values = ds.spectral_flux.values * C
    if verbose:
        prints("... done", lvl=lvl)
    return ds


def correct_cosine_and_motion(ds,
                              cosine_error_file="data/AngularResponse_GUV350_140129.csv",
                              misalignment_file="data/motioncorrection/C3lookup_{channel}.nc",
                              verbose=True,
                              debug=False,
                              lvl=0):
    if verbose:
        prints("Apply cosine error and misalignment correction ...", lvl=lvl)
    wvls = ds.wavelength.data

    # get cosine response correction factors
    angdat = np.loadtxt(cosine_error_file, delimiter=',')
    angwvl = angdat[0, 1:]
    angzen = angdat[1:, 0]
    angcors = angdat[1:, 1:]
    # bb- broadband, sp- spectral
    angcor_bb = np.mean(angcors[:, :], axis=1)
    angcor_sp = np.zeros((len(angzen), len(wvls)))
    for i in range(len(wvls)):
        angcor_sp[:, i] = angcors[:, np.argmin(np.abs(angwvl - wvls[i]))]

    # apply cosine error corrections
    cfactor = griddata(angzen, angcor_sp, ds.ApparentSolarZenithAngle.data)
    ds.spectral_flux.values = ds.spectral_flux.values / cfactor
    cfactor = griddata(angzen, angcor_bb, ds.ApparentSolarZenithAngle.data)
    ds.broadband_flux.values = ds.broadband_flux.values / cfactor

    # correct tilt (spectral)
    ks = []
    for i, wvl in enumerate(wvls):
        misalignment_ds = xr.open_dataset(misalignment_file.format(channel=wvl))
        x, y = np.meshgrid(misalignment_ds.szen.data, misalignment_ds.apparent_szen.data)
        k = griddata((x.flatten(), y.flatten()),
                     misalignment_ds.k.data.flatten(),
                     (ds.SolarZenithAngle.data, ds.ApparentSolarZenithAngle.data))
        if i == 0:
            ks = k[:, np.newaxis]
        else:
            ks = np.hstack((ks, k[:, np.newaxis]))
    ds.spectral_flux.values = ds.spectral_flux.values * ks

    # correct tilt (bb)
    misalignment_ds = xr.open_dataset(misalignment_file.format(channel=0))
    x, y = np.meshgrid(misalignment_ds.szen.data, misalignment_ds.apparent_szen.data)
    k = griddata((x.flatten(), y.flatten()),
                 misalignment_ds.k.data.flatten(),
                 (ds.SolarZenithAngle.data, ds.ApparentSolarZenithAngle.data))
    ds.broadband_flux.values = ds.broadband_flux.values * k
    if verbose:
        prints("... done", lvl=lvl)
    return ds
