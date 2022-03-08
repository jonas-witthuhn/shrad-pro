import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import re
import configparser

import modules.helpers as helpers
import modules.utils as utils
import modules.shcalc as shcalc
from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints
from modules.helpers import print_warning as printw

# read config file
CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
CONFIG.read("ConfigFile.ini")

# Parsing Command Line Arguments
parser = helpers.define_commandline_parser()
args = parser.parse_args()

if args.debug:
    args.verbose = True  # enable Verbose if Debug is active
    printd("Argparse Arguments:")
    printd(str(args))

MESSAGES = {'debug': args.debug,
            'verbose': args.verbose,
            'lvl': 0}

###############################################################################
# Run utility scripts
if args.ShradJob == "utils":
    prints("Run utility script on  GUVis files...",
           style='header',
           lvl=MESSAGES['lvl'])
    MESSAGES.update({'lvl': MESSAGES['lvl'] + 1})

    if args.utility_script == 'dangle':
        drolls, dpitchs, dyaws = [], [], []
        for input_file in args.input:
            if args.verbose:
                prints(str(f" Processing file {input_file} ..."),
                       lvl=MESSAGES['lvl'])
                MESSAGES.update({'lvl': MESSAGES['lvl'] + 1})
            ds = xr.open_dataset(input_file)
            dangles, delta_yaw_guvis = shcalc.estimate_guv2ins_misalignment(ds,
                                                                            args.dyaw)
            dr = np.round(dangles[0], 3)
            dp = np.round(dangles[1], 3)
            dy = np.round(delta_yaw_guvis, 3)
            drolls.append(dr)
            dpitchs.append(dp)
            dyaws.append(dy)
            if args.verbose:
                prints(f" (delta_roll, delta_pitch) = ({dr}, {dp})", lvl=MESSAGES['lvl'])
                prints(f" GUVis yaw relative to INS = {dy}", lvl=MESSAGES['lvl'])
                MESSAGES.update({'lvl': MESSAGES['lvl'] - 1})
        prints(f" Mean: (delta_roll, delta_pitch) = ({np.mean(drolls):.3f}, {np.mean(dpitchs):.3f})",
               lvl=MESSAGES['lvl'])
        prints(f" Std: (delta_roll, delta_pitch) = ({np.std(drolls):.3f}, {np.std(dpitchs):.3f})",
               lvl=MESSAGES['lvl'])
        prints(f" Mean: GUVis yaw relative to INS = {np.mean(dy):.3f}", lvl=MESSAGES['lvl'])
        prints(f" Std: GUVis yaw relative to INS = {np.std(dy):.3f}", lvl=MESSAGES['lvl'])

###############################################################################
# Start File processing
if args.ShradJob == "process":
    prints("Start Processing GUVis files...",
           style='header',
           lvl=MESSAGES['lvl'])
    MESSAGES.update({'lvl': MESSAGES['lvl'] + 1})

    ###########################################################################
    # From raw to level 1a
    if args.OutputLevel == 'l1a':
        if args.disable_ancillary_ins:
            printw(
                "You are not using additional INS data, so we rely on the position information of uLogger or GUVis GPS"
                " and the alignment angles of the included accelerometer. "
                "This is not recommended for the use on the ship,"
                " as alignment angles measured by an accelerometer on a ship are erroneous."
                " If the GUVis measures on a fixed position on Land this might be fine. Continuing...")
        # identify file prefix of GUVis files
        # the prefix will be added to all dataset files and
        # is required to match for ancillary datasets
        # and
        # identify date and time of file creation of the raw files
        result = utils.get_pfx_time_from_raw_input(pattern=args.datetimepattern,
                                                   input_files=args.input,
                                                   **MESSAGES)
        input_pfx, input_dates = result
        input_days = input_dates.astype('datetime64[D]')
        # process and combine files of unique days
        for day in np.unique(input_days):
            if args.verbose:
                prints(str(f" Processing day: {day} ..."),
                       lvl=MESSAGES['lvl'])
                MESSAGES.update({'lvl': MESSAGES['lvl'] + 1})

            # loading raw data to xarray Dataset
            process_files = np.array(args.input)[input_days == day]
            daily_ds = utils.load_rawdata_and_combine(files=process_files,
                                                      calib_file=args.calibration_file,
                                                      **MESSAGES)
            if type(daily_ds) == bool:
                # no data? skip day!
                if args.verbose:
                    MESSAGES.update({'lvl': MESSAGES['lvl'] - 1})
                    prints(str(f"... skipped!"),
                           lvl=MESSAGES['lvl'])
                continue

            # add PFX as global variable to DailyDS
            daily_ds = daily_ds.assign_attrs({'pfx': input_pfx})
            system_meta = dict(today=dt.datetime.today(),
                               pfx=input_pfx,
                               origin=process_files)
            daily_ds = utils.add_nc_global_attrs(ds=daily_ds,
                                                 system_meta=system_meta)

            # add ancillary data of inertial navigation system
            if not args.disable_ancillary_ins:
                # requires:
                #  * the correct paths in ConfigFile.ini
                #  * ancillary data in the correct netCDF format
                #  * correct naming of variables in netCDF:
                #     * time,pitch,roll,yaw,lat,lon
                #  * correct definition of angles in degrees:
                #     * pitch - positive if fore (bow) is up
                #               (platform_pitch_fore_up)
                #     * roll - positive if starboard is down
                #              (platform_roll_starboard_down)
                #     * yaw - positive if ship moves clockwise
                #             (platform_yaw_fore_starboard)
                #     * lat - positive northwards
                #     * lon - positive eastwards
                #   * alignment angles (pitch, roll, yaw) define
                #       the ships alignment and movement 
                #       (not the GUVis)
                daily_ds = utils.add_ins_data(ds=daily_ds, **MESSAGES)
                if type(daily_ds) == bool:
                    if args.verbose:
                        MESSAGES.update({'lvl': MESSAGES['lvl'] - 1})
                        prints(str(f"... skipped!"),
                               lvl=MESSAGES['lvl'])
                    continue

            # add sun position
            daily_ds = utils.add_sun_position(ds=daily_ds,
                                              coords=args.coordinates,
                                              **MESSAGES)

            # store to file
            output_filename = CONFIG['FNAMES']['l1a'].format(pfx=daily_ds.pfx,
                                                             date=pd.to_datetime(day))
            utils.store_nc(ds=daily_ds,
                           output_filename=output_filename,
                           overwrite=args.overwrite,
                           **MESSAGES)

            if args.verbose:
                MESSAGES.update({'lvl': MESSAGES['lvl'] - 1})
                prints(str(f"... done"),
                       lvl=MESSAGES['lvl'])

    ###########################################################################
    # From 1a to level 1b
    # correct misalignment and apply cosine correction
    if args.OutputLevel == 'l1b':
        for input_file in args.input:
            prints(str(f" Processing file {input_file} ..."),
                   lvl=MESSAGES['lvl'])
            MESSAGES.update({'lvl': MESSAGES['lvl'] + 1})

            # open input file
            ds_corrected = xr.open_dataset(input_file)

            # if selected, add ins data
            if args.add_ins:
                ds_corrected = utils.add_ins_data(ds=ds_corrected, **MESSAGES)
                if type(ds_corrected) == bool:
                    continue
                # re-calculate sun position
                ds_corrected = utils.add_sun_position(ds=ds_corrected, **MESSAGES)
            else:
                # check if necessary data is available
                keys = [key for key in ds_corrected.keys()]
                check = np.any([True for var in keys if re.match(".*Roll", var)])
                check *= np.any([True for var in keys if re.match(".*Pitch", var)])
                check *= np.any([True for var in keys if re.match(".*Latitude", var)])
                check *= np.any([True for var in keys if re.match(".*Longitude", var)])
                if not check:
                    raise ValueError("Input data has not enough data for alignment correction."
                                     " Requires at least Pitch, Roll, Latitude and Longitude")

            # define offset pitch, roll, yaw to ins
            if 'offset_angles' in args:
                dr, dp, dy = args.offset_angles
                drdp = (float(dr), float(dp))
                dy = float(dy)
            else:
                drdp, dy = shcalc.estimate_guv2ins_misalignment(ds_corrected,
                                                                **MESSAGES)
            ds_corrected = utils.add_offset_angles(ds_corrected, drdp, dy)

            # calculate apparent zenith and azimuth angles from ship and instrument
            ds_corrected = utils.add_apparent_zenith_angle(ds_corrected,
                                                           **MESSAGES)

            # correct uv cosine response
            if not args.disable_uvcosine_correction:
                # do UV channel corrections
                # for cosine response adjustment depending on
                # diffuser (cosine collector) and inlet
                # ! requires instrument specific calibration file
                # ! -> provided by Biospherical Inc.
                kwords = dict(ds=ds_corrected,
                              channels=args.uvcosine_correction_channel,
                              correction_file=args.uvcosine_correction_file,
                              **MESSAGES)
                ds_corrected = utils.correct_uv_cosine_response(**kwords)

            # correct cosine response error
            #  and correct misalignment
            ds_corrected = utils.correct_cosine_and_motion(ds=ds_corrected,
                                                           cosine_error_file=args.cosine_error_correction_file,
                                                           # misalignment_file=args.misalignment_correction_file,
                                                           **MESSAGES)

            # store to file
            day = ds_corrected.time.values[0].astype('datetime64[D]')
            output_filename = CONFIG['FNAMES']['l1b'].format(pfx=ds_corrected.pfx,
                                                             date=pd.to_datetime(day))
            utils.store_nc(ds=ds_corrected,
                           output_filename=output_filename,
                           overwrite=args.overwrite,
                           **MESSAGES)

            if args.verbose:
                MESSAGES.update({'lvl': MESSAGES['lvl'] - 1})
                prints(str(f"... done"),
                       lvl=MESSAGES['lvl'])





# calculate AOD
#         # add meteorological ancillary data
# if not args.disable_ancillary_met:
#     # requires:
#     #  * the correct paths in ConfigFile.ini
#     #  * ancillary data in the correct netCDF format
#     #  * correct naming of variables in netCDF:
#     #     * time,T,P,RH
#     #  * correct definition of units:
#     #     * T - [K] - air_temperature
#     #     * P - [Pa] - air_pressure
#     #     * RH - [1] - relative_humidity
#     daily_ds = utils.add_met_data(ds=daily_ds, **MESSAGES)
#     if type(daily_ds) == bool:
#         continue
