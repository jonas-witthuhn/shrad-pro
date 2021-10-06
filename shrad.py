import numpy as np
import datetime as dt
import configparser

import modules.helpers as helpers
import modules.utils as utils
import pandas as pd
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

if args.disable_ancillary_ins:
    printw("You are not using additional INS data, so we rely on the position information of uLogger or GUVis GPS"
           " and the alignment angles of the included accelerometer. This is not recommended for the use on the ship,"
           " as alignment angles measured by an accelerometer on a ship are erroneous."
           " If the GUVis measures on a fixed position on Land this might be fine. Continuing...")

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
        # identify file prefix of GUVis files
        # the prefix will be added to all dataset files and
        # is required to match for ancillary datasets
        # and
        # identify date and time of file creation of the raw files
        result = utils.get_pfx_time_from_input(pattern=args.datetimepattern,
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
                    continue

            # add sun position
            daily_ds = utils.add_sun_position(ds=daily_ds,
                                              coords=args.coordinates,
                                              **MESSAGES)

            # store to file
            output_filename = CONFIG['FNAMES']['l1a'].format(pfx=input_pfx,
                                                             date=pd.to_datetime(day))
            utils.store_nc(ds=daily_ds,
                           output_filename=output_filename,
                           overwrite=args.overwrite,
                           **MESSAGES)



# alignment correction
#             if not args.disable_uvcosine_correction:
#                 # do UV channel corrections
#                 # for cosine response adjustment depending on
#                 # diffuser (cosine collector) and inlet
#                 # ! requires instrument specific calibration file
#                 # ! -> provided by Biospherical Inc.
#                 kwords = dict(ds=daily_ds,
#                               channels=args.uvcosine_correction_channel,
#                               correction_file=args.uvcosine_correction_file,
#                               **MESSAGES)
#                 daily_ds = utils.correct_uv_cosine_response(**kwords)


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
