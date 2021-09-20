import os
import numpy as np

import modules.utils as utils
import modules.helpers as helpers
from modules.helpers import print_status as prints
from modules.helpers import print_debug as printd
from modules.helpers import print_warning as printw


# config



# Parsing Command Line Arguments
parser = helpers.define_commandline_parser()
args = parser.parse_args()

# set verbose level

VerboseLevel = 0
if args.debug:
    args.verbose = True #  enable Verbose if Debug is activ
    printd("Argparse Arguments:")
    printd(str(args))

MESSAGES = {'DEBUG':args.debug,
            'VERBOSE':args.verbose,
            'lvl':VerboseLevel}
  
if args.disable_ancillary_ins:
    printw("You are not using additional INS data, so we rely on the position information of uLogger or GUVis GPS and the alignment angles of the included accelerometer. This is not recommended for the use on the ship, as alignment angles measured by an accelerometer on a ship are errorouse. If the GUVis measures on a fixed positon on Land this might be fine. Continuing...")

###############################################################################
### Start File processing
if args.ShradJob == "process":
    prints("Start Processing GUVis files...",
           style='header',
           lvl=MESSAGES['lvl'])
    MESSAGES.update({'lvl':MESSAGES['lvl']+1})
    
    ###########################################################################
    ### From raw to level 1a
    if args.OutputLevel == 'l1a':
        # identify file prefix of GUVis files
        # the prefix will be added to all dataset files and
        # is required to match for ancillary datasets
        # and
        # identify date and time of file creation of the raw files
        result = utils.get_pfx_time_from_input(Pattern=args.datetimepattern,
                                               InputFiles=args.input,
                                               **MESSAGES)
        InputPfx,InputDates = result
        InputDays = InputDates.astype('datetime64[D]')
        # process and combine files of unique days
        for day in np.unique(InputDays):
            if args.verbose:
                prints(str(f" Processing day: {day} ..."),
                           lvl=MESSAGES['lvl'])
                MESSAGES.update({'lvl':MESSAGES['lvl']+1})
            
            # loading raw data to xarray Dataset
            ProcessFiles = np.array(args.input)[InputDays==day]
            DailyDS = utils.load_rawdata_and_combine(Files=ProcessFiles,
                                                     **MESSAGES)
            if type(DailyDS) == bool:
                # no data? skip day!
                continue
                    
            # add PFX as global variable to DailyDS
            DailyDS = DailyDS.assign_attrs({'pfx':InputPfx})
            
            # add ancillary data of inertal navigation system
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
                DailyDS = utils.add_ins_data(DS=DailyDS,
                                             **MESSAGES)
                
                if type(DailyDS) == bool:
                    continue

                
            
            
            if not args.disable_uvcosine_correction:
                # do UV channel corrections
                # for cosine response adjustment depending on
                # diffuser (cosine collector) and inlet
                # ! requires instrument specific calibration file
                # ! -> provided by Biospherical Inc.
                kwords = dict(DS=DailyDS,
                              Channels=args.uvcosine_correction_channel,
                              File=args.uvcosine_correction_file,
                              **MESSAGES)
                DailyDS = utils.correct_uv_cosine_response(**kwords)
            

                        
                        
            

        #for date in np.unique(input_dates
            # read and combine daily input files



            # load calibration file and interpolate calibration factors
#             with open(args.calibration,'r') as f:
#                 calibrations=json.load(f)
