import numpy as np

import modules.utility as utility
import modules.helpers as helpers
from modules.helpers import print_status as prints
from modules.helpers import print_debug as printd


# config



# Parsing Command Line Arguments
parser = helpers.define_commandline_parser()
args = parser.parse_args()

# set verbose level
if args.verbose:
    VerboseLevel = 0
else:
    # Messages on negative level wont be printed
    VerboseLevel = -999
    
if args.debug:
    printd("Argparse Arguments:")
    printd(str(args))

MESSAGES = {'DEBUG':args.debug,
            'VERBOSE':args.verbose,
            'lvl':VerboseLevel}
            

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
        InputDates = utility.get_datetime_from_input(Pattern=args.datetimepattern,
                                                     InputFiles=args.input,
                                                     **MESSAGES)
        InputDays = InputDates.astype('datetime64[D]')
    
        # process unique days separately
        for day in np.unique(InputDays):
            ProcessFiles = np.array(args.input)[InputDays==day]
            if args.verbose:
                prints(str(f"Load raw data from {len(ProcessFiles)} file(s)"+
                           f" for date: {day} ..."),
                       lvl=MESSAGES['lvl'])
            if args.debug:
                printd(f"Files for date: {day}:")
                printd(str(ProcessFiles))
            
            DailyDS = utility.load_rawdata_and_combine(Files=ProcessFiles,
                                                       **MESSAGES)
            

        #for date in np.unique(input_dates
            # read and combine daily input files



            # load calibration file and interpolate calibration factors
#             with open(args.calibration,'r') as f:
#                 calibrations=json.load(f)
