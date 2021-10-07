#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:51:12 2018

@author: walther
"""
import argparse
import sys
import textwrap
import warnings

# setting global intention level and tab
glvl = 0
gtab = '  |'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if sys.version_info[0] == 3:
    def _warning(message,
                 category=UserWarning,
                 filename='',
                 lineno=-1,
                 file='',
                 line=''):
        """Replace the default warnings.showwarning function to include in print_status
        """
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
        print_status(str(msg.message), glvl, gtab, 'WARNING: ', 'warning')


    warnings.showwarning = _warning
else:
    def _warning(message,
                 category=UserWarning,
                 filename='',
                 lineno=-1):
        """Replace the default warnings.showwarning function to include in print_status
        """
        print_status(message[0], glvl, gtab, 'WARNING: ', 'warning')


    warnings.showwarning = _warning


def print_debug(txt):
    print_status(txt, lvl=0, pfx='DEBUG: ', style='b')


def print_warning(txt):
    warnings.warn(txt)


def print_status(txt, lvl=0, tab='  |', pfx='', style='', end='', flush=False):
    r"""
    Print `txt` to stout with defined intention level for easy looks ;)
    
    Parameters
    ----------
    txt : str
        `txt` represents the text to print to stout.
    lvl : int, optional
        `lvl` represents the intention level of the message to print. It sets
        the global `glvl` variable.
        default: 0
    tab : str, optional
        `tab` represents the prefix string which is repeated `lvl` times. It
        sets the global `gtab` variable.
        default: '  |'
    pfx : str, optional
        `pfx` represents the prefix printed directly before `txt`.
        default: ''
    style : str, optional
        `style` choose the style of `txt`. Default is None.
        It can be anything of  ['blue' or 'b',
                                'green' or 'g',
                                'fail',
                                'warning',
                                'bold',
                                'header',
                                'underline']
    """
    if lvl < 0:
        return 0
    global glvl
    glvl = lvl
    offset = ''
    lines = textwrap.wrap(txt, width=69 - lvl * len(tab) - len(pfx), break_long_words=False)
    for i, l in enumerate(lines):
        if i != 0:
            offset = '  '
        if (style.lower() == 'blue') or (style.lower() == 'b'):
            pl = bcolors.OKBLUE + l + bcolors.ENDC
        elif (style.lower() == 'green') or (style.lower() == 'g'):
            pl = bcolors.OKGREEN + l + bcolors.ENDC
        elif style.lower() == 'fail':
            pl = bcolors.FAIL + l + bcolors.ENDC
        elif style.lower() == 'warning':
            pl = bcolors.WARNING + l + bcolors.ENDC
        elif style.lower() == 'bold':
            pl = bcolors.BOLD + l + bcolors.ENDC
        elif style.lower() == 'header':
            pl = bcolors.HEADER + l + bcolors.ENDC
        elif style.lower() == 'underline':
            pl = bcolors.UNDERLINE + l + bcolors.ENDC
        else:
            pl = l
        print(lvl * tab + pfx + offset + pl + end)
        if flush:
            sys.stdout.flush()
    return 0


def define_commandline_parser():
    """ Setup of the command line parser
    """

    def _add_default(pars):
        pars.add_argument('input', nargs='*',
                          help=textwrap.dedent("""\
                                               Input files to perform task on.
                                                 - Use unix wildcards to specify multiple files
                                                 - Path can be relative or absolute
                                               """))

        pars.add_argument('-v', '--verbose', action='store_true',
                          help="enable processing status messages")
        pars.add_argument('-d', '--debug', action='store_true',
                          help="enable additional debug messages")
        pars.add_argument('-o', '--overwrite', action='store_true',
                          help="overwrite existing output files")
        return pars

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='ShRad jobs', dest='ShradJob',
                                       help=textwrap.dedent("""\
                                                            Choose:
                                                                *utils: collection of utility scripts
                                                                *process: advance input GUVis files in processing level
                                                            """))
    # utils parser
    utils_parser = subparsers.add_parser('utils', formatter_class=argparse.RawTextHelpFormatter,
                                         description=textwrap.dedent("""\
                                                                     Run utility scripts.
                                                                     """))
    utils_subparser = utils_parser.add_subparsers(title='utility scripts', dest='utility_script',
                                                  help=textwrap.dedent("""\
                                                                       Choose:
                                                                           dangle: Calculate offset angles between
                                                                                   INS platform and GUV instrument
                                                                       """))
    dangle_parser = utils_subparser.add_parser("dangle",
                                               formatter_class=argparse.RawTextHelpFormatter,
                                               description=("Calculate offset angles of GUV instrument set up on "
                                                            "the INS platform. These are required for the "
                                                            "cosine error and misalignment correction using INS data. "
                                                            "Requires the input of l1a netcdf data."))
    dangle_parser = _add_default(dangle_parser)

    # process parser
    process_parser = subparsers.add_parser('process', formatter_class=argparse.RawTextHelpFormatter,
                                           description=textwrap.dedent("""\
                                                                       Advance input files in processing level.
                                                                       """))

    process_subparser = process_parser.add_subparsers(title='Output data level', dest='OutputLevel',
                                                      help=textwrap.dedent("""\
                                                                           Choose:
                                                                               l1a: calibrated GUVis data
                                                                               l1b: calibrated and corrected GUVis data
                                                                               l1c: processed irradiance components
                                                                               l2_aod: spectral aerosol optical depth
                                                                           """))

    # process l1a
    l1a_parser = process_subparser.add_parser("l1a",
                                              formatter_class=argparse.RawTextHelpFormatter,
                                              description=("Process raw .csv data files to level 1a:"
                                                           " calibrated data in daily files of netCDF format."))

    l1a_parser = _add_default(l1a_parser)
    l1a_parser.add_argument('--datetimepattern', type=str,
                            default=".*(?P<year>[0-9]{2})(?P<month>[0-9]{2})(?P<day>[0-9]{2})[_](?P<hour>[0-9]{2})(?P<minute>[0-9]{2}).*",
                            help=textwrap.dedent("""\
                Search pattern of date and time for raw data files.
                At least, the search groups <year>,<month> and <day> have to be defined.
                Possible Searchgroup names:
                    year, month, day,
                    hour, minute, second, microsecond
                A Year of two digits will be interpreted as 2000 + year.
                However, the retrieved dates of all input files have to be unique.
                The default is:
                %(default)s
                """))
    l1a_parser.add_argument('--calibration-file', type=str,
                            default="data/GUVis_calibrations.json",
                            help=textwrap.dedent("""\
                            Path to absolute spectral calibration file (.json),
                            for all channels of the GUVis. Using this file,
                            the radiation channels are calibrated with drift
                            corrected calibration factors, by linear interpolation
                            of specified calibration factors and dates in the file.
                            If this is not wanted and the files are pre-calibrated
                            form the GUVis internal storage, than this can be
                            switched of by specifying '--calbration-file ""'.
                            The default is:
                            %(default)s
                            """))
    l1a_parser.add_argument("--disable-ancillary-ins",
                            action="store_true",
                            help=textwrap.dedent("""\
                            Disables the use of ancillary 
                            inertal navigation system (INS) data.
                            If not set, the use is enabled and the 
                            following parameters are acquired from the
                            ancillary database (see ConfigFile.ini):
                                * Position of the GUVis: latitude, longitude
                                * Alignment of the GUVis: pitch, roll, yaw
                            Subsequently, the solar zenith and azimuth angle
                            will be calculated based on this information.
                            """))
    l1a_parser.add_argument('--coordinates', nargs=2,
                            default=[False, False],
                            help=textwrap.dedent("""\
                            Latitude [degree East] and Longitude [degree North],
                            required if the observation is stationary on land and
                            no GPS information is available.
                            The default is %(default)s.
                            """))

    # process l1b
    l1b_parser = process_subparser.add_parser("l1b",
                                              formatter_class=argparse.RawTextHelpFormatter,
                                              description=("Process level 1a files to level 1b:"
                                                           " adding misalignment and cosine error correction."))
    l1b_parser = _add_default(l1b_parser)
    l1b_parser.add_argument('--uvcosine-correction-channel', action='append', type=int,
                            default=[305, 313],
                            help=textwrap.dedent("""\
                            Add to the list of channels to apply cosine response correction
                            (for UV channels (e.g., 305,313))
                            based on Biospherical correction file.
                            Requires the file specified by --uvchannel-correction-file.
                            The default channels are %(default)s if available.
                            """))
    l1b_parser.add_argument('--uvcosine-correction-file', type=str,
                            default="data/Correction_function_GUVis3511_SN351.csv",
                            help=textwrap.dedent("""\
                            File of correction factors for UV-channel cosine response correction.
                            File requires comma separated columns and a header line
                            (solar zenith angle, channel names ...), e.g.:
                                SZA,Es305,Es313
                            Provided by Biospherical Inc. (Instrument specific).
                            The default is %(default)s.
                            """))
    l1b_parser.add_argument('--disable-uvcosine-correction',
                            action='store_true',
                            help=textwrap.dedent("""\
                            Optionally switch off UV channel cosine correction.
                            """))
    l1b_parser.add_argument("--add-ins",
                            action="store_true",
                            help=textwrap.dedent("""\
                            Adds or overwrites ancillary 
                            inertal navigation system (INS) data.
                            If set, the following parameters are acquired from the
                            ancillary database (see ConfigFile.ini):
                                * Position of the GUVis: latitude, longitude
                                * Alignment of the GUVis: pitch, roll, yaw
                            Subsequently, the solar zenith and azimuth angle
                            will be re-calculated based on this information.
                            """))
    l1b_parser.add_argument("-a", "--offset-angles", nargs=3,
                            default=[0, 0, 0],
                            help=textwrap.dedent("""\
                            Set the offset-angles (pitch, roll, yaw) for the
                            difference of ships INS system to instrument set up.
                            For land station, yaw can be used to define the
                            orientation of the instrument.
                            Angles are in degrees, orientation as follows:
                            * pitch - positive if fore (bow) is up
                            * roll - positive if starboard is down
                            * yaw - positive if ship moves clockwise
                                    or clockwise from north
                            The default is %{default}s.
                            """))

    # process l1c
    l1c_parser = process_subparser.add_parser("l1c",
                                              formatter_class=argparse.RawTextHelpFormatter,
                                              description="""Process level 1b files to level 1c:
                                                          separate GHI, DHI, and DNI from sweep data
                                                          and add cloud mask.""")
    # process l2aod
    l2aod_parser = process_subparser.add_parser("l2aod",
                                                formatter_class=argparse.RawTextHelpFormatter,
                                                description="""Process level 1c files to level 2 - AOD:
                                                            calculate aerosol optical depth from DNI observations.""")
    #     l2aod_parser.add_argument("--disable-ancillary-met",
    #                             action="store_true",
    #                             help=textwrap.dedent("""\
    #                             Disables the use of ancillary
    #                             meteorological (INS) data.
    #                             If not set, the use is enabled and the
    #                             following parameters are acquired from the
    #                             ancillary database (see ConfigFile.ini):
    #                                 * Basic meteorological parameters:
    #                                     * air temperature
    #                                     * air pressure
    #                                     * relative humidity
    #                             This data is required for estimation
    #                             of Rayleigh-scattering OD and WV-OD
    #                             for the AOD calculation.
    #                             """))
    return parser
