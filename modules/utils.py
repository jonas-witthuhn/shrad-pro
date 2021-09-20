import os
import re
import configparser
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints
from modules.helpers import print_warning as printw

CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
CONFIG.read("ConfigFile.ini")



def get_pfx_time_from_input(Pattern,InputFiles,
                            VERBOSE=True,DEBUG=False,lvl=0):
    """Parsing Date and Time according to --datetimepattern from raw GUVis files. In addition, the file prefix is identified for later use for for all dataset files and to identify ancillary data.
    """
    DatetimeParser = re.compile(Pattern)
        # identify input file dates
    InputDates = np.array([],dtype=np.datetime64)
    PFXs=[]
    for InputFile in InputFiles:
        _,filename = os.path.split(InputFile)
        pfx = filename.split('_')[0]
        m = DatetimeParser.match(filename)
        if DEBUG:
            printd(f"Parse: {InputFile}")
            printd(f"    -- Matches: {m.groupdict()}")
        m = m.groupdict()
        for key in ['year','month','day']:
            if not key in m.keys():
                raise ValueError(f"Cannot Identify Date with given option --datetimepattern: {args.datetimepattern}. Missing year, month or day searchgroup?")
        # string to int conversion
        if len(m['year']) == 2:
            m.update({'year':2000+int(m['year'])})
        m = dict([a, int(x)] for a, x in m.items())
        # convert to numpy datetime64
        InputDatetime = np.datetime64(dt.datetime(**m))
        InputDates = np.append(InputDates,InputDatetime)
        PFXs.append(pfx)
    PFX = np.unique(PFXs)
    if len(PFX) != 1:
        raise ValueError("Input files should always have the same prefix")
    PFX = str(PFX[0])
    return PFX,InputDates


def load_rawdata_and_combine(Files,
                             VERBOSE=True,DEBUG=False,lvl=0):
    """ Reading raw GUVis files and combine to one dataframe
    """
    
    if VERBOSE:
        prints(f"Load raw data from {len(Files)} file(s)...",
               lvl=lvl)
    if DEBUG:
        printd(f"Files for to process:")
        printd(str(Files))
    
    DF=[]
    for i,fname in enumerate(Files):               
        fsize=os.path.getsize(fname)/1024**2
        if fsize<0.1:
            printw(str(f'Skip file {fname} of size {fsize:.2f}MB.'+
                       ' ... less or no data.'))
            continue
        if VERBOSE:
            prints(f"Read file {i+1}/{len(Files)} with {fsize:.2f}MB.",
                   lvl=lvl+1)
            
        df = pd.read_csv(fname, sep=',', encoding=None)
        if type(DF)==list:
            DF=df.copy()
        else:
            DF=DF.append(df,ignore_index=True)
      
    if type(DF)==list:
        # no files, or all files are empty
        if VERBOSE:
            prints(str(f"No data - skipping.."),lvl=lvl)
        return False

    # homogenize  dataframe
    DF.drop_duplicates(subset='DateTimeUTCISO',keep='first',inplace=True)  
    DF.reset_index(drop=True,inplace=True)
    # remove the unit appendix in standard raw csv data of GUVis        
    keys = {}
    for k in DF.keys():
        keys.update({k:k.split()[0]})
    DF = DF.rename(keys,axis='columns')

    DS = xr.Dataset.from_dataframe(DF)
    # parse datetime for date objects
    for key in DS.keys():
        if key[:8]=='DateTime':
            datetime = pd.to_datetime(DS[key].values,
                                      infer_datetime_format=True)
            DS[key].values = datetime

    DS = DS.rename_vars({'DateTimeUTCISO':'time'})
#     DS = DS.set_coords(['time'])
    DS = DS.swap_dims({'index':'time'})
    DS = DS.reset_coords(names=['index'],drop=True)
    if VERBOSE:
        prints("... done",lvl=lvl)
    return DS
    
    
def add_ins_data(DS,
                 VERBOSE=True,
                 DEBUG=False,
                 lvl=0):
    day = pd.to_datetime(DS.time.values[0].astype("datetime64[D]"))
    fname = CONFIG['FNAMES']['ancillary'].format(pfx = DS.pfx,
                                                 AncillaryType='INS',
                                                 date=day)
    if DEBUG:
        printd(f"INS file: {fname}")
    if not os.path.exists(fname):
        printw("There is no INS data available for this day. I try skipping this day for now. Consider to disable the usage of ancillary INS data with --disable-ancillary-ins, or provide data for this day. Continuing...")
        return False
    INS = xr.open_dataset(fname)
    
    ## check variables
    required_vars = ['pitch','roll','yaw','lat','lon']
    available_vars = [var for var in INS.keys()]
    if not all(var in available_vars for var in required_vars):
        raise ValueError(f"The INS data given did not have all required variables {required_vars}!")
        
    # interpolate to DS
    INS = INS.interp_like(DS,
                          method='linear',
                          kwargs=dict(bounds_error=False,
                                      fill_value=np.nan))
    DS = xr.merge([DS,INS])
    return DS
    
    
    
    
def correct_uv_cosine_response(DS,
                               Channels,
                               File,
                               VERBOSE=True,
                               DEBUG=False,
                               lvl=0):
    """ Based on the diffuser material and inlet optic, the UV channels, e.g., Es305 and Es313 need additional cosine response correction. The correction factor is provided by Biospherical Inc. based on solar zenith angle.
    """
    # check if file is there:
    if not os.path.exists(File):
        raise ValueError(f"Cosine Correction of UVchannels is switched on, but can't find the file specified by --uvcosine-correction-file: {File}.")
    
    if DS.time.values[0]<np.datetime64("2016-02-29"):
        printw("For TROPOS GUVis-3511 SN:000350, the UVchannel cosine response correction is required only after Diffuser replacement on 2016-02-29. Consider switch of by setting --uvcosine-correction-disable")
    
    if VERBOSE:
        prints("Apply UV channel cosine response correction ...",lvl=lvl)
    CorrDS = pd.read_csv(File, sep=',')
    Channels = np.unique(Channels)
    for chan in Channels:
        if DEBUG:
            printd(f"Processing Es{chan} / Es{chan}_corr.")
        # delete data which is corrected by the uLogger software,
        # as this correction is applied to global irradiance only,
        # e.g., when the BioSHADE is in Z or P position.
        if f"Es{chan}_corr" in DS.keys():
            DS = DS.drop_vars(f"Es{chan}_corr")
        if f"Es{chan}" in DS.keys():
            c = griddata(CorrDS['SZA'],
                         CorrDS[f'Es{chan}'],
                         DS.SolarZenithAngle.values)
            DS[f'Es{chan}'].values = DS[f'Es{chan}'].values / c
    if VERBOSE:
        prints("... done",lvl=lvl)
    return DS
            
            
            