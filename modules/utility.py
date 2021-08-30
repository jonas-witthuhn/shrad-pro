import os
import re
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr


from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints


def get_datetime_from_input(Pattern,InputFiles,
                            VERBOSE=True,DEBUG=False,lvl=0):
    """Parsing Date and Time according to --datetimepattern from raw GUVis files
    """
    DatetimeParser = re.compile(Pattern)
        # identify input file dates
    InputDates = np.array([],dtype=np.datetime64)
    for InputFile in InputFiles:
        _,filename = os.path.split(InputFile)
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
    return InputDates


def load_rawdata_and_combine(Files,
                             VERBOSE=True,DEBUG=False,lvl=0):
    """ Reading raw GUVis files and combine to one dataframe
    """
    DF=[]
    for i,fname in enumerate(Files):               
        fsize=os.path.getsize(fname)/1024**2
        if fsize<0.1:
            prints(str(f'Skip file {fname} of size {fsize:.2f}MB.'+
                       ' ... less or no data.'),
                         lvl=lvl,
                         style='warning')
            continue
        if VERBOSE:
            prints(f"Read file {i+1}/{len(Files)} with {fsize:.2f}MB.",
                   lvl=lvl)
            
        df = pd.read_csv(fname, sep=',', encoding=None)
        if type(DF)==list:
            DF=df
        else:
            DF=df.append(df,ignore_index=True)
      
    if type(DF)==list:
        # no files, or all files are empty
        return False
    else:
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
        return DS