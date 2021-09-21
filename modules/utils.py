import os
import re
import configparser
import json
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

import trosat.sunpos as sp

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
                             CalibFile="",
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
    units = []
    variables = []
    for k in DF.keys():
        ksplit = k.split(' ',1)
        ksplit += [''] * (2 - len(ksplit))
        units.append(ksplit[1])
        variables.append(ksplit[0])
        keys.update({k:ksplit[0]})
    DF = DF.rename(keys,axis='columns')

    
    DS = xr.Dataset.from_dataframe(DF)
    # parse datetime for date objects
    for key in DS.keys():
        if key[:8]=='DateTime':
            datetime = pd.to_datetime(DS[key].values,
                                      infer_datetime_format=True)
            DS[key].values = datetime

    DS = DS.rename_vars({'DateTimeUTCISO':'time'})
    DS = DS.swap_dims({'index':'time'})
    DS = DS.reset_coords(names=['index'],drop=True)
    
    
    ### Bug correction for uLogger version < 1.0.24
    if ("BioGpsTime" in list(DS.keys()))and(DS.time.values[0]<np.datetime64("2016-04-01")):
        lat=DS.BioGpsLatitude.values
        ni=lat<0
        lat[ni]=np.floor(lat[ni])+1.-(lat[ni]-np.floor(lat[ni]))
        DS.BioGpsLatitude.values = lat
    
    ### detect uncalibrated files
    # index of first radiation data
    idx = [i for i,var in enumerate(variables) if re.match("Es\d", var)][0]
    radunit = units[idx]
    
    if CalibFile:
        if VERBOSE:
            prints("Calibrate Radiation data ...", lvl=lvl+1)
        # get the calibration
        CDS = get_calibration_factor(date=DS.time.values[0],
                                     File=CalibFile)
        if not re.match(".*V",radunit):
            if DEBUG:
                printd(" Files are precalibrated with from the GUVis internal storage, now uncalibrate...")
            # data is already calibrated with stored calibration,
            # thus, remove stored calibration befor calibration
            # with drift corrected calibration factor
            for channel in CDS.channel.values:
                if channel == 0:
                    # broadband channel calibration is never stored
                    continue
                cds = CDS.sel(channel=channel)
                Cs = cds.calibration_factor_stored
                DS[f"Es{channel}"] = DS[f"Es{channel}"] * Cs 

        # apply calibration
        if DEBUG:
            printd(" Now Calibrate the radiation data ...")
        for channel in CDS.channel.values:
            cds = CDS.sel(channel=channel)
            Ca = cds.calibration_factor
            DS[f"Es{channel}"] = DS[f"Es{channel}"] / Ca 
        if VERBOSE:
            prints("... done",lvl=lvl+1)
    else:
        if DEBUG:
            printd("No CalibrationFile is specified. Assuming precalibrated files.")
        # no Calibratiofile is given,
        # assuming precalibrated files...
        if re.match(".*V",radunit):
            # Files are not calibrated
            raise ValueError("Files are not calibrated, at least the ASCII Header tells so. Please specify a --calibration-file.")
    
    if VERBOSE:
        prints("... done",lvl=lvl)
    return DS

def get_calibration_factor(date, File):
    """
    Retrieve the corrected calibration factor for GUVis from 
    GUVis_calibrations.json
    
    Parameters
    ----------
    date: numpy.datetime64
        Day of the data
    File: str
        Path to the calibration file (.json)
    
    Returns
    -------
    CDS: xarray.Dataset
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
    date=date.astype('datetime64[D]').astype(int)
    with open(File,'r') as f:
        calibrations=json.load(f)
    
    
        
    channel=calibrations['_CHANNEL']
    cwvl=calibrations['_CENTROID_WVL']
    cdates=list(calibrations['calibration'].keys())
    cdates=np.array(cdates,dtype='datetime64[D]').astype(int)
    values=[]
    SNRs=[]
    stored=[]
    for c in calibrations['calibration'].keys():
        val=np.array(calibrations['calibration'][c]['calibF'])
        val[val==None]=np.nan
        snr=np.array(calibrations['calibration'][c]['SNR'])
        snr[snr==None]=np.nan
        stored.append(calibrations['calibration'][c]['stored'])
        if len(values)==0:
            values=np.array(val)
            SNRs=np.array(snr)
        else:
            values=np.vstack((values,val))
            SNRs=np.vstack((values,snr))
    stored=np.array(stored,dtype=bool)

    si=np.argsort(cdates)
    cdates=cdates[si]
    values=np.array(values[si,:],dtype=np.float)
    SNRs=np.array(SNRs[si,:],dtype=np.float)
    stored=stored[si]

    ## fill nan values with interpolatet values
    for i in range(len(values[0,:])):
        mask=np.isnan(values[:,i])
        values[mask,i]=np.interp(np.flatnonzero(mask),
                                 np.flatnonzero(~mask),
                                 values[~mask,i])
        mask=np.isnan(SNRs[:,i])
        SNRs[mask,i]=np.interp(np.flatnonzero(mask),
                                 np.flatnonzero(~mask),
                                 SNRs[~mask,i])

    ## interpolation linear between the closest two calibrations
    ## to correct calibration assuming a linear drift
    Ca=griddata(cdates,values,date,method='linear')
    SNR=griddata(cdates,SNRs,date,method='linear')
    if np.all(np.isnan(Ca)):
        Ca=griddata(cdates,values,date,method='nearest')
        SNR=griddata(cdates,SNRs,date,method='nearest')

    ## stored calibration in GUVis uLogger
    Cs=values[stored,:][np.searchsorted(cdates[stored],date)-1,:]
    
    CDS = xr.Dataset({'centroid_wvl':('channel',cwvl),
                      'calibration_factor':('channel',Ca),
                      'calibration_factor_stored':('channel',Cs),
                      'signal_noise_ratio':('channel',SNR)},
                     coords={'channel':('channel',channel)})
    
    return CDS

    
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
    if VERBOSE:
        prints("Merging INS data ...",lvl=lvl)
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
    if VERBOSE:
        prints("... done",lvl=lvl)
    
    return DS
    
def add_met_data(DS,
                 VERBOSE=True,
                 DEBUG=False,
                 lvl=0):
    day = pd.to_datetime(DS.time.values[0].astype("datetime64[D]"))
    fname = CONFIG['FNAMES']['ancillary'].format(pfx = DS.pfx,
                                                 AncillaryType='MET',
                                                 date=day)
    if DEBUG:
        printd(f"MET file: {fname}")
    if VERBOSE:
        prints("Merging MET data ...",lvl=lvl)
    if not os.path.exists(fname):
        printw("There is no MET data available for this day. I try skipping this day for now. Consider to disable the usage of ancillary MET data with --disable-ancillary-met, or provide data for this day. Continuing...")
        return False
    MET = xr.open_dataset(fname)
    
    ## check variables
    required_vars = ['T','P','RH']
    available_vars = [var for var in MET.keys()]
    if not all(var in available_vars for var in required_vars):
        raise ValueError(f"The MET data given did not have all required variables {required_vars}!")
        
    # interpolate to DS
    MET = MET.interp_like(DS,
                          method='linear',
                          kwargs=dict(bounds_error=False,
                                      fill_value=np.nan))
    DS = xr.merge([DS,MET])
    if VERBOSE:
        prints("... done",lvl=lvl)
    
    return DS
    
def add_sun_position(DS,
                     coords=[False,False]
                     VERBOSE=True,
                     DEBUG=False,
                     lvl=0):
    # get latitude/longitude from preferred sources
    # Sources: BioGPS > Ships INS > provided stationary coords
    latkeys = ['BioGpsLatitude','lat']
    lonkeys = ['BioGpsLongitude','lon']
    

    
    
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
                         DS.SensorZenithAngle.values)
            DS[f'Es{chan}'] = DS[f'Es{chan}']/c
    if VERBOSE:
        prints("... done",lvl=lvl)
    return DS
            
            
            