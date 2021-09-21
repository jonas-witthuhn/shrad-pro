#!/usr/bin/env python
# -*- coding: ISO-8859-1 -*-
"""
Created on Thu Sep 20 08:51:12 2018

@author: walther
"""
import os
import io
import sys
import json
import warnings
import gc
gc.enable()

from datetime import datetime
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata

import sunpos as sp
from helpers import print_status
import shcalc as calc

def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


### Helper functions ..........................................................
def _filename(date,filecfg):
    """Parse filename from file configuration, without suffix
    """
    name=''
    name+=filecfg['pfx'].lower()+'_'
    if filecfg['ftype']=='raw':
        name+='GUV_'
        name+=filecfg['serial']+'_'
        if 'U' in filecfg['rawflags']:
            name+='U_'
        if 'C' in filecfg['rawflags']:
            name+='C_'
        if 'TC' in filecfg['rawflags']:
            name+='TC_'
    elif filecfg['ftype']=='pro':
        name+='processed_'
    elif filecfg['ftype']=='aod':
        name+='AOD_'
    elif filecfg['ftype']=='ins':
        name+='INS_'
    elif filecfg['ftype']=='met':
        name+='MET_' 
    elif filecfg['ftype']=='test':
        name+='TEST_'
    else:
        raise ValueError("filehandling.new_nc -> filecfg['ftype'] not "
                         +"understood! Should be any of "
                         +"'raw','pro','aod','ins','met','test'.")
    datestr=pd.to_datetime(date.astype("datetime64[D]")).strftime("%Y%m%d")
    name+=datestr
    pfsfx='%s/%s/%s/%s/'%(filecfg['pfx'].lower(),
                          filecfg['ftype'],
                          datestr[:4],
                          datestr[4:6])
    return name,pfsfx

### Loading functions .........................................................
def load_raw(date,
             datapf='/../data/',
             filecfg={'pfx':'test',
                      'serial':'000350',
                      'ftype':'raw',
                      'rawflags':['C']},
             tiltcfg={'dangles':[0.,0.,0.],
                      'def_pitch':1,
                      'def_roll':1,
                      'def_yaw':1,
                      'yaw_init':None},
            lookups={'basepf':"../lookups/",
                     'db_ang':"AngularResponse_GUV350_140129.csv",
                     'db_c3':"motioncorrection/C3lookup_",
                     'E_tilt':"uncertainty/motion/motion_unc_"},
             position={"latitude":None,
                       "longitude":None},
             lvl=0,debug=False):
    r"""
    The raw GUVis data is parsed from either csv of nc data. Radiometric
    calibrations happen on previous calibrated and uncalibrated csv data.
    
    Parameters
    ----------
    date : np.datetime64, optional
        Date of the measurement day. For naming of the daily file.
    datapf : str, optional
        `datapf` represents the path-string where to find the data.
        default: "../data/"
    filecfg : dict, optional
        `filecfg` stores all information to choose the right nc-configuration
        from. Mandatory input: 
            'pfx' - Prefix used to sort for campaigns e.g.(ps83); 
            'serial' - Serial number of GUVis Radiometer
            'ftype' -  should be 'raw';
            'rawflags'  - list of flags to identify status of raw data
                ['C' - calibrated,
                'U' - uncalibrated,
                'TC' - calibrated and tilt corrected -> only nc-data]
    tiltcfg : dict, optional
        This is the input required to run the tilt correction function. The 
        dict contains yaw, roll and pitch definitions and offset angles.
        Default: 'dangles':[0,0,0] (pitch,roll,yaw),
                 'def_pitch':1 ->positive bow up,
                 'def_roll':1 -> positive portside up,
                 'def_yaw':1 -> positive clockwise from north
                 'db_ang':str ->path to angular response file
                 'db_c3': str->path to diffuse correction factor lookuptables
    position : dict, optional
        If GUVisGPS is not attached and there is no INS data, than latitude and
        longitude can be defined with this dictionary.
        Default: position=dict('latitude':None,'longitude';None)
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default=0    
            
    """  
    
    def _get_calibF(date):
        """get the actual calibration Factor for GUVis from 
        GUVis_calibrations.json
        """
        date=date.astype('datetime64[D]').astype(int)
        fn=str(os.path.split(os.path.realpath(__file__))[0]
                +'/GUVis_calibrations.json')
        with open(fn,'r') as f:
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
   
        ## interpolation linear between the closest two calibrations to get the
        ## 'actual' calibration with linear assumed drift
        Ca=griddata(cdates,values,date,method='linear')
        SNR=griddata(cdates,SNRs,date,method='linear')
        if np.all(np.isnan(Ca)):
            Ca=griddata(cdates,values,date,method='nearest')
            SNR=griddata(cdates,SNRs,date,method='nearest')
        
        ## stored calibration in GUVis uLogger
        Cs=values[stored,:][np.searchsorted(cdates[stored],date)-1,:]
        return channel,cwvl,Ca,Cs,SNR
        
    def _loadRAW(date,datapf,filecfg,lookups,position,tiltcfg,lvl):
        """Load uncalibrated /calibrated raw csv data from uLogger output 
        and calibrate with actual calibrations.
        """
        def _305correction(datet,lookup_pf,sza):
            C=pd.read_csv(os.path.join(lookup_pf,
                                       'calibration',
                                       'Correction_function_GUVis3511_SN351.csv'),
                          sep=',')
            c=griddata(C['SZA'],C['Es305'],sza)
            return 1./c

        datedtobj=pd.to_datetime(date.astype("datetime64[D]"))
        pfsfx=filecfg['pfx']+'/raw/'
        ldir=os.listdir(datapf+pfsfx)
        rawname,_=_filename(date,filecfg)
        rawname=rawname[:-8]+rawname[-6:] # cut out leading two digits from year
        counter=0
        for l in ldir:
            if l[:len(rawname)]==rawname and l[-4:]=='.csv':
                counter+=1
        print_status(str("Load raw data from %d file(s)"%counter
                         +" for date: %s ..."%(datedtobj.strftime("%Y-%m-%d"))),
                     lvl=lvl)
        N=counter
        counter=0
        pf=[]
        for l in np.sort(ldir):
            if l[:len(rawname)]==rawname and l[-4:]=='.csv':
                
                fsize=os.path.getsize(datapf+pfsfx+l)/1024**2
                if fsize<0.1:
                    print_status(str('Not load file %d/%d '%(counter,N)
                                     +' with %.2fMB'%(fsize) 
                                     +' ... less or no data.'),
                                 lvl=lvl,
                                 style='warning')
                    continue
                counter+=1
                print_status(str('Load file %d/%d'%(counter,N)
                                     +' with %.2fMB'%(fsize)),
                             lvl=lvl)
                pf1=pd.read_csv(datapf+pfsfx+l,sep=',',encoding = "ISO-8859-1")
                if type(pf)==list:
                    pf=pf1
                else:
                    pf=pf.append(pf1,ignore_index=True)
                  
        if type(pf)!=list:
            pf.drop_duplicates(subset='DateTimeUTCISO',keep='first',inplace=True)  
            pf.reset_index(drop=True,inplace=True)
            ### remove the unit appendix in standard raw csv data of GUVis        
            keys={}
            for k in pf.keys():
                keys.update({k:k.split()[0]})
            pf=pf.rename(keys,axis='columns')
            #np.save('test.npy',pf)
            if 'BioGpsTime' not in pf.keys():
                GPS=False
            else:
                GPS=True
            
            if 'BioShadeMode' not in pf.keys():
                SHADE=False
            else:
                SHADE=True
            ### calculate seconds since 1970
            tim=pf['DateTimeUTCISO'][:]
            datet=np.array(tim,
                           dtype=(np.datetime64))
            del(tim)
            datdt=np.array(datet-np.datetime64("1970-01-01T00:00:00.0"),
                           dtype=(int))/1000.#seconds since 1970
                           
            ### the same for GPS time if available
            if GPS:
                datetgps=np.array(pf['BioGpsTime'],dtype=(np.datetime64))
                datdtgps=np.array(datetgps-np.datetime64("1970-01-01T00:00:00.0"),
                                  dtype=(int))/1000.#seconds since 1970            
                                  
            ### get radiation values and replace corrected data with measured
            ### if possible
            if 'Es305_corr' in pf.keys():
                rad=pf.drop(columns='Es305_corr')
                # since Es305_corr is only corrected in global irradiance mode
                # we cannot use it, instead we correct it with the manufacturer
                # correction lookuptable later
                #rad['Es305']=pf['Es305_corr']    
                rad=rad.filter(regex='Es\d').values
            else:
                rad=pf.filter(regex='Es\d').values
               
            ### calibrate radiation data  to W m-2 nm with interpolating 
            ### between calibrations
            channel,cwvl,Ca,Cs,SNR=_get_calibF(date)
            
            ### [uW cm-2 nm]
            if 'U' in filecfg['rawflags']:
                rad=rad/Ca
            elif 'C' in filecfg['rawflags']:
                rad[:,:-1]=rad[:,:-1]*Cs[:-1]/Ca[:-1]
                rad[:,-1]=rad[:,-1]/Ca[-1]  ## Es0 is stored uncalibated everytime
            else:
                raise ValueError("_loadRAW -> filecfg['rawflags'] not "
                         +"understood! Should be any of "
                         +"'U','C'")
            ### [W m-2 nm]
            rad=rad*(10**-2) 

            ### radiometer and solar position
            pos=True
            INSdat=load_ins(date,
                            datapf,
                            {'pfx':filecfg['pfx'],'ftype':'ins'},
                            tiltcfg,
                            lvl+1)
            if GPS:
                lat=pf['BioGpsLatitude'].values
                lon=pf['BioGpsLongitude'].values
                ##Version correction (uLogger <v 1.0.24)
                if date<np.datetime64("2016-04-01"): 
                    ni=lat<0
                    lat[ni]=np.floor(lat[ni])+1.-(lat[ni]-np.floor(lat[ni]))
            else:
                if INSdat!=False:
                    instim=INSdat['time'][:]
                    lat=griddata(instim,INSdat['lat'][:],datdt)
                    lon=griddata(instim,INSdat['lon'][:],datdt)
                    zen,azi=sp.zenith_azimuth(sp.datetime2julday(datet),lat,lon)
                else:
                    try:
                        lat=np.ones(len(datdt))*position['latitude']
                        lon=np.ones(len(datdt))*position['longitude']
                    except:
                        pos=False
                        "There is no Position information"
                                      +"available! Further calculations"
                                      +"involving sun/radiometer position"
                                      +"are incorrect!")
            if pos:
                jday=sp.datetime2julday(datet)
                zen,azi=sp.zenith_azimuth(jday,lat,lon)
            
            ### Es305 correction
            # correct Es305 (after diffuser replacement in 29.02.2016)
            if np.min(datet)>np.datetime64("2016-02-29"):
                print_status(str("Apply Es305 correction"), lvl=lvl+1)
                c=_305correction(datet,lookups['basepf'],zen)
                rad[:,0]=rad[:,0]*c
                print_status(str("...done!"),lvl=lvl+1)


            ### make data dict
            rawdata={'time':datdt.astype(np.float),
                     "calibF":Ca,
                     "channels":channel,
                     "channels_cwvl":cwvl,
                     "channels_SNR":SNR,
                     "roll_guvis":pf['EsRoll'],
                     'pitch_guvis':pf['EsPitch']}
            if SHADE:
                rawdata.update({'shmode':pf['BioShadeMode'],
                                'shangle':pf['BioShadeAngle']})
            else:
                rawdata.update({'shmode': ['Z']*len(rawdata['time']),
                                'shangle':np.zeros(len(rawdata['time']))})
            if "EsTemp" in pf.keys():
                rawdata.update({'T_s':pf['EsTemp']})
            else:
                rawdata.update({'T_s':pf['Ed0Temp']}) #earlier uLogger version name
            if 'U' in filecfg['rawflags']:
                rawdata.update({'rad':rad*Ca})
            else:
                rawdata.update({'rad':rad})
                       
            if INSdat!=False:
                instim=INSdat['time'][:]
                rawdata.update({"roll":griddata(instim,INSdat['roll'][:],datdt),
                               "pitch":griddata(instim,INSdat['pitch'][:],datdt),
                               "yaw":griddata(instim,INSdat['yaw'][:],datdt)})
            else:
                rawdata.update({"roll":pf['EsRoll'],
                                'pitch':pf['EsPitch']})
                
            if GPS:
                rawdata.update({'gpstime':datdtgps.astype(np.float),
                                'satcount':pf['BioGpsSatelliteCount']})
            if pos:
                rawdata.update({'lat':lat, 'lon':lon,
                                'zen':zen, 'azi':azi})
            if 'AveragingCount' in pf.keys():
                rawdata.update({'avcount':pf['AveragingCount']})
                
        if counter==0:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return None
        else:
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return rawdata

    def _loadRAWnc(date,datapf,filecfg,lvl):
        print_status("Load raw data from nc-file",lvl=lvl)
        rawname,pfsfx=_filename(date,filecfg)
        rawname+='.nc'
        if os.path.exists(datapf+pfsfx+rawname):
            f=Dataset(datapf+pfsfx+rawname,'r')
            rawdata={}
            for v in f.variables:
                rawdata.update({v:f.variables[v][:]})
            f.close()
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return rawdata
        else:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return None
    ##########################################################################
    
    print_status("Load raw data...",lvl=lvl)
    if (filecfg['ftype']!='raw') or ('rawflags' not in filecfg.keys()):
        raise ValueError("ftype should be 'raw' and 'rawflags' should be defined")
    if 'TC' in filecfg['rawflags']:
        rawdata=_loadRAWnc(date,datapf,filecfg,lvl+1)
        if rawdata==None:
            rfilecfg=filecfg.copy()
            rfilecfg.update({'rawflags':['C']})
            craw=load_raw(date=date,
                          datapf=datapf,
                          filecfg=rfilecfg,
                          tiltcfg=tiltcfg,
                          lookups=lookups,
                          position=position,
                          lvl=lvl+1,
                          debug=debug)
            if type(craw)==type(None):
                print_status("...no data was found!",lvl,style='warning')
                print_status("",lvl-1)
                return None
            rfilecfg.update({'ftype':'ins'})
            ins=load_ins(date=date,
                         datapf=datapf,
                         filecfg=rfilecfg,
                         tiltcfg=tiltcfg,
                         lvl=lvl+1)
            rawdata=calc.correctRAW(craw,
                                     ins,
                                     tiltcfg['dangles'],
                                     tiltcfg['yaw_init'],
                                     dbang=lookups['basepf']+lookups['db_ang'],
                                     dbc3=lookups['basepf']+lookups['db_c3'],
                                     dbemotion=lookups['basepf']+lookups['E_tilt'],
                                     lvl=lvl+1,
                                     debug=debug)
    else:
        rawdata=_loadRAWnc(date=date,
                           datapf=datapf,
                           filecfg=filecfg,
                           lvl=lvl+1)
        if rawdata==None:
            rawdata=_loadRAW(date=date,
                             datapf=datapf,
                             filecfg=filecfg,
                             tiltcfg=tiltcfg,
                             lookups=lookups,
                             position=position,
                             lvl=lvl+1)
    if rawdata==None:
        print_status("...no data was found!",lvl,style='warning')
        print_status("",lvl-1)
        return rawdata
#        raise ValueError("No data for %s!"%(pd.to_datetime(date).strftime("%Y-%m-%d")))
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return rawdata    

def load_pro(date,
             datapf='/../data/',
             filecfg={'pfx':'test',
                      'serial':'000350',
                      'ftype':'pro',
                      'rawflags':[None]},
             tiltcfg={'dangles':[0.,0.,0.],
                      'def_pitch':1,
                      'def_roll':1,
                      'def_yaw':1,
                      'yaw_init':None},      
            lookups={'basepf':"../lookups/",
                    'db_ang':"AngularResponse_GUV350_140129.csv",
                      'db_c3':"motioncorrection/C3lookup_",
                      'E_ext':"uncertainty/extraerror.dat",
                      'E_tilt':"uncertainty/motion/motion_unc_",
                      'db_response':'calibration/spectralResponse.dat'},
            position={"latitude":None,
                       "longitude":None},
             lvl=0,
             debug=False):
                 
    def _loadPROnc(date,datapf,filecfg,lvl):
        print_status("Loading processed data...",lvl)
        fname,pfsfx=_filename(date,filecfg)
        fname+='.nc'
        if os.path.exists(datapf+pfsfx+fname):
            f=Dataset(datapf+pfsfx+fname,'r')
            prodata={}
            for v in f.variables:
                prodata.update({v:f.variables[v][:]})
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return prodata 
        else:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return False
    ##########################################################################
    print_status("Load processed data...",lvl=lvl)
    if filecfg['ftype']!='pro':
        raise ValueError("ftype should be 'pro'!")
    
    prodat=_loadPROnc(date,datapf,filecfg,lvl+1)
    if prodat==False:
        rfilecfg=filecfg.copy()
        rfilecfg.update({'ftype':'raw','rawflags':['C','TC']})
        craw=load_raw(date=date,
                      datapf=datapf,
                      filecfg=rfilecfg,
                      tiltcfg=tiltcfg,
                      lookups=lookups,
                      position=position,
                      lvl=lvl+1,debug=debug)
        if type(craw)==type(None):
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return None
        prodat=calc.processData(raw=craw,
                                lbasepf=lookups['basepf'],
                                Eext=lookups['E_ext'],
                                lresponse=lookups['db_response'],
                                lvl=lvl+1)
        if type(prodat)==type(None):
            print_status("... no sweeps to process!",lvl,style='warning')
            print_status("",lvl-1)
            return None
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return prodat 


def load_aod(date,
             datapf='../data/',
             filecfg={'pfx':'test'},
             lvl=0):
                
    def _loadAODnc(date,datapf,filecfg,lvl):
        print_status("Loading AOD data...",lvl)
        fname,pfsfx=_filename(date,filecfg)
        fname+='.nc'
        if os.path.exists(datapf+pfsfx+fname):
            f=Dataset(datapf+pfsfx+fname,'r')
            aoddata={}
            for v in f.variables:
                aoddata.update({v:f.variables[v][:]})
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return aoddata 
        else:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return False
    ########################################################################## 
    filecfg.update({'ftype':'aod'})
    print_status("Load AOD data...",lvl=lvl)
    aoddat=_loadAODnc(date,datapf,filecfg,lvl+1)
    if aoddat==False:
        print_status("...no data was found!",lvl,style='warning')
        print_status("",lvl-1)
        return False
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return aoddat
           
def load_ins(date,
             datapf='../data/',
             filecfg={'pfx':'test',
                      'serial':'000350',
                      'ftype':'ins',
                      'rawflags':[None]},
             tiltcfg={'dangles':[0.,0.,0.],
                      'def_pitch':1,
                      'def_roll':1,
                      'def_yaw':1,
                      'yaw_init':None},
             lvl=0):
    r"""
    The inertial naviagation sytem data (INS) is parsed
    from either csv of nc data.
    
    Parameters
    ----------
    date : np.datetime64, optional
        Date of the measurement day. For naming of the daily file.
    datapf : str, optional
        `datapf` represents the path-string where to find the data.
        default: "../data/"
    filecfg : dict, optional
        `filecfg` stores all information to choose the right nc-configuration
        from. Mandatory input: 
            'pfx' - Prefix used to sort for campaigns e.g.(ps83); 
            'ftype' -  should be 'ins';
    tiltcfg : dict, optional
        This is the input required to run the tilt correction function. The 
        dict contains yaw, roll and pitch definitions and offset angles.
        Default: 'dangles':[0,0,0] (pitch,roll,yaw),
                 'def_pitch':1 ->positive bow up,
                 'def_roll':1 -> positive portside up,
                 'def_yaw':1 -> positive clockwise from north
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default=0    
            
    """ 
    def _loadINS(date,datapf,filecfg,tiltcfg,lvl): 
        print_status("Loading INS (csv) data...",lvl)
        pfsfx=filecfg['pfx']+'/ins/'
        fname=datapf+pfsfx+pd.to_datetime(date).strftime("%Y%m%d")+'INS.dat'

        if os.path.exists(fname):
            pf=pd.read_csv(fname,sep='\t',skiprows=4,header=None)
            dat=pf.values
            dat=dat[:-1,:]
            if len(dat[0,0].split('/')[0])!=4:#if not year is first
                #move format dd/mm/yyyy to yyyy-mm-dd
                b=np.array(np.vstack(np.core.chararray.split(np.array(dat[:,0],
                                                        dtype='|U12'),'/')),
                                                        dtype='|S5')
                c=np.chararray(shape=(len(dat[:,0]),3),buffer=b,itemsize=5)
                dat[:,0]=np.chararray.decode(c[:,2]+b'-'+c[:,1]+b'-'+c[:,0])
            if len(dat[0,1])==8:#split seconds are missing
                datet=dat[:,0]+'T'+dat[:,1]+'.'
                datet=datet+np.array((np.arange(len(dat[:,0])))%10,dtype='|U1')#apply splitseconds to timestr
            else:
                datet=dat[:,0]+'T'+dat[:,1]
            datet=np.core.defchararray.replace(np.array(datet,dtype='|U21'),'/','-')#convert timestring to numpy format YYYY/MM//DD -> YYYY-MM-DD
            datet=np.array(datet,dtype=np.datetime64)
            tim=np.array(datet-np.datetime64("1970-01-01T00:00:00.0"),
                         dtype=(int))/1000. # seconds from 1970
            lat=dat[:,5]
            lon=dat[:,6]
            ltim=tim.copy()
            ltim=ltim[lat<9999]
            lon=lon[lat<9999]
            lat=lat[lat<9999]
            lat=griddata(ltim,lat,tim)
            lon=griddata(ltim,lon,tim)            
            insdat={'time':tim,
                    'yaw':dat[:,2]*tiltcfg['def_yaw'],
                    'pitch':dat[:,3]*tiltcfg['def_pitch'],
                    'roll':dat[:,4]*tiltcfg['def_roll'],
                    'lat':lat,
                    'lon':lon}
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return insdat
        else:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return False
            
    def _loadINSnc(date,datapf,filecfg,lvl):
        print_status("Loading INS (nc) data...",lvl)
        fname,pfsfx=_filename(date,filecfg)
        fname+='.nc'
        if os.path.exists(datapf+pfsfx+fname):
            f=Dataset(datapf+pfsfx+fname,'r')
            insdat={}
            for v in f.variables:
                insdat.update({v:f.variables[v][:]})
            f.close()
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return insdat
        else:
            ### look for old version nc
            fname=datapf+filecfg['pfx']+'/ins/'+pd.to_datetime(date).strftime("%Y%m%d")+'INS.nc'
            if os.path.exists(fname):
                f=Dataset(fname,'r')
                insdat={}
                for v in f.variables:
                    insdat.update({v:f.variables[v][:]})
                print_status("...done!",lvl,style='g')
                print_status("",lvl-1)
                return insdat
            else:
                print_status("...no data was found!",lvl,style='warning')
                print_status("",lvl-1)
                return False
            
        
    ##########################################################################

    print_status("Load INS data...",lvl=lvl)
    if filecfg['ftype']!='ins':
        raise ValueError("ftype should be 'ins'!")
    
    insdat=_loadINSnc(date,datapf,filecfg,lvl+1)
    if insdat==False:
        insdat=_loadINS(date,datapf,filecfg,tiltcfg,lvl+1)
        if insdat==False:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return False
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return insdat


def load_met(date,
             datapf='../data/',
             filecfg={'pfx':'test',
                      'serial':'000350',
                      'ftype':'met',
                      'rawflags':[None]},
             lvl=0):
    r"""
    The meteorological data (MET) is parsed
    from either csv of nc data.
    
    Parameters
    ----------
    date : np.datetime64, optional
        Date of the measurement day. For naming of the daily file.
    datapf : str, optional
        `datapf` represents the path-string where to find the data.
        default: "../data/"
    filecfg : dict, optional
        `filecfg` stores all information to choose the right nc-configuration
        from. Mandatory input: 
            'pfx' - Prefix used to sort for campaigns e.g.(ps83); 
            'ftype' -  should be 'met';
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default=0    
            
    """
    def _SMSmet2met(date,datapf,filecfg,lvl):
        print_status("Loading MET (csv) data (MORDOR)...",lvl)
        pfsfx=filecfg['pfx']+'/met/'
        datestr=pd.to_datetime(date).strftime("%Y-%m-%d")
        SMSmet_fname=datapf+pfsfx+'%s_Meteorologie.dat'%(datestr)
        SMSmet=np.loadtxt(SMSmet_fname,skiprows=4,delimiter=',',
                          converters={0: lambda s: (datetime.strptime(s.decode().strip('"'),"%Y-%m-%d %H:%M:%S")-datetime(1970,1,1)).total_seconds()})
        metdat={'time':SMSmet[:,0],
                'P':SMSmet[:,5],
                'T':SMSmet[:,2],
                'RH':SMSmet[:,3],
                'WD_rel':[np.nan],
                'WV_rel':[np.nan],
                'WD_tru':[np.nan],
                'WV_tru':[np.nan]}
        print_status("...done!",lvl,style='g')
        print_status("",lvl-1)
        return metdat

    def _SMSmet2metnc(date,datapf,filecfg,lvl):
        print_status("Loading MET (nc) data (MORDOR)...",lvl)
        pfsfx=filecfg['pfx']+'/met/'
        datestr=pd.to_datetime(date).strftime("%Y/%m/%Y-%m-%d")
        SMSmet_fname=datapf+pfsfx+'%s_Meteorologie.nc'%(datestr)
        f=Dataset(SMSmet_fname,'r') 
        metdat={'time':f.variables['time'][:],
                'P':f.variables['P'][:],
                'T':f.variables['T'][:],
                'RH':f.variables['RH'][:],
                'WD_rel':[np.nan],
                'WV_rel':[np.nan],
                'WD_tru':[np.nan],
                'WV_tru':[np.nan]}
        print_status("...done!",lvl,style='g')
        print_status("",lvl-1)
        return metdat
    
    def _loadMET(date,datapf,filecfg,lvl):
        print_status("Loading MET (csv) data ...",lvl)
        pfsfx=filecfg['pfx']+'/met/'
        fname=datapf+pfsfx+pd.to_datetime(date).strftime("%Y%m%d")+'MET.dat'
        if os.path.exists(fname):
            #check delimiter
            with io.open(fname,'r',encoding='ISO-8859-1') as txt:
                for i,line in enumerate(txt):
                    if i==4:
                        l=line
                        break
            s=l[10]#check delimiter
#            print(s,l)
            pf=pd.read_csv(fname,sep=s,header=None, comment='#',skiprows=4)
            dat=pf.values
#            print(dat)
            dat=dat[:-1,:]
            if len(dat[0,0].split('/')[0])!=4:#if not year is first
                #move format dd/mm/yyyy to yyyy-mm-dd
                b=np.vstack(np.char.split(np.array(dat[:,0],dtype='|U12'),'/').flatten())
                c=np.char.add(b[:,2],'-')
                c=np.char.add(c,b[:,1])
                c=np.char.add(c,'-')
                c=np.char.add(c,b[:,0])
                dat[:,0]=c
#                b=np.array(np.vstack(np.core.chararray.split(np.array(dat[:,0],
#                                                            dtype='|U12'),'/')),
#                                                            dtype='|S5')
#                c=np.chararray(shape=(len(dat[:,0]),3),buffer=b,itemsize=5)
#                dat[:,0]=np.chararray.decode(c[:,2]+b'-'+c[:,1]+b'-'+c[:,0])
            datet=np.array(dat[:,0]+'T'+dat[:,1],dtype='|U20')
            datet=np.char.replace(datet,'/','-')
#            datet=np.core.defchararray.replace(np.array(datet,dtype='|U21'),'/','-')#convert timestring to numpy format YYYY/MM//DD -> YYYY-MM-DD
            datet=np.array(datet,dtype=np.datetime64)
            tim=np.array(datet-np.datetime64("1970-01-01T00:00:00.0"),dtype=(int))/1000.             

            metdat={'time':tim,
                    'P':dat[:,6],
                    'T':dat[:,7],
                    'RH':dat[:,8],
                    'WD_rel':dat[:,9],
                    'WV_rel':dat[:,10],
                    'WD_tru':dat[:,11],
                    'WV_tru':dat[:,12]}          
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return metdat
        else:
            datestr=pd.to_datetime(date).strftime("%Y-%m-%d")
            SMSmet_fname=datapf+pfsfx+'%s_Meteorologie.dat'%(datestr)
            if os.path.exists(SMSmet_fname):
                metdat=_SMSmet2met(date,datapf,filecfg,lvl+1)
                print_status("...done!",lvl,style='g')
                print_status("",lvl-1)
                return metdat
            else:
                print_status("...no data was found!",lvl,style='warning')
                print_status("",lvl-1)
                return False
            
    def _loadMETnc(date,datapf,filecfg,lvl):
        print_status("Loading MET (nc) data...",lvl)
        fname,pfsfx=_filename(date,filecfg)
        fname+='.nc'
        fname=datapf+pfsfx+fname
        if os.path.exists(fname):
            f=Dataset(fname,'r')
            metdat={}
            for v in f.variables:
                metdat.update({v:f.variables[v][:]})
            f.close()
            print_status("...done!",lvl,style='g')
            print_status("",lvl-1)
            return metdat
        else:
            datestr=pd.to_datetime(date).strftime("%Y-%m-%d")
            SMSmet_fname=datapf+pfsfx+'%s_Meteorologie.nc'%(datestr)
            if os.path.exists(SMSmet_fname):
                metdat=_SMSmet2metnc(date,datapf,filecfg,lvl+1)
                print_status("...done!",lvl,style='g')
                print_status("",lvl-1)
                return metdat
            else:    
                print_status("...no data was found!",lvl,style='warning')
                print_status("",lvl-1)
                return False
    ##########################################################################
    
    print_status("Load MET data...",lvl=lvl)
    if filecfg['ftype']!='met':
        raise ValueError("ftype should be 'met'!")            

    metdat=_loadMETnc(date,datapf,filecfg,lvl+1)
    if metdat==False:
        metdat=_loadMET(date,datapf,filecfg,lvl+1)
        if metdat==False:
            print_status("...no data was found!",lvl,style='warning')
            print_status("",lvl-1)
            return False
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return metdat

### Make daily dataset (netcdf)................................................
def new_nc(data,
           date=np.datetime64('1970-01-01'),
           datapf='../data/',
           filecfg={'pfx':'test',
                    'serial':'000350',
                    'ftype':'test',
                    'rawflags':[None]},
           nccfg="netcdf_cfg_guvis.json",
           ncfmt={'campaign':'test',
                  'script_name':'test.py',
                  'script_version':1.0,
                  'user':'Witthuhn',
                  'serial':'000350:',
                  'troposID':'A201400022',
                  'dt':datetime.today()},
           lvl=0):
    r"""
    Save data dictionary `data` to netCDF4 file with netCDF attributes
    stored in `nccfg`.
    
    Parameters
    ----------
    data : dict
        `data` includes all data (one day) to store in nc file. Every 
        data.key() should be defined in `nccfg` file, except for 
        `filecfg`['ftype']=='test' data.
    date : np.datetime64, optional
        Date of the measurement day. For naming of the daily file.
        default: 1970-01-01
    datapf : str, optional
        `datapf` represents the path-string to store the data to.
        default: "../data/"
    filecfg : dict, optional
        `filecfg` stores all information to choose the right nc-configuration
        from. Mandatory input: 
            'pfx' - Prefix used to sort for campaigns e.g.(ps83); 
            'serial' - Serial number of GUVis Radiometer
            'ftype' -  should be any of 'raw','pro','aod','ins','met','test';
            'rawflags'  - list of flags to identify status of raw data
                ['C' - calibrated,
                'U' - uncalibrated,
                'TC' - tilt corrected]
    nccfg : str, optional
        Filename of the json config file which stores the ncattributes for each
        `data`.key().
    ncfmt : dict, optional
        `ncfmt` should include all keywords used to format global attributes in
        the `nccfg` json file.
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default=0    
            
    """
    gc.enable()
    if filecfg['ftype']=='raw':
        if 'U' in filecfg['rawflags'] and 'rad' in data.keys():
            data['rad_u']=data.pop('rad')
        elif 'C' in filecfg['rawflags'] and\
            'TC' not in filecfg['rawflags'] and\
            'rad' in data.keys():
            data['rad_c']=data.pop('rad')
        elif 'TC' in filecfg['rawflags'] and 'rad' in data.keys():
            data['rad_tc']=data.pop('rad')
    
    fname,pfsfx=_filename(date,filecfg)
    fname+='.nc'
    print_status("Save data to file: '%s%s'"%(datapf,fname),lvl=lvl)
    datestr=pd.to_datetime(date.astype("datetime64[D]")).strftime("%Y,%m,%d")
    year,month,day=datestr.split(',')
    
    #### load netCDF configuration.............................................
    fn=os.path.split(os.path.realpath(__file__))[0]+'/'+nccfg
    with open(fn,'r') as f:
        nccfg=json.load(f)

    #### load the data.........................................................
    tablename=filecfg['ftype']
        
    #### get config for table..................................................
    if tablename != "test":
        cfg=nccfg[tablename.lower()]
    #### create directory .....................................................
    sfx=''
    for s in pfsfx.split('/')[:-1]:
        sfx+=s+'/'
        if not os.path.exists(datapf+sfx):
            os.mkdir(datapf+sfx)
        
    #### create nc file   .....................................................   
    if not os.path.exists(datapf+pfsfx+fname):
        f=Dataset(datapf+pfsfx+fname,'w')
    else:
        os.remove(datapf+pfsfx+fname)
        f=Dataset(datapf+pfsfx+fname,'w')
#        warnings.warn('%s already exists!'%fname)
#        if sys.version_info[0]==3:
#            userout=input("Append:a Overwrite:w, else stop:")
#        else:
#            userout=raw_input("Append:a Overwrite:w, else stop:")
#        if userout=='a':
#            f=Dataset(datapf+pfsfx+fname,'a')
#        elif userout=='w':
#            os.remove(datapf+pfsfx+fname)
#            f=Dataset(datapf+pfsfx+fname,'w')
#        else:
#            sys.exit()
    ### separation for testcases
    if tablename=='test':
        dims=set()
        for v in data.keys():
            dims.update(set(data[v].shape))
        for i,d in enumerate(dims):
            f.createDimension("d%d"%i,d)
        for v in data.keys():
            dtuple=[]
            for i in data[v].shape:
                dtuple.append("d%d"%(np.argwhere(np.array(list(dims))==i)[0][0]))
            dtuple=tuple(dtuple)
            var=f.createVariable(v,data[v].dtype,dtuple)
            var[:]=data[v][:]
        f.close()
        print_status("...done!",lvl,style='g')
        print_status("",lvl-1)
        return 0
    
    ### create dimensions  ....................................................
    glocfg=cfg.pop('global')
    dims=glocfg.pop('dimensions')
    for d in dims.keys():
        if not dims[d]['name'] in f.dimensions:
            f.createDimension(dims[d]['name'],dims[d]['size'])  
    # write global attributes..............................................
    for attrs in glocfg:
        f.__setattr__(attrs, glocfg[attrs].format(**ncfmt))
    # create and fill variables............................................    
    for v in list(set(data.keys())&set(cfg.keys())):
#        print(v)
        vcfg=cfg[v]#variable config
        ccfg=vcfg.pop('createVariable') # create variable config
        if not ccfg['name'] in f.variables:
            d=np.array(data[v],dtype=ccfg['dtype'])
            if len(ccfg['dimensions'])>1:
                cs=[]
                for i in range(len(ccfg['dimensions'])):
                    cs.append(int(d.shape[i]/10.)+1)
                cs=tuple(cs)
            else:
                cs=None
#            print(cs)
            var=f.createVariable(ccfg['name'],
                                 ccfg['dtype'],
                                 tuple(ccfg['dimensions']),
                                 zlib=ccfg['zlib'],
                                 least_significant_digit=ccfg['lsd'],
                                 chunksizes=cs)
            for attrs in vcfg: 
                var.__setattr__(attrs, vcfg[attrs])
            var[:]=d

            del d
            del var
            gc.collect()
        else:
            f.variables[ccfg['name']][:]=np.array(\
                       list(f.variables[ccfg['name']][:])+\
                       list(np.array(data[v][:],dtype=ccfg['dtype'])))
       
    f.close()
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return 0

### to ascii converters........................................................
def guv2pangaea(date,datapf,aod,pro,altitude):

    head=str('Date/Time\tLatitude\tLongitude\tAltitude [m]\t'+
             'Ed_305 [W/m**2/nm]\tEd_340 [W/m**2/nm]\tEd_380 [W/m**2/nm]\t'+
             'Ed_412 [W/m**2/nm]\tEd_443 [W/m**2/nm]\tEd_510 [W/m**2/nm]\t'+
             'Ed_610 [W/m**2/nm]\tEd_625 [W/m**2/nm]\tEd_665 [W/m**2/nm]\t'+
             'Ed_694 [W/m**2/nm]\tEd_750 [W/m**2/nm]\tEd_765 [W/m**2/nm]\t'+
             'Ed_875 [W/m**2/nm]\tEd_940 [W/m**2/nm]\tEd_1020 [W/m**2/nm]\t'+
             'Ed_1245 [W/m**2/nm]\tEd_1550 [W/m**2/nm]\tEd_1640 [W/m**2/nm]\t'+
             'SWD [W/m**2]\t'+
             'Ei_305 [W/m**2/nm]\tEi_340 [W/m**2/nm]\tEi_380 [W/m**2/nm]\t'+
             'Ei_412 [W/m**2/nm]\tEi_443 [W/m**2/nm]\tEi_510 [W/m**2/nm]\t'+
             'Ei_610 [W/m**2/nm]\tEi_625 [W/m**2/nm]\tEi_665 [W/m**2/nm]\t'+
             'Ei_694 [W/m**2/nm]\tEi_750 [W/m**2/nm]\tEi_765 [W/m**2/nm]\t'+
             'Ei_875 [W/m**2/nm]\tEi_940 [W/m**2/nm]\tEi_1020 [W/m**2/nm]\t'+
             'Ei_1245 [W/m**2/nm]\tEi_1550 [W/m**2/nm]\tEi_1640 [W/m**2/nm]\t'+
             'DIR [W/m**2]\t'+
             'Ed-Ei_305 [W/m**2/nm]\tEd-Ei_340 [W/m**2/nm]\tEd-Ei_380 [W/m**2/nm]\t'+
             'Ed-Ei_412 [W/m**2/nm]\tEd-Ei_443 [W/m**2/nm]\tEd-Ei_510 [W/m**2/nm]\t'+
             'Ed-Ei_610 [W/m**2/nm]\tEd-Ei_625 [W/m**2/nm]\tEd-Ei_665 [W/m**2/nm]\t'+
             'Ed-Ei_694 [W/m**2/nm]\tEd-Ei_750 [W/m**2/nm]\tEd-Ei_765 [W/m**2/nm]\t'+
             'Ed-Ei_875 [W/m**2/nm]\tEd-Ei_940 [W/m**2/nm]\tEd-Ei_1020 [W/m**2/nm]\t'+
             'Ed-Ei_1245 [W/m**2/nm]\tEd-Ei_1550 [W/m**2/nm]\tEd-Ei_1640 [W/m**2/nm]\t'+
             'DIF [W/m**2]\t'+
             'AOT_340\tAOT_380\tAOT_412\t'+
             'AOT_443\tAOT_510\tAOT_610\t'+
             'AOT_625\tAOT_665\tAOT_694\t'+
             'AOT_750\tAOT_765\tAOT_875\t'+
             'AOT_940\tAOT_1020\tAOT_1245\tAOT_1550\tAOT_1640')

    channels=pro['channels']
    fmt='%19s\t%.5f\t%.5f\t%d'
    dt=[('dates','|S19'),('lat',np.float),('lon',np.float),('alt',np.int)]
    for i in range(len(channels)):
        fmt+='\t%s'
        dt.append(('ghi%d'%(channels[i]),'|S10'))
    for i in range(len(channels)):
        fmt+='\t%s'
        dt.append(('dir%d'%(channels[i]),'|S10'))
    for i in range(len(channels)):
        fmt+='\t%s'  
        dt.append(('dif%d'%(channels[i]),'|S10'))
    for i in range(len(channels)-2):
        fmt+='\t%s'  
        dt.append(('aod%d'%(channels[i+1]),'|S7'))
        
        
        
    if len(pro['Iglo'][:,:].mask.shape)!=0:
        empty=(pro['Iglo'][:,:].data<0)+(pro['Iglo'][:,:].data>2000)+np.isnan(pro['Iglo'][:,:].data)+pro['Iglo'][:,:].mask
    else:
        empty=(pro['Iglo'][:,:].data<0)+(pro['Iglo'][:,:].data>2000)+np.isnan(pro['Iglo'][:,:].data)
    emptyi=np.all(empty,axis=1)
    A=np.zeros(len(pro['time'][~emptyi]),dtype=np.dtype(dt))
    
    
    ind=np.searchsorted(pro['time'][~emptyi],aod['time']+1e-5)-1           

    dtim=np.datetime64('1970-01-01')+(pro['time'][~emptyi]*1000.).astype('timedelta64[ms]')
    dates=pd.to_datetime(dtim)
    
    A['dates']=dates.strftime("%Y-%m-%dT%H:%M:%S")
    A['lat']=pro['lat'][~emptyi].data
    A['lon']=pro['lon'][~emptyi].data
    A['alt']=altitude

    for i in range(len(channels)):
        glo=pro['Iglo'][~emptyi,i].data
        glo[glo<0]=np.nan
        glo[glo>2000]=np.nan
        glos=np.round(glo,5).astype('|S10')
        glos[np.isnan(glo)]=''
        A['ghi%d'%(channels[i])]=glos
        
        DIR=pro['Idir'][~emptyi,i].data
        DIR[DIR<0]=np.nan
        DIR[DIR>2000]=np.nan
        dirs=np.round(DIR,5).astype('|S10')
        dirs[np.isnan(DIR)]=''
        A['dir%d'%(channels[i])]=dirs
        
        DIF=pro['Idif'][~emptyi,i].data
        DIF[DIF<0]=np.nan
        DIF[DIF>2000]=np.nan
        difs=np.round(DIF,5).astype('|S10')
        difs[np.isnan(DIF)]=''
        A['dif%d'%(channels[i])]=difs
        
        if i!=len(channels)-1 and i!=0:
            sel=~(aod['aod'][:,i].mask+np.isnan(aod['aod'][:,i].data))
            if np.all(~sel):
                continue
            aods=aod['aod'][sel,i].data
            aods=np.round(aods,5).astype('|S7')
            inds=ind[sel]
            A['aod%d'%(channels[i])][inds]=aods
    
    

      
    if not os.path.exists(datapf+'pangaea'):
        os.mkdir(datapf+'pangaea')
    
    fname=datapf+'pangaea/'+pd.to_datetime(date).strftime('%Y-%m-%d_GUVis.tab')
    np.savetxt(fname,A,
               fmt=fmt,
               delimiter='\t',
               comments='',
               header=head)            
    return 0

def pro2mesor_land(date,datapf,pro,position):
    channels=pro['channels'][:-1]
    head=str('MESOR V1.1 11.2007\n'
             +'type squenceOfRadiationValue\n'
             +'unitName W/m^2/nm\n'
             +'valueType spectral Irradiance\n'
             +'IPR.providerName TROPOS\n'
             +'IPR.providerURL https://www.tropos.de\n'
             +'IPR.timeSeriesTitle MetPVNet\n'
             +'IPR.copyrightText TROPOS\n'
             +'location.latitude %f\n'%(position['latitude'])
             +'location.longitude %f\n'%(position['longitude'])
             +'location.height %.1f\n'%(float(position['h_station']))
             +'location.summarizationType 1m\n'
             +'spectral.begin 305\n'
             +'spectral.end 1640\n'
             +'spectral.unit nm\n'
             +'timezone UTC\n'
             +'comment location name MS01\n'
             +'comment instrument 0 manufacturer   Biospherical Instruments Inc.\n' 
             +'comment instrument 0 model GUVis-3511 + BioSHADE\n'
             +'comment instrument 0 serial 351 + 358\n'
             +'comment instrument 0 comment multispectral shadow-band radiometer\n' 
             +'comment instrument 0 calibrated from Biospherical Instruments Inc.\n' 
             +'comment instrument 0 calib remark 1 Calibration units  V uW-1 cm2 nm\n'
             +'comment instrument 0 calib remark 2 Calibration factor %s\n'%(str(pro['calibF'][:]))
             +'comment instrument 0 calib remark 3 Calibration date 2019-04-09\n'
             +'channel date "date" YYYY-MM-DD\n'
             +'channel time "time" HH:MM:SS\n')
    for i in range(len(channels)):
        head+=str('channel DNI "direct %snm" "W/m2nm"\n'%(str(channels[i]))
             +'channel GHI "global %snm" "W/m2"\n'%(str(channels[i]))
             +'channel DHI "diffuse %snm" "W/m2"\n'%(str(channels[i])))
    head+=str('channel SZEN "solar zenith angle" "degree"\n'
             +'channel SAZI "solar azimuth angle" "degree"\n'
             +'begindata')   
    
    fmt='%10s %11s'
    dt=[('dates','|U10'),
        ('times','|U11')]
    for i in range(len(channels)):
        fmt+=' %10.5f %10.5f %10.5f'
        dt.append(('dni%d'%(channels[i]),np.float))
        dt.append(('ghi%d'%(channels[i]),np.float))
        dt.append(('dhi%d'%(channels[i]),np.float))
    fmt+=' %7.2f %7.2f'
    dt.append(('zen',np.float))
    dt.append(('azi',np.float))
    
    A=np.zeros(len(pro['time'][:]),dtype=np.dtype(dt))
                 
    
    dtim=np.datetime64('1970-01-01')+(pro['time'][:]*1000.).astype('timedelta64[ms]')
    dates=pd.to_datetime(dtim)
    
    A['dates']=dates.strftime("%Y-%m-%d")
    A['times']=dates.strftime("%H:%M:%S.%f")
    for i in range(len(channels)):
        A['dni%d'%(channels[i])]=pro['Idir'][:,i]
        A['ghi%d'%(channels[i])]=pro['Iglo'][:,i]
        A['dhi%d'%(channels[i])]=pro['Idif'][:,i]
        A['dni%d'%(channels[i])][A['dni%d'%(channels[i])]<0]=np.nan
        A['ghi%d'%(channels[i])][A['ghi%d'%(channels[i])]<0]=np.nan
        A['dhi%d'%(channels[i])][A['dhi%d'%(channels[i])]<0]=np.nan
    A['zen']=pro['zen'][:]
    A['azi']=pro['azi'][:]
    if not os.path.exists(datapf+'mesor'):
        os.mkdir(datapf+'mesor')
    
    #print fmt
    fname=datapf+'mesor/'+pd.to_datetime(date).strftime('%Y-%m-%d_GUVis_Irradiance.mesor')
    np.savetxt(fname,A,
               fmt=fmt,
               delimiter=' ',
               comments='#',
               header=head,
               footer='enddata')     

def aod2mesor_land(date,datapf,aod,pro,position):
    channels=pro['channels'][:-1]
    head=str('MESOR V1.1 11.2007\n'
             +'type squenceOfAODValue\n'
             +'unitName #\n'
             +'valueType spectral aerosol optical depth\n'
             +'IPR.providerName TROPOS\n'
             +'IPR.providerURL https://www.tropos.de\n'
             +'IPR.timeSeriesTitle MetPVNet\n'
             +'IPR.copyrightText TROPOS\n'
             +'location.latitude %f\n'%(position['latitude'])
             +'location.longitude %f\n'%(position['longitude'])
             +'location.height %.1f\n'%(float(position['h_station']))
             +'location.summarizationType 1m\n'
             +'spectral.begin 305\n'
             +'spectral.end 1640\n'
             +'spectral.unit nm\n'
             +'timezone UTC\n'
             +'comment location name MS01\n'
             +'comment instrument 0 manufacturer   Biospherical Instruments Inc.\n' 
             +'comment instrument 0 model GUVis-3511 + BioSHADE\n'
             +'comment instrument 0 serial 351 + 358\n'
             +'comment instrument 0 comment multispectral shadow-band radiometer\n' 
             +'comment instrument 0 calibrated from Biospherical Instruments Inc.\n' 
             +'comment instrument 0 calib remark 1 Calibration units  V uW-1 cm2 nm\n'
             +'comment instrument 0 calib remark 2 Calibration factor %s\n'%(str(pro['calibF'][:]))
             +'comment instrument 0 calib remark 3 Calibration date 2019-04-09\n'
             +'channel date "date" YYYY-MM-DD\n'
             +'channel time "time" HH:MM:SS\n')
    for i in range(len(channels)):
        head+=str('channel AOD "%snm" "#"\n'%(str(channels[i])))
    head+=str('channel SZEN "solar zenith angle" "degree"\n'
             +'channel SAZI "solar azimuth angle" "degree"\n'
             +'begindata')   
    
    fmt='%10s %11s'
    dt=[('dates','|U10'),
        ('times','|U11')]
    for i in range(len(channels)):
        fmt+=' %8.5f'
        dt.append(('aod%d'%(channels[i]),np.float))

    fmt+=' %7.2f %7.2f'
    dt.append(('zen',np.float))
    dt.append(('azi',np.float))
    
    A=np.zeros(len(aod['time'][:]),dtype=np.dtype(dt))
                 
    
    dtim=np.datetime64('1970-01-01')+(aod['time'][:]*1000.).astype('timedelta64[ms]')
    dates=pd.to_datetime(dtim)
    
    A['dates']=dates.strftime("%Y-%m-%d")
    A['times']=dates.strftime("%H:%M:%S.%f")
    for i in range(len(channels)):
        A['aod%d'%(channels[i])]=aod['aod'][:,i]
        A['aod%d'%(channels[i])][A['aod%d'%(channels[i])]<0]=np.nan

    A['zen']=aod['zen'][:]
    A['azi']=aod['azi'][:]
    if not os.path.exists(datapf+'mesor'):
        os.mkdir(datapf+'mesor')
    
    #print fmt
    fname=datapf+'mesor/'+pd.to_datetime(date).strftime('%Y-%m-%d_GUVis_AOD.mesor')
    np.savetxt(fname,A,
               fmt=fmt,
               delimiter=' ',
               comments='#',
               header=head,
               footer='enddata')         
    
