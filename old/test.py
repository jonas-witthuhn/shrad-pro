#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:48:39 2018

@author: walther
"""
import matplotlib
matplotlib.use("Agg")

import os
os.environ['TZ']='UTC'
import sys
import gc
import xarray as xr
gc.enable()
if sys.version_info[0]==3:
    from urllib.request import build_opener
    from urllib.request import install_opener
    from urllib.request import HTTPPasswordMgrWithDefaultRealm
    from urllib.request import HTTPBasicAuthHandler
    from urllib.request import HTTPCookieProcessor
    from http.cookiejar import CookieJar
else:
    from urllib2 import build_opener
    from urllib2 import install_opener
    from urllib2 import HTTPPasswordMgrWithDefaultRealm
    from urllib2 import HTTPBasicAuthHandler
    from urllib2 import HTTPCookieProcessor
    from cookielib import CookieJar

import getpass
import json
import numpy as np

from datetime import datetime

import filehandling as fh
from calc_aod import get_aod
from helpers import print_status
from make_plots import quicklookDATA,plot_TCtest
from shcalc import dangle


def alldays_pro(pf,pfx):
    dates=[]
    for p,d,fs in os.walk(pf+pfx+'/pro/',followlinks=True):
        for f in fs:
            if f[-3:]=='.nc':
                dates.append(np.datetime64('%s-%s-%s'%(f[-11:-7],f[-7:-5],f[-5:-3])))

    dates=np.sort(np.array(dates))        
    return dates

def alldays(pf,pfx):
    ldir=np.sort(np.array(os.listdir(pf+pfx+'/met/')))
    dates=[]
    for f in ldir:
        if f[-3:]=="dat":
            dates.append(np.datetime64('%s-%s-%s'%(f[:4],f[4:6],f[6:8])))
    if len(dates)==0:
        for p,d,fs in os.walk(pf+pfx+'/met/',followlinks=True):
            for f in fs:
                if f[-3:]=='.nc':
                    try:
                        dates.append(np.datetime64('%s-%s-%s'%(f[-11:-7],f[-7:-5],f[-5:-3])))
                    except:
                        dates.append(np.datetime64(f[:10]))
                    
    dates=np.sort(np.array(dates))        
    return dates

### retrieved from shcalc.dangle
dangles={'ps83':(-0.06,-1.558,0),
             'ps95':(0.410,-0.251,0),
             'ps98':(-0.429,1.490,0),
             'ps102':(0.022,0.721,0),
             'ps113':(-0.162,-1.127,0),
             'metpvnet':(0.,0.,0.),
             'tropostest':(-0.17,0.35,0.)}#


    

#datapf='/home/walther/Documents/Instruments/GUVis_scripts/ShRad/data/'
datapf='/vols/satellite/home/jonas/data/guvis/'



####


date=np.datetime64("2018-05-17")

filecfg={'pfx':'ps113',
         'serial':'000350',
         'ftype':'met',
         'rawflags':[]}

tiltcfg={'dangles':dangles[filecfg['pfx']],
         'def_roll':1,
         'def_pitch':-1,
         'def_yaw':1,
         'yaw_init':None}

lookups={'basepf':"/home/walther/Documents/Instruments/GUVis_scripts/ShRad/lookuptables/",
         'db_ang':"calibration/AngularResponse_GUV350_140129.csv",
         'db_c3':"motioncorrection/C3lookup_",
         'E_ext':"uncertainty/extraerror.dat",
         'E_tilt':"uncertainty/motion/motion_unc_",
         'db_response':'calibration/spectralResponse.dat',
         'db_o3':"O3Serdyuchenko2014_293K.txt",
         'db_no2':"NO2schneider1987.txt"}

nccfg='netcdf_cfg_guvis.json'
ncfmt={'campaign':'test',
       'script_name':'ShRad',
       'script_version':1.0,
       'user':'Witthuhn',
       'serial':'000350',
       'troposID':'A201400022',
       'met_instrument':'Polarstern_Weatherstation',
       'ins_instrument':'Polarstern_MINS',
       'dt':datetime.today()}

position={"latitude":None,
          "longitude":None,
          "h_station":15}####[m]

lvl=0

online=True

if online:
#    if sys.version_info[0]==3:
#        
#        omiuser=input("NASA GESDISC DATA ARCHIVE - User:")
#    else:
#        omiuser=raw_input("NASA GESDISC DATA ARCHIVE - User:")
#    omipsw=getpass.getpass("NASA GESDISC DATA ARCHIVE - password:")   
    omiuser='witthuhn@tropos.de'
    omipsw='Krank3rmist!'
    psw_manager=HTTPPasswordMgrWithDefaultRealm()
    psw_manager.add_password(realm=None,
                              uri='https://urs.earthdata.nasa.gov',
                              user=omiuser,
                              passwd=omipsw)
    cookie_jar = CookieJar()                          
    opener = build_opener(HTTPBasicAuthHandler(psw_manager),HTTPCookieProcessor(cookie_jar))
    install_opener(opener)


for pfx in ['metpvnet']:#,'ps83','ps95','ps98','ps102','ps113','mpv','tropostest'
#for pfx in ['ps95','ps98','ps102']:
#for pfx in ['ps83']:
    lvl=0
    
    
    filecfg['pfx']=pfx
    tiltcfg['dangles']=dangles[filecfg['pfx']]
    ncfmt['campaign']=pfx
    if pfx=='ps113':
        ncfmt['ins_instrument']='Polarstern_Hydrins'
        tiltcfg['def_pitch']=-1
        
    elif pfx=='ps83':
        position['h_station']=20
    elif pfx=='metpvnet': #MetPVNet
        tiltcfg['def_pitch']=1
        tiltcfg['yaw_init']=270.
        ncfmt['ins_instrument']='GUVis'
        ncfmt['met_instrument']='MORDOR'
        position['latitude']=47.715833
        position['longitude']=10.314133
        position['h_station']=718.
        ### new path extension
        datapf=datapf+"2018_2019_"
    elif pfx=='tropostest': #tropos roof
        tiltcfg['def_pitch']=1
        tiltcfg['yaw_init']=135.
        ncfmt['ins_instrument']='GUVis'
        ncfmt['met_instrument']='MORDOR'
        position['latitude']=51.353,
        position['longitude']=12.436
        position['h_station']=150. 
        ### new path extension
        datapf=datapf+"201912_"
    else:
        ncfmt['ins_instrument']='Polarstern_MINS'
        tiltcfg['def_pitch']=1

    dates=alldays(pf=datapf,
                  pfx=filecfg['pfx'])
    dates=np.array(dates)
    
    # dates=alldays_pro(pf=datapf,
    #               pfx=filecfg['pfx'])
    # dates=np.array(dates)
    dates=dates[dates.astype('datetime64[D]').astype(np.int)>np.datetime64('2019-07-01').astype(np.int)]
    # dates=[np.datetime64('2018-09-28')]
    
#    dangle(datapf=datapf,
#            filecfg=filecfg,
#            tiltcfg=tiltcfg,
#            lookups=lookups,
#            position=position,
#            lvl=lvl)
#    
    
    for date in dates:
#        try:
            lvl=0
            print_status("Processing %s date: %s"%(pfx,date),lvl=lvl,style='header')
            lvl+=1
            
            filecfg['ftype']='met'
            met=fh.load_met(date=date,
                            datapf=datapf,
                            filecfg=filecfg,
                            lvl=lvl)
            
    #        fh.new_nc(data=met,
    #                  date=date,
    #                  datapf=datapf,
    #                  filecfg=filecfg,
    #                  nccfg=nccfg,
    #                  ncfmt=ncfmt,
    #                  lvl=lvl)
    #        #
#            filecfg['ftype']='ins'
#            ins=fh.load_ins(date=date,
#                            datapf=datapf,
#                            filecfg=filecfg,
#                            tiltcfg=tiltcfg,
#                            lvl=lvl)
#    #        
#            fh.new_nc(data=ins,
#                      date=date,
#                      datapf=datapf,
#                      filecfg=filecfg,
#                      nccfg=nccfg,
#                      ncfmt=ncfmt,
#                      lvl=lvl)
    #        
            # filecfg['ftype']='raw'
            # filecfg['rawflags']=['C']
            # raw=fh.load_raw(date=date,
            #                 datapf=datapf,
            #                 filecfg=filecfg,
            #                 tiltcfg=tiltcfg,
            #                 lookups=lookups,
            #                 position=position,
            #                 lvl=lvl)
            # if type(raw)==type(None):
            #     print_status("No data for this date!",lvl=lvl-1,style='warning')
            #     continue
            # fh.new_nc(data=raw,
            #           date=date,
            #           datapf=datapf,
            #           filecfg=filecfg,
            #           nccfg=nccfg,
            #           ncfmt=ncfmt,
            #           lvl=lvl)
            # #
            filecfg['ftype']='raw'
            filecfg['rawflags']=['C','TC']
            raw_tc=fh.load_raw(date=date,
                            datapf=datapf,
                            filecfg=filecfg,
                            tiltcfg=tiltcfg,
                            lookups=lookups,
                            position=position,
                            lvl=lvl,debug=False)
            
# #            plot_TCtest(raw,raw_tc,
# #                        sdate=np.datetime64('2016-04-22T08:26:45'),
# #                        edate=np.datetime64('2016-04-22T19:29:15'),
# #                        Fu=760, Fo=850)
            
            
            
            # fh.new_nc(data=raw_tc,
            #           date=date,
            #           datapf=datapf,
            #           filecfg=filecfg,
            #           nccfg=nccfg,
            #           ncfmt=ncfmt,
            #           lvl=lvl)
        
            filecfg['ftype']='pro'
            pro=fh.load_pro(date=date,
                         datapf=datapf,
                         filecfg=filecfg,
                         tiltcfg=tiltcfg,      
                        lookups=lookups,
                        position=position,
                        lvl=lvl)
            # if type(pro)==type(None):
            #     print_status("No data for this date!",lvl=lvl-1,style='warning')
            #     continue
        
            
            
        
            # fh.new_nc(data=pro,
            #           date=date,
            #           datapf=datapf,
            #           filecfg=filecfg,
            #           nccfg=nccfg,
            #           ncfmt=ncfmt,
            #           lvl=lvl)
            
    
            
            filecfg['ftype']='aod'
            aod=fh.load_aod(date=date,
                            datapf=datapf,
                            filecfg=filecfg,
                            lvl=lvl)
        
            
            if aod==False:
                aod=None
                # aod=get_aod(raddat=pro,
                #             metdat=met,
                #             datapf=datapf+filecfg['pfx']+'/',
                #             basepf=lookups['basepf'],
                #             dbresponse=lookups['db_response'],
                #             db_o3=lookups['db_o3'],
                #             db_no2=lookups['db_no2'],
                #             h_station=position['h_station'],
                #             lvl=lvl)
                
                
                # if type(aod)!=type(None):
                #     fh.new_nc(data=aod,
                #               date=date,
                #               datapf=datapf,
                #               filecfg=filecfg,
                #               nccfg=nccfg,
                #               ncfmt=ncfmt,
                #               lvl=lvl)

                
    
            # quicklookDATA(datapf=datapf+filecfg['pfx']+'/',
            #               rad=pro,
            #               aod=aod)
#            
#            if aod==False:
#                aod={'aod':np.ma.masked_where(True,pro['Iglo']),'time':pro['time']}
#            fh.guv2pangaea(date,
#                           datapf+'/'+filecfg['pfx']+'/',
#                           aod,pro,
#                           altitude=position['h_station'])
            if type(pro)!=type(None):
                fh.pro2mesor_land(date,
                                  datapf+filecfg['pfx']+'/',
                                  pro,
                                  position)
            else:
                # only global irradiance (no shadowband)
                if type(raw_tc)!=type(None):
                    
                    times=np.datetime64('1970-01-01')+(raw_tc['time'][:]*1000.).astype('timedelta64[ms]')

                    ds = xr.Dataset({"channels":('chan',raw_tc['channels']),
                                     "Idir":(('time','chan'),np.zeros(raw_tc['rad'].shape)*np.nan),
                                     "Iglo":(('time','chan'),raw_tc['rad']),
                                     "Idif":(('time','chan'),np.zeros(raw_tc['rad'].shape)*np.nan),
                                     "zen":('time',raw_tc['zen']),
                                     "azi":('time',raw_tc['azi']),
                                     "calibF":('chan',raw_tc['calibF'])
                                     },
                                     coords = {"time":('time',times)})
                    ds = ds.coarsen(time=60*15,boundary='trim').mean(skipna=True)
                    ds = ds.resample(time='1min').interpolate('linear')
                    pro={"time":(ds.time.values-np.datetime64("1970-01-01")).astype('timedelta64[ms]').astype(float)/1000.,
                         "channels":raw_tc['channels'],
                         "Idir":ds.Idir.values,
                         "Iglo":ds.Iglo.values,
                         "Idif":ds.Idif.values,
                         "zen":ds.zen.values,
                         "azi":ds.azi.values,
                         "calibF":raw_tc['calibF']
                         }
                    fh.pro2mesor_land(date,
                                  datapf+filecfg['pfx']+'/',
                                  pro,
                                  position)
            if type(aod)!=type(None):
                fh.aod2mesor_land(date,
                                  datapf+filecfg['pfx']+'/',
                                  aod,
                                  pro,
                                  position)
                
            
            del met
            del pro
#            del aod
            gc.collect()
#        except:
#            print_status("FAIL: Processing %s date: %s"%(pfx,date),lvl=lvl,style='fail')
#            continue
#from get_toa import get_I0
#import numpy as np
#dates=np.arange("2014-04-01T10:00:00","2014-04-10T14:00:00",dtype='datetime64[h]')
#dates=[np.datetime64("2014-04-11T10:00:00"),np.datetime64("2016-04-01T10:00:00")]
#wvls=[340,625,875,900]
#for i,d in enumerate(dates):
#    i0,ei0=get_I0(d,wvls,lvl=0)
#    if i==0:
#        F=i0
#        eF=ei0
#    else:
#        F=np.vstack((F,i0))
#        eF=np.vstack((eF,ei0))
#np.save("F.npy",F)
#np.save("eF.npy",eF)        


#import os
#import sys
#import warnings
#
#import numpy as np
#import pandas as pd
#
#if sys.version_info[0]==3:
#    from urllib.request import urlopen
#else:
#    from urllib import urlopen
#
#from helpers import print_status



#
#date=np.datetime64("2014-04-01")
#
#url=str("http://lasp.colorado.edu/lisird/latis/"
#        +"sorce_tsi_24hr_l3.tab?&time>="
#        +pd.to_datetime(date).strftime("%Y-%m-%dT00:00:00&time<%Y-%m-%dT23:59:59"))
#
#
#print_status("testprint",lvl=2)
#print_status("Get 24h mean of totalat 1AU for date: "
#             +(pd.to_datetime(date).strftime("%Y-%m-%d 12:00 UTC")),lvl=1)
#print_status(url,lvl=1)
#warnings.warn("warning")
#
#u=urlopen(url)
#ulines=u.readlines()
#u.close()
#print(ulines)
#
#l=ulines[-1].decode()
#l=l.replace('\t' ,' ')
#print(l)