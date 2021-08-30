#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:52:10 2018

@author: walther
"""
import os
import sys
if sys.version_info[0]==3:
    from urllib.request import urlopen
    from urllib.request import urlretrieve
else:
    from urllib2 import urlopen


from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter1d as gauss

from helpers import print_status
import get_toa


def _od(flux,flux_0,coszen):
    """
    claculate optical depth from beer-lambert-law
    flux - array(len(zen),len(wvl)) - spectral direct normal Irradiance [W/m2um]
    wvl - array() - channel wavelenghts
    coszen - array() - cosine of solar zenit angles
    """

    
    tau=-1.*np.log(flux/flux_0)
    tau=(tau.T*coszen).T
    return tau


def get_aod(raddat,
            metdat,
            datapf,
            basepf,
            dbresponse,
            db_o3,
            db_no2,
            h_station,
            online=True,
            lvl=0,
            cloudscreen=True,
            request_raw=False):
    print_status('Calculate AOD.',lvl=lvl)
    
    
    time=raddat['time'][:]
    if type(time)==np.ma.MaskedArray:
        mask=time.mask
        time=time.data
        time[mask]=np.nan
        
    index=~np.isnan(time)
    time=time[index]
    
    time64=np.datetime64('1970-01-01')+(time*1000).astype('timedelta64[ms]')
    mdate=np.median(time64.astype('datetime64[D]').astype(int)).astype('datetime64[D]')
    mdate=pd.to_datetime(mdate)
    ### radiation data
    Idir=raddat['Idir'][index,:-1]
    zen=raddat['zen'][index]
    azi=raddat['azi'][index]
    coszen=np.cos(zen*np.pi/180.)
    lat=raddat['lat'][index]
    lon=raddat['lon'][index]

    ### uncertainty estimates of Idir
    Eext=raddat['E_ext'][index]
    Eali=raddat['E_ali'][index,:-1]
    Enoi=raddat['E_noise'][index,:-1]
    
    ### meteorological data
    p=metdat['P'][:]
    t=metdat['T'][:]
    q=metdat['RH'][:]
    mettim=metdat['time'][:]
    p=griddata(mettim,p,time)
    t=griddata(mettim,t,time)
    q=griddata(mettim,q,time)
    
#    print(time64.shape)
#    print()
#    print(time64)
    channels=raddat['channels'][:-1]
    I0,eI0=get_toa.get_I0(date=np.median(time64.astype('datetime64[D]').astype(int)).astype('datetime64[D]'),
                      wvls=channels,
                      cwvls=raddat['channels_cwvl'][:-1],
                      basepf=basepf,
                      dbresponse=dbresponse,
                      lvl=lvl+1)
    
    ### total
    OD=_od(Idir,I0,coszen)
    

    o3DU=None
    no2col=None

    o3DU=_get_o3col(date=mdate,
                    datapf=datapf,
                    lat=np.nanmean(lat),
                    lon=np.nanmean(lon),
                    lvl=lvl+1)
    no2col=_get_no2col(date=mdate,
                       datapf=datapf,
                       lat=np.nanmean(lat),
                       lon=np.nanmean(lon),
                       lvl=lvl+1)
        
    ### ozone
    if o3DU!=None:
        o3_od=_o3od(wvl=channels,
                    basepf=basepf,
                    db_o3=db_o3,
                    o3DU=o3DU)
    else:
        o3_od=_o3od(wvl=channels,
                    basepf=basepf,
                    db_o3=db_o3)
        o3DU=np.nan
    
    ### no2
    if no2col!=None:
        no2_od=_no2od(wvl=channels,
                      no2col=no2col,
                      basepf=basepf,
                      db_no2=db_no2)
    else:
        no2_od=_no2od(wvl=channels,
                      basepf=basepf,
                      db_no2=db_no2)
        no2col=np.nan
        
    ### rayleigh scattering
    rod=_rayod(wvl=channels,
               P=p,
               lat=lat,
               alt=h_station)
    
    ### ch4 and co2 like aeronet
    ch4_od=np.zeros((len(p),len(channels)))
    ch4_od[:,-1]=0.0036*p/1013.25
    co2_od=np.zeros((len(p),len(channels)))
    co2_od[:,-1]=0.0089*p/1013.25
    
    ### aerosol optical depth
    AOD=OD-rod-o3_od-no2_od-co2_od-ch4_od
    
    ### water correction
    h2o_od,e_h2o_od=_h2ood(wvl=channels,
                           cosz=coszen,
                           I0=I0[13],
                           eI0=eI0[13],
                           aod440=AOD[:,4],
                           aod870=AOD[:,12],
                           I940=Idir[:,13],
                           w440=raddat['channels_cwvl'][4],
                           w870=raddat['channels_cwvl'][12],
                           w940=raddat['channels_cwvl'][13])

    AOD=AOD-h2o_od

    E_aod=_E_AOD(date=mdate,
                 datapf=datapf,
                 sza=zen,
                 I=Idir,
                 I0=I0,
                 eI0=eI0,
                 aod=AOD,rod=rod,o3OD=o3_od,no2OD=no2_od,h2ood=h2o_od,
                 E_ext=Eext,E_ali=Eali,E_noi=Enoi,e_h2o_od=e_h2o_od,P=p,dP=5)

    AOT=(AOD.T/coszen).T
    if len(AOD)<=3:
        return None
    
    
    aoddat={'time':time,
             'aod':AOD}
    if request_raw:
        return aoddat
    
    if cloudscreen:
        index=_cloudscreen(aoddat,lvl=lvl+1)
        if type(index)==type(None):
            print_status("... no aod data for this date!",lvl,style='warning')
            print_status("",lvl-1)
            return None
    else:
        index=np.arange(len(aoddat['time']))
    
#    AOD smoothing
    aods=[]
    time64=np.datetime64('1970-01-01')+time[index].astype('timedelta64[s]')#seconds resolution
    for i,wvl in enumerate(channels):
        df=pd.DataFrame({'time':time64,
                        'aod':AOD[index,i]})
        df_resample=df.resample('1s',on='time').first()
        dfr=df_resample['aod'].rolling(300,center=True,min_periods=2)
        me=dfr.mean()[~np.isnat(df_resample['time'])]
        if type(aods)==type([]):
            aods=np.array([me.values])
        else:
            aods=np.vstack((aods,me.values))
    aods=aods.T

    aoddata={'time':time[index],
             'aod':aods,#AOD[index,:]
             'lat':lat[index],
             'lon':lon[index],
             'aot':AOT[index,:],
             'od':OD[index,:],
             'od_ray':rod[index,:],
             'od_o3':o3_od[:],
             'od_no2':no2_od[:],
             'od_h2o':h2o_od[index,:],
             'od_co2':co2_od[index,:],
             'od_ch4':ch4_od[index,:],
             'o3DU':o3DU,
             'no2DU':(no2col)/(0.44615*6.022*10**20),
             'E_aod':E_aod[index,:],
             'P':p[index],
             'zen':zen[index],
             'azi':azi[index],
             'channels':channels,
             'channels_cwvl':raddat['channels_cwvl'][:-1]}

 

    if len(aoddata['time'])<=3:
        print_status("... no aod data for this date!",lvl,style='warning')
        print_status("",lvl-1)
        return None
        
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return aoddata


def _cloudscreen(aoddat,lvl=0):
    def _plot(tim,aod):
        from matplotlib import pyplot
        time=np.datetime64('1970-01-01')+tim.astype('timedelta64[s]')
        pyplot.figure()
        pyplot.plot(time,aod,marker='.',linestyle='')
        pyplot.grid(True)
        pyplot.show()
    print_status('Screening for clouds etc.',lvl=lvl)
    time=aoddat['time'][:]
    ctime=time.copy()
    aod=aoddat['aod'][:,:]
    aod=aod[~np.isnan(time),:]
#    np.save("test.npy",time)
    time=time[~np.isnan(time)]
    
    ### minimum cutoff
#    _plot(time,aod[:,[1]])
    aod[aod<0]=np.nan
#    _plot(time,aod[:,[1,5]])
    
    ### triplet variability
    df=pd.DataFrame({'time':np.datetime64('1970-01-01')+time.astype('timedelta64[s]'),
                    'aod340':aod[:,1]})
    df_resample=df.resample('1s',on='time').first()
    tim=(df_resample['time'][:]-np.datetime64('1970-01-01')).astype(int)*10**-9
#    _plot(tim,df_resample['aod340'])
    
    dfr=df_resample['aod340'].rolling(300,center=True,min_periods=2)
    print(len(dfr.min()),np.count_nonzero(~np.isnat(df_resample['time'])))
    mi=dfr.min()[~np.isnat(df_resample['time'])]
    ma=dfr.max()[~np.isnat(df_resample['time'])]
    me=dfr.mean()[~np.isnat(df_resample['time'])]
    mi=dfr.min()[np.isin(tim,time.astype('timedelta64[s]').astype(int))]
    ma=dfr.max()[np.isin(tim,time.astype('timedelta64[s]').astype(int))]
    me=dfr.mean()[np.isin(tim,time.astype('timedelta64[s]').astype(int))]
#    _plot(time,np.vstack([me*0.1,ma-mi]).T)
    index=ma-mi<np.where(0.02>me*0.1,np.ones(len(me))*0.02,me*0.1)

    time=time[index]
    aod=aod[index,5]
#    _plot(time,aod)
    ###third check daylie smoothness
    while True:
        if len(aod)<=3:
            return None
        daystd=np.nanstd(aod)
        if daystd<0.015:
#            print('below std')
            print_status('done!',lvl=lvl-1,style='green')
            return np.isin(ctime,time)
        taui=np.log(aod[:-2])
        taui1=np.log(aod[1:-1])
        taui2=np.log(aod[2:])
        ti=time[:-2]/(3600.)
        ti1=time[1:-1]/(3600.)
        ti2=time[2:]/(3600.)
        Ds=(((taui-taui1)/(ti-ti1))-((taui1-taui2)/(ti1-ti2)))**2
        D=np.sqrt((1./(len(Ds)))*np.nansum(Ds))
#        print(D,daystd)
        if D>16.:
            maxi=np.nanargmax(Ds)
            maxtaui=np.nanargmax(aod[maxi:maxi+3])
            maxi=maxi+maxtaui
            aod=np.delete(aod[:],maxi,0)
            time=np.delete(time,maxi,0)
        else:
            break
#    _plot(time,aod)
    index=np.ones(len(aod[:]))
    index[aod>np.nanmean(aod)+3.*np.nanstd(aod)]=0
    index[aod<np.nanmean(aod)-3.*np.nanstd(aod)]=0
    
    time=time[index==1]
#    _plot(time,aod[index==1])

    index=np.isin(ctime,time)
    print_status('done!',lvl=lvl-1,style='green')
#    _plot(aoddat['time'][index],aoddat['AOD'][index,5])
    return index


def _rayod(wvl,P=np.array([1013.25]),lat=45.,alt=0.):
    """
    calculate rayleigth optical depth 
    wvl - array() - channel wavelenghts [nm]
    P - array() - Pressure in hPa
    lat - array() (len(P)) or float - latitude north [deg]
    alt - same shape as lat - altitude [m] 
    return: tau - array(len(P),len(wvl)) - opticalthicknes
    """ 
    
    #ray od calculation after Bodhaine et al. 1999
    T=288.15
    cCO2=400.
    A=6.0221367*10.**(23)      
    
    c=np.asarray(cCO2)
    w=np.asarray(wvl)/1000.#nm -> um
    P=np.asarray(P)*1000.# hPa in dyn/cm^2
    lat=np.asarray(lat)
    alt=np.asarray(alt)
    
    Ns=(A/22.4141)*(273.15/T)*(1./1000.) 
    FN=1.034 + 3.17*(10**(-4))/(w**2)
    FO=1.096 + 1.385*(10**(-3))/(w**2)+1.448*(10**(-4))/(w**4)
    F_air=(78.084*FN + 20.946*FO + 0.934 + (cCO2/10000.)*1.15)/(78.084 + 20.946 + 0.934 + (cCO2/10000.))
    n=((8060.51+2480990/(132.274-w**(-2))+17455.7/(39.32957-w**(-2)))/(10.**8))+1.
    n= ((1.+0.54*((c*(10.**(-6)))-0.0003))*(n-1.))+1. 
    
    sigma=(24.*np.pi**3 *(n**2-1.)**2)/((0.0001*w)**4 * Ns**2 *(n**2 +2.)**2)*F_air
    m_a=15.0556*c*1.0e-6 + 28.9595
    
    cos2phi=np.cos(2.*lat*np.pi/180.)
    g0=980.6160*(1.-0.0026373*cos2phi+0.0000059*cos2phi**2)
    z=0.73737*alt+5517.56
    g=(g0-(3.085462e-4 + 2.27e-7 * cos2phi)*z + 
       (7.254e-11 + 1.0e-13 * cos2phi)*z**2 - 
       (1.517e-17 + 6e-20 * cos2phi)*z**3)

    sigmas,P=np.meshgrid(sigma,P)
    sigmas,g=np.meshgrid(sigma,g)
    tau_r=sigmas*P*A/(m_a*g)       
    return tau_r



def _no2od(wvl,basepf,db_no2,no2col=3.*10.**19):
    """
    calculate no2 optical depth from schneider 1987 crosssection
    wvl-array()- wavelenghts [nm]
    no2col - float - no2 column amount molekules/m**2
    return tau- array(len(cosz),len(wvl)) - optical depth
    """
    wvl=np.array(wvl)   
    no2cs=np.loadtxt(basepf+db_no2)
    no2cs[:,1]=gauss(no2cs[:,1],10./(2.*np.sqrt(2.*np.log(2.))))
    wvls=no2cs[:,0]
    taus=(no2cs[:,1]/(100.**2.))*no2col
    taus=griddata(wvls,taus,wvl)
    taus[np.isnan(taus)]=0
    tau=taus
    return tau

def _get_no2col(date,datapf,lat,lon,lvl=0):
    if not os.path.exists(datapf+"no2/"):
        os.mkdir(datapf+"no2/")
    if os.path.exists(datapf+"no2/no2.dat"):
        d=np.loadtxt(datapf+"no2/no2.dat",delimiter=',')
        d=np.vstack((np.zeros(4),d))
        for i in range(len(d[1:,0])):
            if int(d[i+1,0])==int(date.strftime("%Y%m%d")) and "%.1f"%(d[i+1,1])=="%.1f"%(lat) and "%.1f"%(d[i+1,2])=="%.1f"%(lon):
                return d[i+1,3]

    #daily gridded data
    url=str("https://acdisc.gesdisc.eosdis.nasa.gov/data/"
            +"Aura_OMI_Level3/OMNO2d.003/%s/"%(date.strftime("%Y")))
    try:
        u=urlopen(url)
    except:
        print_status(str("acdisc.gesdisc.eosdis.nasa.gov is not available."
                         +"No Internet? >>NO2 column == None"),
                         lvl=lvl,
                         style='fail')
        return None
    
    flist=[]
    for l in u.readlines():
        l=l.decode()
        f=l[l.find('href=')+6:l.find('href=')+54+6]
        if f[-3:]=='he5':
            flist.append(f)
    flist=np.unique(np.array(flist))
    nofile=True
    for f in flist:
        if f[19:23]+f[24:28] == date.strftime("%Y%m%d"):
            fname=f
            if os.path.exists(datapf+"no2/"+fname)==False:
                if sys.version_info[0]==3:
                    urlretrieve(url+fname,datapf+"no2/"+fname)
                else:
                    url_no2=urlopen(url+fname)
                    with open(datapf+"no2/"+fname,'w') as no2file:
                        no2file.write(url_no2.read())
            d=Dataset(datapf+"no2/"+fname,'r')
            c=d.groups['HDFEOS'].groups['GRIDS'].groups['ColumnAmountNO2'].groups['Data Fields'].variables['ColumnAmountNO2'][:,:]
            d.close()
            c=c*100**2 # molecules/cm2 -> molecules/m2
            lati=np.arange(-89.875,90.125,0.25)
            loni=np.arange(-179.875,180.125,0.25)
            c[c<0]=np.nan
            nofile=False
            break
    i=0
    while nofile and i<5:
        for f in flist:
            if int(f[19:23]+f[24:28]) == int(date.strftime("%Y%m%d"))+i+1:
                fname=f
                if os.path.exists(datapf+"no2/"+fname)==False:
                    if sys.version_info[0]==3:
                        urlretrieve(url+fname,datapf+"no2/"+fname)
                    else:
                        url_no2=urlopen(url+fname)
                        with open(datapf+"no2/"+fname,'w') as no2file:
                            no2file.write(url_no2.read())
                d=Dataset(datapf+"no2/"+fname,'r')
                c=d.groups['HDFEOS'].groups['GRIDS'].groups['ColumnAmountNO2'].groups['Data Fields'].variables['ColumnAmountNO2'][:,:]
                d.close()
                c=c*100**2 # molecules/cm2 -> molecules/m2
                lati=np.arange(-89.875,90.125,0.25)
                loni=np.arange(-179.875,180.125,0.25)
                c[c<0]=np.nan
                nofile=False
                break
        i+=1
    if nofile:
        return None
    lo,la=np.meshgrid(loni,lati)
    la=la[np.isnan(c)==False]
    lo=lo[np.isnan(c)==False]
    c1=c[np.isnan(c)==False]  
    
    co=griddata((lo.ravel(),la.ravel()),c1.ravel(),(lon,lat))    
    with open(datapf+"no2/no2.dat",'a') as txt:
        txt.write(date.strftime("%Y%m%d")+','+str(lat)+','+str(lon)+','+str(co)+'\n')
    return co






def _o3od(wvl,basepf,db_o3,o3DU=300.):
    """
    calculate o3 optical depth from serdyuchenko 2014 crosssection
    wvl-array()- wavelenghts [nm]
    cosz - array() - cosine of zenith angle
    o3DU - float - ozone column amount [DU] 1DU=2.687*10**20 molekules/m**2

    return tau- array(len(cosz),len(wvl) - optical depth
    """
    wvl=np.array(wvl)    
    o3cs=np.loadtxt(basepf+db_o3)
    o3cs[:,1]=gauss(o3cs[:,1],1000./(2.*np.sqrt(2.*np.log(2.))))
    wvls=o3cs[:,0]
    taus=(o3cs[:,1]/(100.**2.))*o3DU*2.687*10**20.
    taus=griddata(wvls,taus,wvl)
        
    taus[np.isnan(taus)]=0
    tau=taus

    return tau

def _get_o3col(date,datapf,lat,lon,lvl=0):
    if not os.path.exists(datapf+"o3/"):
        os.mkdir(datapf+"o3/")
    if os.path.exists(datapf+"o3/o3.dat"):
        d=np.loadtxt(datapf+"o3/o3.dat",delimiter=',')
        d=np.vstack((np.zeros(4),d))
        for i in range(len(d[1:,0])):
            if int(d[i+1,0])==int(date.strftime("%Y%m%d")) and "%.1f"%(d[i+1,1])=="%.1f"%(lat) and "%.1f"%(d[i+1,2])=="%.1f"%(lon):
                return d[i+1,3]        
    #daily gridded data
    url=str("https://acdisc.gesdisc.eosdis.nasa.gov/"
            +"data/Aura_OMI_Level3/OMTO3e.003/%s/"%(date.strftime("%Y")) )
    try:
        u=urlopen(url)
    except:
        print_status(str("acdisc.gesdisc.eosdis.nasa.gov is not available."
                         +"No Internet? >>O3 column == None"),
                         lvl=lvl,
                         style='fail')
        return None
    flist=[]
    for l in u.readlines():
        l=l.decode()
        f=l[l.find('href=')+6:l.find('href=')+54+6]
        if f[-3:]=='he5':
            flist.append(f)
    flist=np.unique(np.array(flist))
    nofile=True
    for f in flist:
        if f[19:23]+f[24:28] == date.strftime("%Y%m%d"):
            fname=f
            if os.path.exists(datapf+"o3/"+fname)==False:
                if sys.version_info[0]==3:
                    urlretrieve(url+fname,datapf+"o3/"+fname)
                else:
                    url_o3=urlopen(url+fname)
                    with open(datapf+"o3/"+fname,'w') as o3file:
                        o3file.write(url_o3.read())
            
            d=Dataset(datapf+'o3/'+fname,'r')
            c=d.groups['HDFEOS'].groups['GRIDS'].groups['OMI Column Amount O3'].groups['Data Fields'].variables['ColumnAmountO3'][:,:]
            d.close()
            lati=np.arange(-90,90,0.25)
            loni=np.arange(-180,180,0.25)
            c[c<0]=np.nan
            nofile=False
            break
    i=0
    while nofile and i<5:
        for f in flist:
            if int(f[19:23]+f[24:28]) == int(date.strftime("%Y%m%d"))+i+1:
                fname=f
                if os.path.exists(datapf+"o3/"+fname)==False:
#                        print(fname)
                    if sys.version_info[0]==3:
                        urlretrieve(url+fname,datapf+"o3/"+fname)
                    else:
                        url_o3=urlopen(url+fname)
                        with open(datapf+"o3/"+fname,'w') as o3file:
                            o3file.write(url_o3.read())
                d=Dataset(datapf+'o3/'+fname,'r')
                c=d.groups['HDFEOS'].groups['GRIDS'].groups['OMI Column Amount O3'].groups['Data Fields'].variables['ColumnAmountO3'][:,:]
                d.close()
                lati=np.arange(-90,90,0.25)
                loni=np.arange(-180,180,0.25)
                c[c<0]=np.nan
                nofile=False
                break  
        i+=1
    if nofile:
        return None
    lo,la=np.meshgrid(loni,lati)
    la=la[np.isnan(c)==False]
    lo=lo[np.isnan(c)==False]
    c1=c[np.isnan(c)==False]  
    
    co=griddata((lo.ravel(),la.ravel()),c1.ravel(),(lon,lat))   
    with open(datapf+"o3/o3.dat",'a') as txt:
        txt.write(date.strftime("%Y%m%d")+','+str(lat)+','+str(lon)+','+str(co)+'\n')
    return co

def _h2ood(wvl,cosz,I0,eI0,aod440,aod870,I940,w440,w870,w940):
    """
    calculate h2o optical depth 
    wvl - channels
    cosz - array() - cosine of zenith angle
    I0 940nm
    error I0 940nm
    aod440
    aod870
    I940
    w440
    w870
    w940
    return tau- array(len(cosz),len(channels)) - optical depth
    """        
    a=0.6131
    b=0.6712
    
    ang=-1.*np.log(aod440/aod870)/np.log(w440/w870)
    aod940=aod870*((w940/w870)**(-1.*ang))

    Taod=np.exp(-1.*aod940/cosz)
    T=I940/I0
    eT=I940*eI0/(I0**2)
    a_H2O=0.05480
    b_H2O=2.650   
    c_H2O=1.452 
    zen=np.arccos(cosz)*180./np.pi
    am= cosz+ a_H2O*((90.-zen+b_H2O)**(-c_H2O))# h2o airmass  according to Kasten 1965    Kasten (Arch. Met. Geoph. Bioklim. B, Bd. 14, II.2, 206, 1965)  
    am= 1./am

    w=np.exp((np.log(np.log(Taod/T))-np.log(a))/b)/am
    ew=np.abs(((a**(-1./b))*((np.log(Taod)-np.log(T))**((1./b)-1)))/(b*am*T))*eT
    h2o_od=np.zeros((len(cosz),len(wvl)))
    e_h2o_od=np.zeros((len(cosz),len(wvl)))
    h2o_od[:,-4]=0.0023*w-0.0002
    e_h2o_od[:,-4]=0.0023*ew
    h2o_od[:,-1]=0.0014*w-0.0003
    e_h2o_od[:,-1]=0.0014*ew
    return h2o_od,e_h2o_od


def _E_AOD(date,datapf,sza,I,I0,eI0,aod,rod,o3OD,no2OD,h2ood,E_ext,E_ali,E_noi,e_h2o_od,P,dP=5.):

    dI=np.sqrt((E_ali*I/(100.))**2+E_noi**2+(0.02*I)**2+(0.01*I)**2)#E_ali, E_noise, _E_cal, E_extra
    dOD=(np.cos(sza*np.pi/180.)*(dI/I).T).T**2
    
    A=np.sqrt(dOD)
    dI0,CZEN=np.meshgrid((1./I0)*eI0,np.cos(sza*np.pi/180.))

    dOD+=(dI0*CZEN)**2 #error propagated from I0 spectra
    
    A=np.hstack((A,dI0*CZEN))

    dOD+=(rod.T*dP/P).T**2
    A=np.hstack((A,(rod.T*dP/P).T))
    #A=(rod.T*dP/P).T

    dOD+=np.ones((len(dOD[:,0]),len(dOD[0,:])))*(no2OD*0.2)**2
    A=np.hstack((A,np.ones((len(dOD[:,0]),len(dOD[0,:])))*(no2OD*0.2)))

    dOD+=np.ones((len(dOD[:,0]),len(dOD[0,:])))*(o3OD*0.03)**2
    A=np.hstack((A,np.ones((len(dOD[:,0]),len(dOD[0,:])))*(o3OD*0.03)))

    rem=np.zeros((len(dOD[:,0]),len(I[0,:])))
    rem[:,-1]=(4.361*10**(-5))**2+(1.764*10**(-5))**2+(4.76*10**(-5))**2+e_h2o_od[:,-1]**2#dCO2,dCH4,DH20
    rem[:,-4]=(7.82*10**(-5))**2+e_h2o_od[:,-4]**2 #dH20
    dOD+=rem
    dOD=np.sqrt(dOD)
    A=np.hstack((A,np.sqrt(rem)))
    if not os.path.exists(datapf+'err3/'):
        os.mkdir(datapf+'err3/')
    np.save(datapf+'err3/Err'+date.strftime("%Y%m%d")+'.npy',A.data)
    
    return dOD  