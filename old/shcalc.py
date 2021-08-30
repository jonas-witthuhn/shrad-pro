#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 07:56:47 2018

@author: walther
"""
import os
os.environ['TZ']='UTC'
import gc

import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter, freqz
from scipy.optimize import minimize_scalar

import filehandling as fh
from helpers import print_status
import error
import get_toa
import sun.hsunpos as sp

from matplotlib import pyplot


def correctRAW(raw,
                INSdat,
                dangles,
                yaw_init=None,
                dbang="AngularResponse_GUV350_140129.csv",
                dbc3="C3lookup_",
                dbemotion="uncertainty/motion/motion_unc_",
                lvl=0,
                debug=False):
        
    def _get_k(wvls,zen,beta,dbc3):
        k=[]
        X,Y=np.meshgrid(np.arange(90),np.arange(90))
        X=X.flatten()
        Y=Y.flatten()
        for i,wvl in enumerate(wvls):
            C3=np.load(dbc3+"%d.npy"%(int(wvl)))
            vals=C3.flatten()
            beta[np.isnan(beta)==True]=-1
            C=griddata((X,Y),vals,(zen,beta))
            if len(k)==0:
                k= C
            else:
                k=np.vstack((k,C))
        return k
    
    def _calc_viewzen(roll,pitch,yaw,zen,azi,droll=0.,dpitch=0.,dyaw=0.):
        #calculate the angle between radiometer normal to sun position vektor
        c=np.pi/180.
        r=roll*c+droll*c
        p=pitch*c+dpitch*c
        y=yaw*c+dyaw*c
        z=zen*c
        a=azi*c
        g=a-y 
        coszen=np.sin(z)*np.sin(r)*np.sin(g)-np.sin(z)*np.sin(p)*np.cos(r)*np.cos(g)+np.cos(z)*np.cos(p)*np.cos(r)
        zenX=np.arccos(coszen)/c
        zenX[zenX>=89]=np.nan
        try:
            zenX=zenX.data
        except:
            pass
        return zenX # [degrees] angle between radiometer normal and solar vector
    gc.enable()
    print_status("Correct tilt and cosine response deviation.",lvl=lvl)
    ### load angular response data for each channel
    wvls=raw['channels']
    angdat=np.loadtxt(dbang,delimiter=',')
    angwvl=angdat[0,1:]
    angzen=angdat[1:,0]
    angcors=angdat[1:,1:]
    angcor=np.zeros((len(angzen),len(wvls)))
    for i in range(len(wvls)):
        if wvls[i]==0: #broadband
            angcor[:,i]=np.mean(angcors[:,:],axis=1)
        else:
            angcor[:,i]=angcors[:,np.argmin(np.abs(angwvl-wvls[i]))]
    
    ### define variables for tilt correction
    dpitch,droll,dyaw=dangles

    tim=raw['time'][:]
    azi=raw['azi'][:]
    zen=raw['zen'][:]
    roll=raw['roll'][:]
    pitch=raw['pitch'][:]
    if INSdat!=False:
        instim=INSdat['time'][:]
        roll=griddata(instim,INSdat['roll'][:],tim.data)
        pitch=griddata(instim,INSdat['pitch'][:],tim.data)
        yaw=griddata(instim,INSdat['yaw'][:],tim.data) 
        
        if debug:
            from make_plots import plot_angles_test
            plot_angles_test(instime=instim,
                             insroll=INSdat['roll'][:],
                             inspitch=INSdat['pitch'][:],
                             time=tim.data,
                             roll=roll,
                             pitch=pitch,
                             sdate=np.datetime64('2016-04-22T15:26:45'),
                             edate=np.datetime64('2016-04-22T15:29:15'))

        
        raw['roll'][:]=roll
        raw['pitch'][:]=pitch
        raw.update({'yaw':yaw})
    else:
        yaw=yaw_init
        
    ### apply both corrections with true zenith angles
    if INSdat!=False:
        #correct angular response
        beta=_calc_viewzen(roll,pitch,yaw,zen,azi,droll,dpitch,dyaw)
        

        
        angcors=griddata(angzen,angcor,beta)
        raw['rad'][:,:]=raw['rad'][:,:]/angcors[:,:]
        
        angerror=error.E_motion(wvls,zen,beta,dbemotion)
        #correct tilt
        k=_get_k(wvls,zen,beta,dbc3)
        if debug:
            np.save('/home/walther/Documents/test/nangcors.npy',angcors)
            np.save('/home/walther/Documents/test/nk.npy',k)
            np.save('/home/walther/Documents/test/nrad.npy',raw['rad'][:,:].data)
            np.save('/home/walther/Documents/test/nbeta.npy',beta)
            np.save('/home/walther/Documents/test/nroll.npy',roll)
            np.save('/home/walther/Documents/test/npitch.npy',pitch)
            np.save('/home/walther/Documents/test/nyaw.npy',yaw)
            np.save('/home/walther/Documents/test/ntim.npy',raw['time'].data)
            np.save('/home/walther/Documents/test/nzen.npy',zen.data)
            
        if len(wvls)>1:
            raw['rad']=raw['rad']*k.T
        else:
            raw['rad']=raw['rad']*k
    else:
        ### without any yaw given, we just assume GUVis is alignet to north 
        ### and not tilted
        if yaw==None and dyaw==None: 
            warnings.warn('No yaw information - assuming no tilt  -'
                          +" corrections might be dubious!")
            #correct angular response
            beta=_calc_viewzen(roll,pitch,0,azi,zen,azi,droll,dpitch,0)
            raw.update({'yaw':[np.NaN]})
            angcors=griddata(angzen,angcor,zen)
            #maximum correction factor radiometer - is tilted towards the sun
            angcors2=griddata(angzen,angcor,beta)
            drad=raw['rad'][:,:]/angcors2[:,:]
            raw['rad'][:,:]=raw['rad'][:,:]/angcors[:,:]
            angerror=np.abs(drad-raw['rad'][:,:])*100./raw['rad'][:,:]#[%]
            angerror+=error.E_motion(wvls,zen,beta,dbemotion)
        ### fixed yaw is given ('yaw_init' and or dyaw) so we calculate the 
        ### corrections. -This is the option for landside stations
        else:
            print_status("Calculate corrections with fixed yaw angle",lvl)
            yaw=np.ones(len(tim))*yaw
            raw.update({'yaw':yaw}) 
            beta=_calc_viewzen(roll,pitch,yaw,zen,azi,droll,dpitch,dyaw)
            angcors=griddata(angzen,angcor,beta)
            raw['rad'][:,:]=raw['rad'][:,:]/angcors[:,:]
            angerror=error.E_motion(wvls,zen,beta,dbemotion)
            #correct tilt
            k=_get_k(wvls,zen,beta,dbc3)
            if len(wvls)>1:
                raw['rad']=raw['rad']*k.T
            else:
                raw['rad']=raw['rad']*k
    raw.update({'E_ali':angerror})

    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return raw



def processData(raw,lbasepf,Eext,lresponse,lvl=0,plot=False):
    gc.enable()
    print_status("Processing radiation data.",lvl)

    channels=raw['channels']
    date=np.datetime64('1970-01-01')+np.median(raw['time']).astype('timedelta64[s]')
    #delete uncomplete sweeps by define start and end of data with a zero or home position
    mode=raw['shmode'][:]
    
    index=np.zeros(len(mode))
    check1=True
    check2=True
    for i in range(len(mode)):
        m1=mode[i].decode()
        m2=mode[-1*i-1].decode()
        if check1:
            if m1=='Z' or m1=='P':
                check1=False
            else:
                index[i]=1
        if check2:
            if m2=='Z' or m2=='P':
                check2=False
            else:
                index[-1*i-1]=1
        if check1==False and check2==False:
            break
            
    mode=raw['shmode'][index==0]

    gtime=[]    #mean time of global irradiance measurements
    grad=[]     #mean global irradiance
    gstd=[]     #standard deviation of global irradiance measurement
    dtime=[]    #time of maximum shading
    mins=[]     #minimum irradiance of each sweep
    minsstd=[]  #standard deviation of each minimum due to averaging
    diffs=[]    #extrapolated shadet part of the diffuse irradiance
    es=[]       #extrapolation uncertainty due to noisy data
    
    tim=raw['time'][index==0]
    rad=raw['rad'][index==0,:]
#    smode=mode[0].decode()
    smode='S'
    starti=0
    for i in range(len(mode)):
        m=mode[i].decode()
        if smode!=m:
            # shadowband in park or home position -> measuring global irradiance
            if smode=='P' or smode=='Z' or smode=='S': 
                gtime.append(np.mean(tim[starti:i]))
                gradtmp=np.mean(rad[starti:i,:],axis=0)
                gstdtmp=np.std(rad[starti:i,:],axis=0)
                if len(grad)==0:
                    grad=gradtmp
                    gstd=gstdtmp
                else:
                    grad=np.vstack((grad,gradtmp))
                    gstd=np.vstack((gstd,gstdtmp))                        
            else: # analyze sweep data
                data=rad[starti:i,:] #radiation data of one sweep
                datatime=tim[starti:i]
                mintim,avgmin,avgerrmin,ic_mean,e_mean=_sweepAnalyzer3(data,
                                                                      datatime,
                                                                      raw['channels_SNR'],
                                                                      plot=plot)
                
                if not np.isnan(mintim):
                    # if plot:
                    #     plot_date=pd.to_datetime(np.datetime64("1970-01-01")+np.timedelta64(int(np.mean(mintim)),'s'))
                    #     if not os.path.exists("processed_sweeps/%s"%(plot_date.strftime("%Y%m%d"))):
                    #         os.mkdir("processed_sweeps/%s"%(plot_date.strftime("%Y%m%d")))
                    #     pyplot.savefig("processed_sweeps/%s/%s.png"%(plot_date.strftime("%Y%m%d"),plot_date.strftime("%Y%m%d_%H%M%S")))
                    #     pyplot.close()
                    dtime.append(mintim)
                    if len(mins)==0:
                        mins=avgmin
                        minsstd=avgerrmin
                        diffs=ic_mean
                        es=e_mean
                    else:
                        mins=np.vstack((mins,avgmin))
                        minsstd=np.vstack((minsstd,avgerrmin))
                        diffs=np.vstack((diffs,ic_mean))
                        es=np.vstack((es,e_mean))
            starti=i
            smode=m
    if len(mins)==0:
        return None
    #interpolate global irradiance to time of minimum of each sweep
    Iglo=np.zeros((len(dtime),len(channels)))
    Iglostd=np.zeros((len(dtime),len(channels)))
    for i in range(len(channels)):
        Iglo[:,i]=griddata(gtime,grad[:,i],dtime)
        Iglostd[:,i]=griddata(gtime,gstd[:,i],dtime)
    Idir_h=diffs-mins
    Idif=Iglo-Idir_h
    zen=raw['zen'][index==0]
    zen=griddata(tim,zen,dtime)
    coszen=np.cos(zen*np.pi/180.)
    Idir=np.zeros((len(dtime),len(channels)))
    for i in range(len(channels)):
        Idir[:,i]=Idir_h[:,i]/coszen
        
    azi=raw['azi'][index==0]
    azi=griddata(tim,azi,dtime)
    lat=raw['lat'][index==0]
    lat=griddata(tim,lat,dtime)
    lon=raw['lon'][index==0]
    lon=griddata(tim,lon,dtime)
    shangle=raw['shangle'][index==0]
    shangle=griddata(tim,shangle,dtime)
    T_s=raw['T_s'][index==0]
    T_s=griddata(tim,T_s,dtime)
    yaw=raw['yaw'][index==0]
    yaw=griddata(tim,yaw,dtime)
    alignerror=raw['E_ali'][index==0]
    alignerror=griddata(tim,alignerror,dtime)


    extdat=np.loadtxt(lbasepf+Eext)
    AOD875=extdat[1:,0]
    ZENS=extdat[0,1:]
    ERRORS=extdat[1:,1:]
    
    I0,_=get_toa.get_I0(date,
                      [channels[12]],
                      [raw['channels_cwvl'][12]],
                      basepf=lbasepf,
                      dbresponse=lresponse,
                      assume='close',
                      lvl=lvl+1)
    I0=np.float(I0)
    #eI0=self.eI0[12]
    
    AODS=-1.*np.log(Idir[:,12]/I0)
    AODS=(AODS.T*coszen).T
    E_ext=np.zeros(len(AODS))
    for j,ao in enumerate(AODS):
        tmp=1000
        ind=-99
        for i,a in enumerate(AOD875):
            if np.isnan(a):
                break
            if abs(a-ao)<tmp:
                ind=i
                tmp=abs(a-ao)
        if ind==-99:
            E_ext[j]=99999.
        else:
            E_ext[j]=griddata(ZENS,ERRORS[ind,:],zen[j]) 
    
    prodat={'time':np.array(dtime),
            'Iglo':Iglo, 
            'Idir':Idir, 
            'Idif':Idif,
            'zen':zen,
            'Iglo_std':Iglostd,
            'Idir_std':minsstd,
            'channels':channels,
            'channels_cwvl':raw['channels_cwvl'],
            'calibF':raw['calibF'], 
            'lat':lat,
            'lon':lon,
            'azi':azi,
            'shangle':shangle,
            'T_s':T_s,
            'yaw':yaw,
            'E_ali':alignerror,
            'E_ext':E_ext,
            'E_noise':es}
    print_status("...done!",lvl,style='g')
    print_status("",lvl-1)
    return prodat
    
    
def _sweepAnalyzer(data,tim,SNR,minr=0.05,maxd=0.004,mdc=0.01,mdc2=0.1,cis=3,dr=5,rr=30,ch=12,plot=False):#min_requirement,max_derivative,max_derivative_contact, max_derivative_contact2,contact_index_security,derivative_range,regression_range
    gc.enable()
    def _line(x,a,b):
            return x*a+b
        
    if type(data)==np.ma.MaskedArray:
        mask=data.mask
        data=data.data
        data[mask]=np.nan
    if type(tim)==np.ma.MaskedArray:
        mask=tim.mask
        tim=tim.data
        tim[mask]=np.nan
    
    CH=len(data[0,:])
    if len(data[:,0])<10:
        return np.nan,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH
    noise=data/SNR
    
    if plot:
        from matplotlib import pyplot
#        import matplotlib
#        cmap = matplotlib.cm.get_cmap('nipy_spectral')
        pyplot.figure(figsize=(10,8),dpi=200)
        SCALE=0.1
#        SCALE=np.zeros(CH-1)
#        for i in range(CH-1):
#            SCALE[i]=1./np.nanmax(data[:,i])
#            pyplot.plot(tim,data[:,i]*SCALE[i]+i*0.3,color=cmap(float(i)/float(CH-2)))
#        
#        pyplot.yticks(np.arange(CH-1)*0.3,['305','340','380','412','443','510','610','625','665','694','750','765','875','940','1020','1245','1550','1640'])
        pyplot.plot(tim,data[:,ch]+SCALE,'#a6bddb',label='sweep irradiance v1')
        pyplot.grid(True)
    
    
    #determine minimum and maximum of the sweep
    minimum=np.nanmin(data,axis=0)
    maximum=np.nanmax(data,axis=0)
    
    #calculate minimum criterion and minimum data
    mc=minimum[ch]+(maximum[ch]-minimum[ch])*minr
    derdat1=data[1:,ch]
    derdat2=data[:-1,ch]
    
    data=data[:-1,:]
    tim=tim[:-1]
    noise=noise[:-1,:]
    
    index=np.zeros(len(derdat2))
    index[derdat2>=mc]=1
    index[derdat1==0]=1
    index[derdat2==0]=1
    der=(derdat1-derdat2)/(0.5*(derdat1+derdat2))
    index[der<=-1.*maxd]=1
    index[der>=maxd]=1
    mindat=data[index==0,:]
    mintimd=tim[index==0]
    if len(mindat[:,0])>0:
        avgmin=np.nanmean(mindat,axis=0)
        avgerrmin=np.nanstd(mindat,axis=0)
        mintim=np.nanmean(tim[index==0])
    else:
        avgmin=[np.nan]*CH
        avgerrmin=[np.nan]*CH
        mintim=np.mean(tim)
        ic_mean=[np.nan]*CH
        e_mean=[np.nan]*CH
        return mintim,avgmin,avgerrmin,ic_mean,e_mean
    
    if plot:
#        for i in range(CH-1):
#            pyplot.plot(mintimd,mindata[:,i]*SCALE[i]+i*0.3,'r.')
        pyplot.plot(mintimd,mindat[:,ch]+SCALE,color='#fdbb84',marker='.',label='minimum data v1')
    
    #choose data before and after minimumdata
    minmaxtim=np.nanmax(tim[index==0])
    minmintim=np.nanmin(tim[index==0])
    index=np.zeros(len(tim))
    index[tim>minmaxtim]=1
    dataafter=data[index==1,:]
    timafter=tim[index==1]
    noiseafter=noise[index==1,:]
    index=np.zeros(len(tim))
    index[tim<minmintim]=1
    databefore=data[index==1,:]
    timbefore=tim[index==1]
    noisebefore=noise[index==1,:]
    
    #look for contact points of shade before and after minimum data
    derdat1=dataafter[:-2*dr,ch]
    derdat2=dataafter[2*dr:,ch]
    index=np.zeros(len(derdat1))
    index[derdat1+derdat2==0]=1
    der=(derdat2-derdat1)/(0.5*(derdat1+derdat2))
    der[index==1]=0
    der=np.array([0]*(2*dr)+list(der)+[0]*(2*dr))
    flag=False
    contact=False
    for i in range(len(der)):
        d=der[i]
        if d>mdc2:
            flag=True
        if flag==True  and d<mdc:
            contact=i+cis
            break
    if contact!=False:
        try:
            regdataafter=dataafter[contact:contact+rr+1,:]
            regtimafter=timafter[contact:contact+rr+1]
            regnoiseafter=noiseafter[contact:contact+rr+1,:]
        except:
            regdataafter=dataafter[contact:,:]
            regtimafter=timafter[contact:]
            regnoiseafter=noiseafter[contact:,:]
        afterflag=True
    else:
        regdataafter=np.nan
        regtimafter=np.nan
        regnoiseafter=np.nan
        afterflag=False
        
    if plot and afterflag:
        pyplot.plot(regtimafter,regdataafter[:,ch]+SCALE,color='#addd8e',marker='.',linestyle='',label='interpolation data (after) v1')
        pyplot.plot(timafter[contact],dataafter[contact,ch]+SCALE,color='#addd8e',marker='x',markersize=10,label='contact_point (after) v1')

        
        

    derdat1=databefore[:-2*dr,ch]
    derdat2=databefore[2*dr:,ch]    
    index=np.zeros(len(derdat1))
    index[derdat1+derdat2==0]=1
    der=(derdat2-derdat1)/(0.5*(derdat1+derdat2))
    der[index==1]=0
    der=np.array([0]*(2*dr)+list(der)+[0]*(2*dr))
    flag=False
    contact=False
    for i in range(len(der)-1,-1,-1):
        d=der[i]
        if -1.*d>mdc2:
            flag=True
        if flag==True and -1.*d<mdc:
            contact=i-cis
            break
    if contact!=False:
        try:
            regdatabefore=databefore[contact-rr:contact+1,:]
            regtimbefore=timbefore[contact-rr:contact+1]
            regnoisebefore=noisebefore[contact-rr:contact+1,:]
        except:
            regdatabefore=databefore[:contact+1,:]
            regtimbefore=timbefore[:contact+1]
            regnoisebefore=noisebefore[:contact+1,:]
        beforeflag=True
    else:
        regdatabefore=np.nan
        regtimbefore=np.nan       
        regnoisebefore=np.nan
        beforeflag=False
        
    if plot and beforeflag:
        pyplot.plot(regtimbefore,regdatabefore[:,ch]+SCALE,color='#c994c7',marker='.',linestyle='',label='interpolation data (before) v1')
        pyplot.plot(timbefore[contact],databefore[contact,ch]+SCALE,color='#c994c7',marker='x',markersize=10,label='contact_point (before) v1')

    
    #interpolate shaded diffuse part
    if beforeflag and len(regdatabefore[:,0])>3:
        slope=np.zeros(CH)
        intercept=np.zeros(CH)
        ic_before=np.zeros(CH)
        e_before=np.zeros(CH)
        stim=np.nanmin(regtimbefore)
        for i in range(CH):
            popt,pconv=curve_fit(_line,regtimbefore-stim,regdatabefore[:,i],sigma=regnoisebefore[:,i])
            slope[i],intercept[i]=popt
            ic_before[i]=np.nanmean(slope[i]*(mintimd-stim)+intercept[i])
            if np.array(pconv).size!=4:
                e_before[i]=np.nan
            else:
                e_before[i]=np.nanmean(abs(mintimd-stim)*(pconv[0,0]**0.5)+(pconv[1,1]**0.5))
        if ic_before[ch]<avgmin[ch]:
            beforeflag=False
    else:
        beforeflag=False
    if plot and beforeflag:
        pyplot.plot(tim,((tim-stim)*slope[ch]+intercept[ch])+SCALE,color='#c994c7',linestyle=':',label='linear regression (before) v1')

        
    if afterflag and len(regdataafter[:,0])>3:
        slope=np.zeros(CH)
        intercept=np.zeros(CH)
        ic_after=np.zeros(CH)
        e_after=np.zeros(CH)
        stim=np.nanmin(regtimafter)
        for i in range(CH):
            popt,pconv=curve_fit(_line,regtimafter-stim,regdataafter[:,i],sigma=regnoiseafter[:,i])
            slope[i],intercept[i]=popt
            ic_after[i]=np.nanmean(slope[i]*(mintimd-stim)+intercept[i])
            if np.array(pconv).size!=4:
                e_after[i]=np.nan
            else:
                e_after[i]=np.nanmean(abs(mintimd-stim)*(pconv[0,0]**0.5)+(pconv[1,1]**0.5))
        if ic_after[ch]<avgmin[ch]:
            afterflag=False
    else:
        afterflag=False

    if plot and afterflag:
        pyplot.plot(tim,((tim-stim)*slope[ch]+intercept[ch])+SCALE,color='#addd8e',linestyle=':',label='linear regression (after) v1')


        
    if afterflag and beforeflag:
        ic_mean=(ic_before+ic_after)*0.5
        e_mean=(e_before+e_after)*0.5
    elif afterflag and not beforeflag:
        ic_mean=ic_after
        e_mean=e_after
    elif not afterflag and beforeflag:
        ic_mean=ic_before
        e_mean=e_before
    else:
        ic_mean=[np.nan]*CH
        e_mean=[np.nan]*CH
    return mintim,avgmin,avgerrmin,ic_mean,e_mean

def _sweepAnalyzer2(data,tim,SNR,minr=0.05,maxd=0.004,mdc=0.01,mdc2=0.1,cis=3,dr=1,rr=240,ch=12):#min_requirement,max_derivative,max_derivative_contact, max_derivative_contact2,contact_index_security,derivative_range,regression_range
    gc.enable()
    def _line(x,a,b):
            return x*a+b
#    def _gaus(x,s):
#        xdata=x[0]-np.mean(x[0])
#        sig=x[1]
#        c=x[2]
#        p=0.9
#        win=np.exp(-0.5*np.abs(xdata/sig)**(2.*p))
#        return c-win/s
    
    def _gaus(x,s):
        xdata=x[0]-x[1]
        sig=x[2]
        adata=x[3]
        p=0.8
        win=np.exp(-0.5*np.abs(xdata/sig)**(2.*p))
        return adata-win/s
        
    if type(data)==np.ma.MaskedArray:
        mask=data.mask
        data=data.data
        data[mask]=np.nan
    if type(tim)==np.ma.MaskedArray:
        mask=tim.mask
        tim=tim.data
        tim[mask]=np.nan
        
    
    CH=len(data[0,:])
    if len(data[:,0])<10:
        return np.nan,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH
    
    from matplotlib import pyplot
    pyplot.figure()
    pch=5
    pyplot.plot(tim,data[:,pch],'b')
    
    
    noise=data/SNR
    #determine minimum and maximum of the sweep
    minimum=np.nanmin(data,axis=0)
    maximum=np.nanmax(data,axis=0)
    
    #calculate minimum criterion and minimum data
    mc=minimum[ch]+(maximum[ch]-minimum[ch])*minr
    derdat1=data[1:,ch]
    derdat2=data[:-1,ch]
    
    data=data[:-1,:]
    tim=tim[:-1]
    noise=noise[:-1,:]
    
    index=np.zeros(len(derdat2))
    index[derdat2>=mc]=1
    index[derdat1==0]=1
    index[derdat2==0]=1
    der=(derdat1-derdat2)/(0.5*(derdat1+derdat2))
    index[der<=-1.*maxd]=1
    index[der>=maxd]=1
    mindat=data[index==0,:]
    mintimd=tim[index==0]
    if len(mindat[:,0])>0:
        avgmin=np.nanmean(mindat,axis=0)
        avgerrmin=np.nanstd(mindat,axis=0)
        mintim=np.nanmean(tim[index==0])
    else:
        avgmin=[np.nan]*CH
        avgerrmin=[np.nan]*CH
        mintim=np.mean(tim)
        ic_mean=[np.nan]*CH
        e_mean=[np.nan]*CH
        return mintim,avgmin,avgerrmin,ic_mean,e_mean

    pyplot.plot(mintimd,mindat[:,pch],'r.')

       
    #choose data before and after minimumdata
    minmaxtim=np.nanmax(tim[index==0])
    minmintim=np.nanmin(tim[index==0])
    index=np.zeros(len(tim))
    index[tim>minmaxtim]=1
    dataafter=data[index==1,:]
    timafter=tim[index==1]
    noiseafter=noise[index==1,:]
    index=np.zeros(len(tim))
    index[tim<minmintim]=1
    databefore=data[index==1,:]
    timbefore=tim[index==1]
    noisebefore=noise[index==1,:]
    
    #look for contact points of shade before and after minimum data
    derdat1=dataafter[:-2*dr,ch]
    derdat2=dataafter[2*dr:,ch]
    index=np.zeros(len(derdat1))
    index[derdat1+derdat2==0]=1
    der=(derdat2-derdat1)/(0.5*(derdat1+derdat2))
    der[index==1]=0
    der=np.array([0]*(2*dr)+list(der)+[0]*(2*dr))
    flag=False
    contact=False
    for i in range(len(der)):
        d=der[i]
        if d>mdc2:
            flag=True
        if flag==True  and d<mdc:
            contact=i+cis
            break
    if contact!=False:
        try:
            regdataafter=dataafter[contact:contact+rr+1,:]
            regtimafter=timafter[contact:contact+rr+1]
            regnoiseafter=noiseafter[contact:contact+rr+1,:]
        except:
            regdataafter=dataafter[contact:,:]
            regtimafter=timafter[contact:]
            regnoiseafter=noiseafter[contact:,:]
        afterflag=True
    else:
        regdataafter=np.nan
        regtimafter=np.nan
        regnoiseafter=np.nan
        afterflag=False

    if contact!=False:
        pyplot.plot(regtimafter,regdataafter[:,pch],'g.')



    derdat1=databefore[:-2*dr,ch]
    derdat2=databefore[2*dr:,ch]    
    index=np.zeros(len(derdat1))
    index[derdat1+derdat2==0]=1
    der=(derdat2-derdat1)/(0.5*(derdat1+derdat2))
    der[index==1]=0
    der=np.array([0]*(2*dr)+list(der)+[0]*(2*dr))
    flag=False
    contact=False
    for i in range(len(der)-1,-1,-1):
        d=der[i]
        if -1.*d>mdc2:
            flag=True
        if flag==True and -1.*d<mdc:
            contact=i-cis
            break
    if contact!=False:
        try:
            regdatabefore=databefore[contact-rr:contact+1,:]
            regtimbefore=timbefore[contact-rr:contact+1]
            regnoisebefore=noisebefore[contact-rr:contact+1,:]
        except:
            regdatabefore=databefore[:contact+1,:]
            regtimbefore=timbefore[:contact+1]
            regnoisebefore=noisebefore[:contact+1,:]
        beforeflag=True
    else:
        regdatabefore=np.nan
        regtimbefore=np.nan       
        regnoisebefore=np.nan
        beforeflag=False
    
    if contact!=False:
        pyplot.plot(regtimbefore,regdatabefore[:,pch],'b.')
    
    #interpolate shaded diffuse part
    if beforeflag and len(regdatabefore[:,0])>3:
        slope=np.zeros(CH)
        intercept=np.zeros(CH)
        ic_before=np.zeros(CH)
        e_before=np.zeros(CH)
        stim=np.nanmin(regtimbefore)
        for i in range(CH):
            popt,pconv=curve_fit(_line,regtimbefore-stim,regdatabefore[:,i],sigma=regnoisebefore[:,i])
            slope[i],intercept[i]=popt
            ic_before[i]=np.nanmean(slope[i]*(mintimd-stim)+intercept[i])
            if np.array(pconv).size!=4:
                e_before[i]=np.nan
            else:
                e_before[i]=np.nanmean(abs(mintimd-stim)*(pconv[0,0]**0.5)+(pconv[1,1]**0.5))
        if ic_before[ch]<avgmin[ch]:
            beforeflag=False
        pyplot.plot(tim,(tim-stim)*slope[pch]+intercept[pch],'b:')
    else:
        beforeflag=False
        
    if afterflag and len(regdataafter[:,0])>3:
        slope=np.zeros(CH)
        intercept=np.zeros(CH)
        ic_after=np.zeros(CH)
        e_after=np.zeros(CH)
        stim=np.nanmin(regtimafter)
        for i in range(CH):
            popt,pconv=curve_fit(_line,regtimafter-stim,regdataafter[:,i],sigma=regnoiseafter[:,i])
            slope[i],intercept[i]=popt
            ic_after[i]=np.nanmean(slope[i]*(mintimd-stim)+intercept[i])
            if np.array(pconv).size!=4:
                e_after[i]=np.nan
            else:
                e_after[i]=np.nanmean(abs(mintimd-stim)*(pconv[0,0]**0.5)+(pconv[1,1]**0.5))
        if ic_after[ch]<avgmin[ch]:
            afterflag=False
        pyplot.plot(tim,(tim-stim)*slope[pch]+intercept[pch],'g:')
    else:
        afterflag=False

    if beforeflag and afterflag and len(regdataafter[:,0])>3 and len(regdatabefore[:,0])>3:
        otims=np.array(list(regtimbefore)+list(regtimafter))
        otimd=np.arange(otims[0],otims[-1])
        scaler=np.zeros(CH)
        ic_mean=np.zeros(CH)
        e_mean=np.zeros(CH)
        for i in range(CH):
#            if len(regdatabefore[:-30,i])>10:
#            
            popt,pconv=curve_fit(_line,regtimbefore-mintim,regdatabefore[:,i],sigma=regnoisebefore[:,i])#[:-30]
            slope,inter=popt
            adata=list(slope*(regtimbefore-mintim)+inter)
            popt,pconv=curve_fit(_line,regtimafter-mintim,regdataafter[:,i],sigma=regnoiseafter[:,i])
            slope,inter=popt
            adata=np.array(adata+list(slope*(regtimafter-mintim)+inter))
            
            odata=np.array(list(regdatabefore[:,i])+list(regdataafter[:,i]))
            cmin=1./(0.2*np.abs(np.mean(odata)-avgmin[i]))
            noises=np.array(list(regnoisebefore[:,i])+list(regnoiseafter[:,i]))
            So=float(len(mindat[:,i]))/(2.*np.sqrt(2.*np.log(2.)))
#            Co=np.max(odata)
            popt,pcov=curve_fit(_gaus,[otims,mintim,So,adata],odata,sigma=noises,p0=cmin*4)
#            popt,pcov=curve_fit(_gaus,[otims,So,Co],odata,sigma=noises,p0=cmin*4)#,bounds=(cmin,np.inf))        
            scaler[i]=popt
            
            popt,pconv=curve_fit(_line,regtimbefore-mintim,regdatabefore[:,i],sigma=regnoisebefore[:,i])
            slope,inter=popt
            adata=list(np.array(slope*(otimd-mintim)+inter)[otimd<mintim])
            popt,pconv=curve_fit(_line,regtimafter-mintim,regdataafter[:,i],sigma=regnoiseafter[:,i])
            slope,inter=popt
            adata=np.array(adata+list(np.array(slope*(otimd-mintim)+inter)[otimd>=mintim]))
            ic_mean[i]=np.min(_gaus([otimd,mintim,So,adata],scaler[i]))
            e_mean[i]=np.sqrt(np.diag(pcov))
            if i==pch:
                pyplot.plot(otimd,_gaus([otimd,mintim,So,adata],scaler[i]),'m:')
    else:
        ic_mean=[np.nan]*CH
        e_mean=[np.nan]*CH
      
#    if afterflag and beforeflag:
#        ic_mean=(ic_before+ic_after)*0.5
#        e_mean=(e_before+e_after)*0.5
#    elif afterflag and not beforeflag:
#        ic_mean=ic_after
#        e_mean=e_after
#    elif not afterflag and beforeflag:
#        ic_mean=ic_before
#        e_mean=e_before
#    else:
#        ic_mean=[np.nan]*CH
#        e_mean=[np.nan]*CH
        
        
    pyplot.grid(True)
    date=np.datetime64('1970-01-01')+np.array([mintim]).astype('timedelta64[s]')
    date=pd.to_datetime(date)[0]
#    print date
    if not os.path.exists("testfig/%s"%(date.strftime("%Y%m%d/"))):
        os.mkdir("testfig/%s"%(date.strftime("%Y%m%d/")))
    pyplot.savefig("testfig/%s"%(date.strftime("%Y%m%d/%Y%m%d-%H%M%S.png")))
    pyplot.close()
    return mintim,avgmin,avgerrmin,ic_mean,e_mean
#data,tim,SNR,minr=0.05,maxd=0.004,mdc=0.01,mdc2=0.1,cis=3,dr=1,rr=240,ch=12
    
def _sweepAnalyzer3(data,tim,SNR,Cmin=0.05,Cder=0.01,Mdc=0.05,dr=1,rr=30,ch=12,plot=False):#dr=1 rr=30  rr=40 dr=5
    gc.enable()
    def _line(x,a,b):
            return x*a+b
        
    if type(data)==np.ma.MaskedArray:
        mask=data.mask
        data=data.data
        data[mask]=np.nan
    if type(tim)==np.ma.MaskedArray:
        mask=tim.mask
        tim=tim.data
        tim[mask]=np.nan      
        
    CH=len(data[0,:])  
    
    ind=~(np.isnan(tim)+np.isnan(data[:,ch]))
    data=data[ind,:]
    tim=tim[ind]

     ### filter brocken sweeps
    if data.shape[0]<10:
        return np.nan,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH  

    
    if plot:
        from matplotlib import pyplot
#        import matplotlib
#        cmap = matplotlib.cm.get_cmap('nipy_spectral')
       # pyplot.figure()
#        SCALE=np.zeros(CH-1)
#        for i in range(CH-1):
#            SCALE[i]=1./np.nanmax(data[:,i])
#            pyplot.plot(tim,data[:,i]*SCALE[i]+i*0.3,color=cmap(float(i)/float(CH-2)))
#        
#        pyplot.yticks(np.arange(CH-1)*0.3,['305','340','380','412','443','510','610','625','665','694','750','765','875','940','1020','1245','1550','1640'])
        pyplot.plot(tim,data[:,ch],'#2b8cbe',label='sweep irradiance')
        pyplot.grid(True)
    
    
    noise=data/SNR

    #######################
    ### find minimum data
    #######################
    mini=np.nanmin(data[:,ch],axis=0)
    maxi=np.nanmax(data[:,ch],axis=0)
    #minimum criterion
    mc=mini+(maxi-mini)*Cmin
    #derivative
    der=np.abs((data[1:,ch]-data[:-1,ch])/(0.5*(data[1:,ch]+data[:-1,ch])))
    der=griddata(tim[:-1]+0.5*np.mean(tim[1:]-tim[:-1]),der,tim,fill_value=np.inf)
    ind=(data[:,ch]<mc)*(der<Cder)
    
#    if plot:
#        pyplot.plot(tim,der,'r')
#        print mc
    
    mindata=data[ind,:]
    mintimd=tim[ind]
    ### filter if less data else calculate mean/std/err
    if len(mintimd)>1 and np.count_nonzero(np.isnan(mindata[:,ch]))<2:
        MIN_AVG=np.nanmedian(mindata,axis=0)
        MIN_ERR=np.nanstd(mindata,axis=0)
        MIN_TIM=np.nanmean(mintimd)
    else:
        return np.nan,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH
    
    if plot:
#        for i in range(CH-1):
#            pyplot.plot(mintimd,mindata[:,i]*SCALE[i]+i*0.3,'r.')
        pyplot.plot(mintimd,mindata[:,ch],color='#e34a33',marker='.',label='minimum data')
    
    ########################
    ### find data before and after shading of diffuser
    ########################
    
    
    #########################
    ### data after
    #########################
    ind=tim>np.nanmax(mintimd)
    if np.count_nonzero(ind)<rr*0.7:
        return np.nan,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH
        
    #derivative
    der=np.abs((data[ind,ch][1:]-data[ind,ch][:-1])/(0.5*(data[ind,ch][1:]+data[ind,ch][:-1])))
    der=griddata(tim[ind][:-1]+0.5*np.mean(tim[ind][1:]-tim[ind][:-1]),der,tim[ind],fill_value=-np.inf)
    
#    if plot :
#        pyplot.plot(tim[ind],der,'g:')

    ## contact point
    contact=False
    mder=np.argmax(der)
    for i in range(len(der)):
        if i<mder:
            continue
        if der[i]<Mdc*der[mder]:
            contact=i
            break
    if contact:
        AFTER_flag=True
        AFTER_data=data[ind,:][contact:,:]
        AFTER_noise=noise[ind,:][contact:,:]
        AFTER_tim=tim[ind][contact:]
        try:
            AFTER_Idata=AFTER_data[dr:dr+rr,:]
            AFTER_Itim=AFTER_tim[dr:dr+rr]
            AFTER_Inoise=AFTER_noise[dr:dr+rr,:]
        except:
            AFTER_Idata=AFTER_data[dr:,:]
            AFTER_Itim=AFTER_tim[dr:]
            AFTER_Inoise=AFTER_noise[dr:,:]
        if len(AFTER_Itim)>=rr*0.7: 
            if plot:
                pyplot.plot(AFTER_Itim,AFTER_Idata[:,ch],color='r',marker='.',linestyle='')

            #### filter variations
            popt,pconv=curve_fit(_line,AFTER_Itim-tim[0],AFTER_Idata[:,ch],sigma=AFTER_Inoise[:,ch])
            STD=np.std(AFTER_Idata[:,ch]-(AFTER_Itim-tim[0])*popt[0]-popt[1])
            while STD>0.002:
                di=np.argmax(np.abs(AFTER_Idata[:,ch]-(AFTER_Itim-tim[0])*popt[0]-popt[1]))
                AFTER_Idata=np.delete(AFTER_Idata,di,axis=0)
                AFTER_Itim=np.delete(AFTER_Itim,di)
                AFTER_Inoise=np.delete(AFTER_Inoise,di,axis=0)
                if len(AFTER_tim)<rr*0.7:
                    break
                popt,pconv=curve_fit(_line,AFTER_Itim-tim[0],AFTER_Idata[:,ch],sigma=AFTER_Inoise[:,ch])
                STD=np.std(AFTER_Idata[:,ch]-(AFTER_Itim-tim[0])*popt[0]-popt[1])
        
            DF=np.nanmin(AFTER_data[:3,ch])-(popt[0]*(AFTER_tim[0]-tim[0])+popt[1])
            if len(AFTER_Itim)<rr*0.7 or DF>0.001:
                AFTER_flag=False
        else: 
            AFTER_flag=False
        if plot and AFTER_flag:
            pyplot.plot(AFTER_Itim,AFTER_Idata[:,ch],color='#31a354',marker='.',linestyle='',label='interpolation data (after)')
            pyplot.plot(AFTER_tim[0],AFTER_data[0,ch],color='#31a354',marker='x',markersize=10,label='contact_point (after)')
#            for i in range(CH-1):
#                pyplot.plot(AFTER_Itim,AFTER_Idata[:,i]*SCALE[i]+i*0.3,'g.')
#                pyplot.plot(AFTER_tim[0],AFTER_data[0,i]*SCALE[i]+i*0.3,'g.')
    else:
        AFTER_flag=False

    
    #########################
    ### data before
    #########################
    ind=tim<np.nanmin(mintimd)
    if np.count_nonzero(ind)<rr*0.7:
        return np.nan,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH,[np.nan]*CH
    #derivative
    der=np.abs((data[ind,ch][:-1][::-1]-data[ind,ch][1:][::-1])/(0.5*(data[ind,ch][1:][::-1]+data[ind,ch][:-1][::-1])))
    der=griddata(tim[ind][1:][::-1]+0.5*np.mean(tim[ind][:-1][::-1]-tim[ind][1:][::-1]),der,tim[ind][::-1],fill_value=-np.inf)
#    if plot :
#        pyplot.plot(tim[ind][::-1],der,'c:')
    # contact point
    contact=False
    mder=np.argmax(der)
    for i in range(len(der)):
        if i<mder:
            continue
        if der[i]<Mdc*der[mder]:
            contact=i
            break
    if contact:
        BEFOR_flag=True
        BEFOR_data=data[ind,:][::-1,:][contact:,:][::-1,:]
        BEFOR_noise=noise[ind,:][::-1,:][contact:,:][::-1,:]
        BEFOR_tim=tim[ind][::-1][contact:][::-1]
        
        try:
            BEFOR_Idata=BEFOR_data[-rr-dr:-dr,:]
            BEFOR_Itim=BEFOR_tim[-rr-dr:-dr]
            BEFOR_Inoise=BEFOR_noise[-rr-dr:-dr,:]
        except:
            BEFOR_Idata=BEFOR_data[:-dr,:]
            BEFOR_Itim=BEFOR_tim[:-dr]
            BEFOR_Inoise=BEFOR_noise[:-dr,:]
        if len(BEFOR_Itim)>=rr*0.7: 
            if plot:
                pyplot.plot(BEFOR_Itim,BEFOR_Idata[:,ch],color='r',marker='.',linestyle='')
            #### filter variations
            #print BEFOR_Itim.shape,BEFOR_Idata[:,ch].shape,BEFOR_Inoise[:,ch].shape,BEFOR_noise.shape
            #print BEFOR_Idata[:,ch]
            popt,pconv=curve_fit(_line,BEFOR_Itim-tim[0],BEFOR_Idata[:,ch],sigma=BEFOR_Inoise[:,ch])
            STD=np.std(BEFOR_Idata[:,ch]-(BEFOR_Itim-tim[0])*popt[0]-popt[1])
            while STD>0.002:
                di=np.argmax(np.abs(BEFOR_Idata[:,ch]-(BEFOR_Itim-tim[0])*popt[0]-popt[1]))
                BEFOR_Idata=np.delete(BEFOR_Idata,di,axis=0)
                BEFOR_Itim=np.delete(BEFOR_Itim,di)
                BEFOR_Inoise=np.delete(BEFOR_Inoise,di,axis=0)
                if len(BEFOR_tim)<rr*0.7:
                    break
                popt,pconv=curve_fit(_line,BEFOR_Itim-tim[0],BEFOR_Idata[:,ch],sigma=BEFOR_Inoise[:,ch])
                STD=np.std(BEFOR_Idata[:,ch]-(BEFOR_Itim-tim[0])*popt[0]-popt[1])
        
            DF=np.nanmin(BEFOR_data[-3:,ch])-(popt[0]*(BEFOR_tim[-1]-tim[0])+popt[1])
            if len(BEFOR_Itim)<rr*0.7 or DF>0.001:
                BEFOR_flag=False
        else:
            BEFOR_flag=False
        if plot and BEFOR_flag:
            pyplot.plot(BEFOR_Itim,BEFOR_Idata[:,ch],color='#dd1c77',marker='.',linestyle='',label='interpolation data (before)')
            pyplot.plot(BEFOR_tim[-1],BEFOR_data[-1,ch],color='#dd1c77',marker='x',markersize=10,label='contact_point (before)')
#            for i in range(CH-1):
#                pyplot.plot(BEFOR_Itim,BEFOR_Idata[:,i]*SCALE[i]+i*0.3,'c.')
#                pyplot.plot(BEFOR_tim[-1],BEFOR_data[-1,i]*SCALE[i]+i*0.3,'c.')
    else:
        BEFOR_flag=False       
    

    #############################
    ### Interpolate diffuse blocked
    #############################
    
    if BEFOR_flag:
        BEFOR_slopes=np.zeros(CH)
        BEFOR_interc=np.zeros(CH)
        BEFOR_point=np.zeros(CH)
        BEFOR_err=np.zeros(CH)
        BEFOR_DF=np.zeros(CH)
        for i in range(CH):
            try:
                popt,pconv=curve_fit(_line,BEFOR_tim[-rr-dr:-dr]-tim[0],BEFOR_data[-rr-dr:-dr,i],sigma=BEFOR_noise[-rr-dr:-dr,i])
            except:
                popt,pconv=curve_fit(_line,BEFOR_tim[:-dr]-tim[0],BEFOR_data[:-dr,i],sigma=BEFOR_noise[:-dr,i])
            BEFOR_slopes[i],BEFOR_interc[i]=popt
            BEFOR_DF[i]=np.nanmin(BEFOR_data[-3:,i])-(popt[0]*(BEFOR_tim[-1]-tim[0])+popt[1])
            BEFOR_point[i]=popt[0]*(MIN_TIM-tim[0])+popt[1]+BEFOR_DF[i]
 
            if np.array(pconv).size!=4:
                BEFOR_err[i]=np.nan
            else:
                BEFOR_err[i]=np.nanmean(abs(MIN_TIM-tim[0])*(pconv[0,0]**0.5)+(pconv[1,1]**0.5))
        if BEFOR_point[ch]<MIN_AVG[ch]:
            BEFOR_flag=False
        if plot and BEFOR_flag:
            pyplot.plot(tim,((tim-tim[0])*BEFOR_slopes[ch]+BEFOR_interc[ch]+BEFOR_DF[ch]),color='#dd1c77',linestyle=':',label='linear regression (before)')
            #pyplot.plot(tim,((tim-tim[0])*BEFOR_slopes[ch]+BEFOR_interc[ch]),'c:')
#            for i in range(CH-1):
#                pyplot.plot(tim,((tim-tim[0])*BEFOR_slopes[i]+BEFOR_interc[i]+BEFOR_DF[i])*SCALE[i]+i*0.3,'c:')
#                pyplot.plot(tim,((tim-tim[0])*BEFOR_slopes[ch]+BEFOR_interc[ch])*SCALE+i*0.3,'c:')
    
    if AFTER_flag:
        AFTER_slopes=np.zeros(CH)
        AFTER_interc=np.zeros(CH)
        AFTER_point=np.zeros(CH)
        AFTER_err=np.zeros(CH)
        AFTER_DF=np.zeros(CH)
        for i in range(CH):
            try:
                popt,pconv=curve_fit(_line,AFTER_tim[dr:rr+dr]-tim[0],AFTER_data[dr:rr+dr,i],sigma=AFTER_noise[dr:rr+dr,i])
            except:
                popt,pconv=curve_fit(_line,AFTER_tim[dr:]-tim[0],AFTER_data[dr:,i],sigma=AFTER_noise[dr:,i])
            AFTER_slopes[i],AFTER_interc[i]=popt
            AFTER_DF[i]=np.nanmin(AFTER_data[:3,i])-(popt[0]*(AFTER_tim[0]-tim[0])+popt[1])
            AFTER_point[i]=popt[0]*(MIN_TIM-tim[0])+popt[1]+AFTER_DF[i]
            if np.array(pconv).size!=4:
                AFTER_err[i]=np.nan
            else:
                AFTER_err[i]=np.nanmean(abs(MIN_TIM-tim[0])*(pconv[0,0]**0.5)+(pconv[1,1]**0.5))
        if AFTER_point[ch]<MIN_AVG[ch]:
            AFTER_flag=False
        if plot and AFTER_flag:
            pyplot.plot(tim,((tim-tim[0])*AFTER_slopes[ch]+AFTER_interc[ch]+AFTER_DF[ch]),color='#31a354',linestyle=':',label='linear_regression (after)')
            #pyplot.plot(tim,((tim-tim[0])*AFTER_slopes[ch]+AFTER_interc[ch]),'g:')
#            for i in range(CH-1):
#                pyplot.plot(tim,((tim-tim[0])*AFTER_slopes[i]+AFTER_interc[i]+AFTER_DF[i])*SCALE[i]+i*0.3,'g:')
#                pyplot.plot(tim,((tim-tim[0])*AFTER_slopes[ch]+AFTER_interc[ch])*SCALE+i*0.3,'g:')
#            
    if AFTER_flag and BEFOR_flag:
        DIF_mean=(BEFOR_point+AFTER_point)*0.5
        DIF_err=(BEFOR_err+AFTER_err)*0.5
    elif BEFOR_flag:
        DIF_mean=BEFOR_point
        DIF_err=BEFOR_err
    elif AFTER_flag:
        DIF_mean=AFTER_point
        DIF_err=AFTER_err
    else:
        DIF_mean=[np.nan]*CH
        DIF_err=[np.nan]*CH
    return MIN_TIM,MIN_AVG,MIN_ERR,DIF_mean,DIF_err


def dangle(datapf,filecfg,tiltcfg,lookups,position,lvl=0):
    gc.enable()
    def rpy2XYZ(r,p,y):
        c=np.pi/180.
        X=-np.sin(p*c)*np.cos(r*c)*np.cos(y*c)-np.sin(r*c)*np.sin(y*c)
        Y=np.sin(p*c)*np.cos(r*c)*np.sin(y*c)-np.sin(r*c)*np.cos(y*c)
        Z=np.cos(p*c)*np.cos(r*c)
        return np.array([X,Y,Z])
    
    def XYZy2rp(A,y):
        c=np.pi/180.
        X,Y,Z=A
        r=np.arcsin(-(X*np.sin(y*c)+Y*np.cos(y*c)))
        p=np.arctan2(Y*np.sin(y*c)-X*np.cos(y*c),Z)
        return r/c,p/c
    
    def get_y_Offset_roll(dy,rol,pit,r,p,y,N):
        cutoff=2 #cutoff frequenzy
        fs=15 #Hz
        order=6
        A=rpy2XYZ(rol,pit,y+dy)
        rol1,pit1=XYZy2rp(A,y)
        rols1 = butter_lowpass_filter(rol1, cutoff, fs, order)
        rs = butter_lowpass_filter(r, cutoff, fs, order)
    #    dr=np.convolve(rol1,np.ones((N,))/N,mode='valid')-np.convolve(r,np.ones((N,))/N,mode='valid')
        return 1.-np.corrcoef(rs,rols1)[0,1]
    
    def get_y_Offset_pitch(dy,rol,pit,r,p,y,N):
        cutoff=2 #cutoff frequenzy
        fs=15 #Hz
        order=6
        A=rpy2XYZ(rol,pit,y+dy)
        rol1,pit1=XYZy2rp(A,y)
        pits1 = butter_lowpass_filter(pit1, cutoff, fs, order)
        ps = butter_lowpass_filter(p, cutoff, fs, order)
    #    dp=np.convolve(pit1,np.ones((N,))/N,mode='valid')-np.convolve(p,np.ones((N,))/N,mode='valid')
        return 1.-np.corrcoef(ps,pits1)[0,1]
    
    def get_dr_dp(dy,rol,pit,r,p,y,N):
        cutoff=2 #cutoff frequenzy
        fs=15 #Hz
        order=6
        A=rpy2XYZ(rol,pit,y+dy)
        rol1,pit1=XYZy2rp(A,y)
        rols1 = butter_lowpass_filter(rol1, cutoff, fs, order)
        pits1 = butter_lowpass_filter(pit1, cutoff, fs, order)
        rs = butter_lowpass_filter(r, cutoff, fs, order)
        ps = butter_lowpass_filter(p, cutoff, fs, order)
        dr=np.convolve(rols1,np.ones((N,))/N,mode='valid')-np.convolve(rs,np.ones((N,))/N,mode='valid')
        dp=np.convolve(pits1,np.ones((N,))/N,mode='valid')-np.convolve(ps,np.ones((N,))/N,mode='valid')
        dr=np.array([np.nan]*int(N/2)+list(dr)+[np.nan]*int(N/2-1))
        dp=np.array([np.nan]*int(N/2)+list(dp)+[np.nan]*int(N/2-1))
        return dr,dp
    
        
    
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def circular_mean(X,W):
        #calculates the mean of a set of angles -> mean([355,5])=0 ; mean(170,190)=180
        #X [deg]
        c=np.pi/180.
        mean=np.arctan2(np.sum(np.sin(X*c)*W),np.sum(np.cos(X*c)*W))/c
        return mean

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
    
    c=np.pi/180.
    pfx=filecfg['pfx']
    dates=alldays(datapf,pfx)
    print_status(str("Looking for GUVis offset angles tiltcfg['dangle']"),
                 lvl=lvl)
    ### load data from whole cruise
    keys=['roll','pitch','yaw','roll_guvis','pitch_guvis','time']
    raw={}
    for i,date in enumerate(dates):
        print_status(str("Loading cruise data file "+
                         "%s/%s"%(str(i+1).zfill(2),str(len(dates)).zfill(2))),
                     end='\r',
                     flush=True,
                     lvl=lvl)
                         
        filecfg['ftype']='raw'
        filecfg['rawflags']=['C']
        f=fh.load_raw(date,
                        datapf,
                        filecfg=filecfg,
                        tiltcfg=tiltcfg,
                        lookups=lookups,
                        position=position,
                        lvl=-np.inf)
        if type(f)==type(None):
            continue
        if not 'yaw' in f.keys():
            continue

        ### select only valid values
        ind=f['roll'].mask+f['pitch'].mask+f['yaw'].mask\
            +f['roll_guvis'].mask+f['pitch_guvis'].mask\
            +np.isnan(f['roll'])+np.isnan(f['pitch'])+np.isnan(f['yaw'])\
            +np.isnan(f['roll_guvis'])+np.isnan(f['pitch_guvis'])
        
        for key in keys:
            f[key]=f[key].data[~ind]
        
        ### sort to avoid timejumps
        ind=np.argsort(f['time'])
        for key in keys:
            f[key]=f[key][ind]
        
        ### preselect low variance to reduce raw total size
        gangle=np.arccos(np.cos(f['roll_guvis']*c)*np.cos(f['pitch_guvis']*c))/c
        dtime=np.datetime64("1970-01-01T00:00:00")+np.array(f['time'],dtype=np.timedelta64(1,'s'))
        V=pd.Series(gangle,dtime).rolling(window=pd.offsets.Second(10*60)).var()
        ind=V<=np.percentile(V[~np.isnan(V)],25) 
        
        if len(raw.keys())==0:
            for key in keys:
                raw[key]=f[key][ind]
        else:
            for key in keys:
                raw[key]=np.array(list(raw[key])+list(f[key][ind]))
                
    print  #leaving progess (flushed) line
    print_status(str("select low variance angles"),lvl=lvl)
    ### final select low variance situations
    gangle=np.arccos(np.cos(raw['roll_guvis']*c)*np.cos(raw['pitch_guvis']*c))/c
    dtime=np.datetime64("1970-01-01T00:00:00")+np.array(raw['time'],dtype=np.timedelta64(1,'s'))
    V=pd.Series(gangle,dtime).rolling(window=pd.offsets.Second(10*60)).var()
    ind=V<=np.percentile(V[~np.isnan(V)],10)   
    
    for key in keys:
        raw[key]=raw[key][ind]

    rol=raw['roll_guvis']
    pit=raw['pitch_guvis']
    r=raw['roll']
    p=raw['pitch']
    y=raw['yaw']

    ### find yaw offset by correlating roll and pitch separate
    res_r=minimize_scalar(get_y_Offset_roll,
                          bounds=[0,360],
                          args=(rol,pit,r,p,y,900),
                          method='bounded')
    res_p=minimize_scalar(get_y_Offset_pitch,
                          bounds=[0,360],
                          args=(rol,pit,r,p,y,900),
                          method='bounded')
    dy=np.array([res_r.x,res_p.x])
    dyc=np.array([1.-res_r.fun,1.-res_p.fun])

    vdy=circular_mean(dy,dyc)
    da=(dy-vdy)
    da[da>180]-=360.
    vdys=np.sum(np.abs(da*dyc))/np.sum(dyc)

    ### calculate dr,dp with yaw offset calculated before
    dr,dp=get_dr_dp(vdy,rol,pit,r,p,y,900)
    ldr=np.nanmean(dr)
    ldp=np.nanmean(dp)

    print_status(str("Offset Angles for %s:"%pfx),lvl=lvl)
    print_status(str("dp: %.3f"%ldp),lvl=lvl+1)
    print_status(str("dr: %.3f"%ldr),lvl=lvl+1)
    print_status(str("yaw_init: %.3f +- %.3f"%(vdy,vdys)),lvl=lvl+1)
    
    return ldp,ldr,vdy