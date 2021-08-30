#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:37:08 2018

@author: walther
"""

import numpy as np
from scipy.interpolate import griddata

def E_motion(wvl,zen,gamma,dbemotion):
    try:    ##make sure that wvl is iterable
        len(np.array(wvl))
        wvl=np.array(wvl)
    except:
        wvl=np.array([wvl,])
    E=[]
    for w in wvl:  
        dat=np.load(dbemotion+"%d.npy"%(int(w)))
        vals=dat.flatten()
        X,Y=np.meshgrid(np.arange(90),np.arange(90))
        X=X.flatten()
        Y=Y.flatten()
        gamma[np.isnan(gamma)==True]=-1 #-1 is put of grid -> evaluates to nan in griddata
        E.append(griddata((X,Y),vals,(zen,gamma),fill_value=np.nan))
    return np.array(E).T

def E_AOD(self,sza,I,aod,rod,o3OD,no2OD,h2ood,E_ext,E_ali,E_noi,e_h20_od,P,dP=5.):
    I0=self.I0
    eI0=self.eI0
    dI=np.sqrt((E_ali*I/(100.))**2+E_noi**2+(0.02*I)**2+(0.01*I)**2)#E_ali, E_noise, _E_cal, E_extra

    #dOD=(aod*dI/I)**2 #irradiance error
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

    rem=np.zeros((len(dOD[:,0]),len(self.channels[:-1])))
    rem[:,-1]=(4.361*10**(-5))**2+(1.764*10**(-5))**2+(4.76*10**(-5))**2+e_h20_od[:,-1]**2#dCO2,dCH4,DH20
    rem[:,-4]=(7.82*10**(-5))**2+e_h20_od[:,-4]**2 #dH20
    dOD+=rem
    dOD=np.sqrt(dOD)
    A=np.hstack((A,np.sqrt(rem)))
    if not os.path.exists(self.path+'err3/'):
        os.mkdir(self.path+'err3/')
    np.save(self.path+'err3/Err'+self.date.strftime("%Y%m%d")+'.npy',A)
    
    return dOD  