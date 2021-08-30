#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:19:27 2019

@author: walther
"""
import argparse
import os,sys

from matplotlib import pyplot
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from netCDF4 import Dataset



def quicklookDATA(rad,aod):
    hours=mdates.HourLocator()
    minut=mdates.MinuteLocator(byminute=np.arange(5,60,5))
    datefmt=mdates.DateFormatter("%H:%S")
    
    tim=np.datetime64('1970-01-01')+(rad['time'][:]*1000).astype('timedelta64[ms]')
    mdate=np.nanmedian(tim.astype('datetime64[D]').astype(int)).astype('datetime64[D]')
    mdate=pd.to_datetime(mdate)

    f=pyplot.figure(figsize=(10,7))
    ax=pyplot.subplot(311)
    
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(datefmt)
    ax.xaxis.set_minor_locator(minut)
    pyplot.plot(tim,rad['Iglo'][:,-1],label='GLO',color='b')#,linestyle='',marker='.')
    pyplot.plot(tim,rad['Idir'][:,-1]*np.cos(rad['zen'][:]*np.pi/180.),label='DIR',color='r')#,linestyle='',marker='.')
    pyplot.plot(tim,rad['Idif'][:,-1],label='DIF',color='g')#,linestyle='',marker='.')
    pyplot.ylim([0,1500.])
    pyplot.grid(True) 
    pyplot.legend(loc=1)
    pyplot.ylabel(r'broadband irradiance $\left[\frac{W}{m^2}\right]$')
    ls=['380nm','443nm','510nm','665nm','875nm']

    ax2=f.add_subplot(3,1,2,sharex=ax)
    if type(aod)!=type(None):
        tim2=np.datetime64('1970-01-01')+(aod['time'][:]*1000).astype('timedelta64[ms]')
        aod=aod['aod'][:,:]
        ax2.plot(tim2,aod[:,2],label=ls[0],color='k',linestyle='',marker='.')
        ax2.plot(tim2,aod[:,4],label=ls[1],color='b',linestyle='',marker='.')
        ax2.plot(tim2,aod[:,5],label=ls[2],color='g',linestyle='',marker='.')
        ax2.plot(tim2,aod[:,8],label=ls[3],color='y',linestyle='',marker='.')
        ax2.plot(tim2,aod[:,12],label=ls[4],color='r',linestyle='',marker='.')
    ax2.xaxis.set_major_locator(hours)
    ax2.xaxis.set_major_formatter(datefmt)
    ax2.xaxis.set_minor_locator(minut)
    ax2.legend(loc=1)#prop={'size':20})
    ax2.grid(True)    
    ax2.set_ylabel('AOD [#]')
    ax2.set_xlabel('time UTC')
    
    ax=pyplot.subplot(313)
    if type(aod)!=type(None):
        ax.plot(aod[:,2],label=ls[0],color='k',linestyle='',marker='.')
        ax.plot(aod[:,4],label=ls[1],color='b',linestyle='',marker='.')
        ax.plot(aod[:,5],label=ls[2],color='g',linestyle='',marker='.')
        ax.plot(aod[:,8],label=ls[3],color='y',linestyle='',marker='.')
        ax.plot(aod[:,12],label=ls[4],color='r',linestyle='',marker='.')

    ax.grid(True)    
    ax.set_ylabel('AOD [#]')
    ax.set_xlabel('index')
    
    
    f.autofmt_xdate()
    pyplot.subplots_adjust(hspace=0.05)
    f.tight_layout()
    pyplot.show()
    return 0

def show_data(infile):
    profile=infile.replace('AOD','processed',1).replace('aod','pro',1)
    with Dataset(profile,'r') as f:
        prodata={}
        for v in f.variables:
            prodata.update({v:f.variables[v][:]})           
    
    with Dataset(infile,'r') as f:
        aoddata={}
        for v in f.variables:
            aoddata.update({v:f.variables[v][:]})
    quicklookDATA(prodata,aoddata)





### Parsing command line arguments
##################################
parser=argparse.ArgumentParser()
parser.add_argument('infile',help='aod input file')
parser.add_argument('-c','--cut',nargs=2,type=int,help='cut dataset between both index numbers given')
parser.add_argument('-s','--show',help='only show the data',action="store_true")
args=parser.parse_args()

infile=os.path.abspath(args.infile)

if args.show:
    show_data(infile)

else:
    if len(args.cut)==2:
        with Dataset(infile,'a') as f:
            t=f.dimensions['t'].size
            ch=f.dimensions['ch'].size
            for v in f.variables:
                shape=f.variables[v].shape
                if len(shape)==0:
                    continue
                elif len(shape)==1 and shape[0]==t and shape[0]!=ch:
                    mask1=np.zeros(t).astype(bool)
                    mask1[args.cut[0]:args.cut[1]]=True
                    mask=f.variables[v][:].mask+mask1
                    f.variables[v][:]=np.ma.masked_where(mask,f.variables[v][:])
                elif len(shape)==2:
                    mask1=np.zeros((t,ch)).astype(bool)
                    mask1[args.cut[0]:args.cut[1],:]=True
                    mask=f.variables[v][:,:].mask+mask1
                    f.variables[v][:,:]=np.ma.masked_where(mask,f.variables[v][:,:])
                else:
                    continue
                

