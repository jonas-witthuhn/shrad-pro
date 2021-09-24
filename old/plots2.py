#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 07:16:33 2018

@author: walther
"""
import numpy as np
import sunpos as sp
from matplotlib import pyplot
from netCDF4 import Dataset


def _calc_viewzen(pitch, yaw, zen, azi):
    # calculate the angle between radiometer normal to sun position vektor
    c = np.pi / 180.
    p = -1. * pitch * c
    y = yaw * c
    z = zen * c
    a = azi * c
    g = a - y
    coszen = -np.sin(z) * np.sin(p) * np.cos(g) + np.cos(z) * np.cos(p)
    return coszen


pos = np.array([7, 9, 47, 54, 53, 37, 14, 18, 43, 15, 4, 1, 26, 46, 25, 44, 38, 87, 60, 86, 24, 32, 33, 5, 10, 12, 55])
ref = np.array(
    ['PV01', 'PV02', 'PV04', 'PV05', 'PV06', 'PV06', 'PV06', 'PV07', 'PV08', 'PV10', 'PV11', 'PV11', 'PV12', 'PV13',
     'PV14', 'PV15', 'PV16', 'PV17', 'PV18', 'PV19', 'PV20', 'PV21', 'PV22', 'MS02', 'MS02', 'MS02', 'MS01'])

pf = "/home/walther/Documents/metpvnet/data/pyrnet/NC_lvl1/"

date = "2018-09-12"

f1 = Dataset(pf + "MPVnet_PYRNET_%s_05.nc" % (date), 'r')
D1 = f1.variables
print
D1.keys()

dtim = np.datetime64('1970-01-01') + D1['time'][:].astype('timedelta64[s]')

sza, azi = sp.zenith_azimuth(sp.datetime2julday(dtim), D1['lat'], D1['lon'])
sza = sza.data
azi = azi.data

f2 = Dataset(pf + "MPVnet_PYRNET_%s_10.nc" % (date), 'r')
D2 = f2.variables

pyplot.figure()
ghi = D1['ghi'] / D1['ghi'].scale_factor
pyplot.plot(dtim, ghi, 'b')
gti = D1['ghi_tilt'][:].data / D1['ghi_tilt'].scale_factor

# gti=(D2['ghi_tilt'][:].data/D2['ghi_tilt'].scale_factor)
# czen=np.cos(sza*np.pi/180.)
# tczen=_calc_viewzen(float(f1.zenith),float(f1.azimuth),sza,azi)
# GTIs=[]
# for z in [0]:
#    for a in [-10,10]:
##        nczen=_calc_viewzen(float(f1.zenith)+float(z),float(f1.azimuth),sza,azi)
#        nczen=_calc_viewzen(float(f1.zenith)+float(z),float(f1.azimuth)+a,sza,azi)
#        if len(GTIs)==0:
#            GTIs=gti*nczen/tczen
#        else:
#            GTIs=np.vstack((GTIs,gti*nczen/tczen))
# print(np.max(np.abs(np.min(GTIs,axis=0)-np.max(GTIs,axis=0))))
#
# pyplot.fill_between(dtim,np.min(GTIs,axis=0),np.max(GTIs,axis=0),color='r',alpha=0.5)
pyplot.plot(dtim, gti, 'k')

dtim = np.datetime64('1970-01-01') + D2['time'][:].astype('timedelta64[s]')
# sza,azi=sp.zenith_azimuth(sp.datetime2julday(dtim),D2['lat'],D2['lon'])
# sza=sza.data
# azi=azi.data
# gti=D2['ghi_tilt'][:].data/D2['ghi_tilt'].scale_factor
# tczen=_calc_viewzen(float(f2.zenith),float(f2.azimuth),sza,azi)
# GTIs=[]
# for z in [0]:
#    for a in [-10,10]:
##        nczen=_calc_viewzen(float(f1.zenith)+float(z),float(f1.azimuth),sza,azi)
#        nczen=_calc_viewzen(float(f2.zenith)+float(z),float(f2.azimuth)+a,sza,azi)
#        if len(GTIs)==0:
#            GTIs=gti*nczen/tczen
#        else:
#            GTIs=np.vstack((GTIs,gti*nczen/tczen))
#
# print(np.max(np.abs(np.min(GTIs,axis=0)-np.max(GTIs,axis=0))))
#
# pyplot.fill_between(dtim,np.min(GTIs,axis=0),np.max(GTIs,axis=0),color='m',alpha=0.5)
pyplot.plot(dtim, gti, 'k')

pyplot.xlabel("UTC", fontsize=16)
pyplot.ylabel("Irradiance Wm-2", fontsize=16)
pyplot.grid(True)
pyplot.show()

f1.close()
f2.close()
