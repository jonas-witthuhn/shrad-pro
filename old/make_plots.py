#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:10:56 2018

@author: walther
"""
import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot


def plot_angles_test(instime, insroll, inspitch, time, roll, pitch, sdate, edate):
    hours = mdates.HourLocator()
    minut = mdates.MinuteLocator()
    second = mdates.SecondLocator(bysecond=np.arange(15, 60, 15))
    datefmt = mdates.DateFormatter("%H:%M")

    instim = np.datetime64('1970-01-01') + (instime * 1000).astype('timedelta64[ms]')
    insind = np.logical_and(instim > sdate, instim < edate)

    tim = np.datetime64('1970-01-01') + (time * 1000).astype('timedelta64[ms]')
    ind = np.logical_and(tim > sdate, tim < edate)

    pyplot.figure(figsize=(10, 7))
    f, axarr = pyplot.subplots(1, 1, sharex=True)  # , gridspec_kw = {'height_ratios':[3,3,1]}
    pyplot.subplots_adjust(hspace=0.05)
    axarr.xaxis.set_major_locator(minut)
    axarr.xaxis.set_major_formatter(datefmt)
    axarr.xaxis.set_minor_locator(second)
    axarr.plot(instim[insind], insroll[insind], label='insroll', color='b')  # ,linestyle='',marker='.')
    axarr.plot(instim[insind], inspitch[insind], label='inspitch', color='r')
    axarr.plot(tim[ind], roll[ind], label='roll', color='b', linestyle=':')  # ,linestyle='',marker='.')
    axarr.plot(tim[ind], pitch[ind], label='pitch', color='r', linestyle=':')
    #    axarr.set_ylim([Fu,Fo])
    axarr.grid(True)
    axarr.legend(loc=1)
    axarr.set_ylabel(r'angle [degree]')
    f.autofmt_xdate()
    f.tight_layout()
    pyplot.show()


def plot_TCtest(raw, raw_tc, sdate, edate, Fu, Fo, title=''):
    hours = mdates.HourLocator()
    minut = mdates.MinuteLocator()
    second = mdates.SecondLocator(bysecond=np.arange(0, 60, 15))
    second2 = mdates.SecondLocator(bysecond=np.arange(0, 60, 5))
    datefmt = mdates.DateFormatter("%H:%M")

    tim = np.datetime64('1970-01-01') + (raw['time'][:] * 1000).astype('timedelta64[ms]')
    ind = np.logical_and(tim > sdate, tim < edate)

    #    pyplot.figure(figsize=(10,7))
    pyplot.rc('xtick', labelsize=16)
    pyplot.rc('ytick', labelsize=16)
    f, axarr = pyplot.subplots(1, 1, sharex=True, figsize=(10, 6), dpi=300)  # , gridspec_kw = {'height_ratios':[3,3,1]}
    axarr.set_title(title, fontsize=16)
    pyplot.subplots_adjust(hspace=0.05)
    axarr.xaxis.set_major_locator(minut)
    axarr.xaxis.set_major_formatter(datefmt)
    axarr.xaxis.set_minor_locator(second2)
    axarr.plot(tim[ind], raw['rad'][ind, -1], label='GLO_C', color='b', linewidth=3)  # ,linestyle='',marker='.')
    axarr.plot(tim[ind], raw_tc['rad'][ind, -1] - 20, label='GLO_TC', color='r', linewidth=3)
    axarr.set_ylim([Fu, Fo])
    axarr.grid(True, 'major', linewidth=2)
    axarr.grid(True, 'minor', linewidth=1)
    axarr.legend(loc=1, fontsize=16)
    axarr.set_ylabel(r'broadband irradiance $\left[\frac{W}{m^2}\right]$', fontsize=16)
    axarr.set_xlabel('UTC', fontsize=16)
    #    f.autofmt_xdate()
    f.tight_layout()
    #    pyplot.show()
    pyplot.savefig("/home/walther/Documents/Poster/201903_GUVis_posterupdate/tiltcorrect.png")
    return 0


def quicklookDATA(datapf, rad, aod):
    hours = mdates.HourLocator()
    minut = mdates.MinuteLocator(byminute=np.arange(5, 60, 5))
    datefmt = mdates.DateFormatter("%H:%S")

    tim = np.datetime64('1970-01-01') + (rad['time'][:] * 1000).astype('timedelta64[ms]')
    mdate = np.nanmedian(tim.astype('datetime64[D]').astype(int)).astype('datetime64[D]')
    mdate = pd.to_datetime(mdate)

    #    cm=np.load(datapf+'cm/%s'%(mdate.strftime('%Y%m%d_cm.npy')))
    #    ctim=np.datetime64('1970-01-01')+(cm[:,0]).astype('timedelta64[s]')
    #
    #

    pyplot.figure(figsize=(10, 7))
    f, axarr = pyplot.subplots(2, 1, sharex=True)  # , gridspec_kw = {'height_ratios':[3,3,1]}
    pyplot.subplots_adjust(hspace=0.05)
    axarr[0].xaxis.set_major_locator(hours)
    axarr[0].xaxis.set_major_formatter(datefmt)
    axarr[0].xaxis.set_minor_locator(minut)
    axarr[0].plot(tim, rad['Iglo'][:, -1], label='GLO', color='b')  # ,linestyle='',marker='.')
    axarr[0].plot(tim, rad['Idir'][:, -1] * np.cos(rad['zen'][:] * np.pi / 180.), label='DIR',
                  color='r')  # ,linestyle='',marker='.')
    axarr[0].plot(tim, rad['Idif'][:, -1], label='DIF', color='g')  # ,linestyle='',marker='.')
    axarr[0].set_ylim([0, 1500.])
    axarr[0].grid(True)
    axarr[0].legend(loc=1)
    axarr[0].set_ylabel(r'broadband irradiance $\left[\frac{W}{m^2}\right]$')
    ls = ['380nm', '443nm', '510nm', '665nm', '875nm']

    if type(aod) != type(None):
        tim2 = np.datetime64('1970-01-01') + (aod['time'][:] * 1000).astype('timedelta64[ms]')
        aod = aod['aod'][:, :]
        axarr[1].plot(tim2, aod[:, 2], label=ls[0], color='k', linestyle='', marker='.')
        axarr[1].plot(tim2, aod[:, 4], label=ls[1], color='b', linestyle='', marker='.')
        axarr[1].plot(tim2, aod[:, 5], label=ls[2], color='g', linestyle='', marker='.')
        axarr[1].plot(tim2, aod[:, 8], label=ls[3], color='y', linestyle='', marker='.')
        axarr[1].plot(tim2, aod[:, 12], label=ls[4], color='r', linestyle='', marker='.')
    axarr[1].xaxis.set_major_locator(hours)
    axarr[1].xaxis.set_major_formatter(datefmt)
    axarr[1].xaxis.set_minor_locator(minut)
    axarr[1].legend(loc=1)  # prop={'size':20})
    axarr[1].set_ylim([0, 0.8])
    axarr[1].grid(True)
    axarr[1].set_ylabel('AOD [#]')
    axarr[1].set_xlabel('time UTC')

    #    cm30=cm[:,2]
    #    cmal=cm[:,-1]
    #    ind=cm30>cmal
    #    axarr[2].fill_between(ctim,cm30,cmal,where=ind,interpolate=True,color='r')
    #    axarr[2].fill_between(ctim,cm30,cmal,where=~ind,interpolate=True,color='b')
    #    axarr[2].fill_between(ctim,cm30,0,where=~ind,interpolate=True,color='r')
    #    axarr[2].fill_between(ctim,cmal,0,where=ind,interpolate=True,color='b')
    #    axarr[2].set_yticks([0,0.25,0.5,0.75,1.])
    #    axarr[2].xaxis.set_major_locator(hours)
    #    axarr[2].xaxis.set_major_formatter(datefmt)
    #    axarr[2].xaxis.set_minor_locator(minut)
    #    axarr[2].grid(True)
    #    axarr[2].set_xlabel('time UTC')
    #    axarr[2].set_ylabel('Cloud\nCover [#]')
    #
    f.autofmt_xdate()
    f.tight_layout()
    #    pyplot.subplots_adjust({'hspace':0})

    if not os.path.exists(datapf + "quicklooks/"):
        os.mkdir(datapf + "quicklooks/")
    pyplot.savefig(datapf + "quicklooks/DATA%s.png" % (mdate.strftime("%Y%m%d")))
    return 0


if __name__ == '__main__':
    from netCDF4 import Dataset

    pf = "/home/walther/Documents/Instruments/GUVis_scripts/ShRad/data/ps98/raw/2016/04/"
    sdate = np.datetime64("2016-04-22T15:26:45")
    edate = np.datetime64("2016-04-22T15:29:15")
    title = 'ps98 - 22.04.2016'
    with Dataset(pf + 'ps98_GUV_000350_C_20160422.nc', 'r') as C, Dataset(pf + 'ps98_GUV_000350_C_TC_20160422.nc',
                                                                          'r') as TC:
        plot_TCtest(C, TC, sdate, edate, 760, 850, title)
