#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:58:11 2019

@author: walther
"""

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot


def plot_nk(sdate, edate):
    def smooth(x, window_len=11, window='hanning'):
        """smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    
        output:
            the smoothed signal
            
        example:
    
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
     
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        np.r_
        # s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        s = x
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='same')
        return y

    pf = "/home/walther/Documents/test/"
    k = np.load(pf + 'nk.npy')
    time = np.load(pf + 'ntim.npy')
    zen = np.load(pf + 'nzen.npy')
    beta = np.load(pf + 'nbeta.npy')

    roll = np.load(pf + 'nroll.npy')
    pitch = np.load(pf + 'npitch.npy')
    yaw = np.load(pf + 'nyaw.npy')

    ksmooth = smooth(k[-1, :])

    hours = mdates.HourLocator()
    minut = mdates.MinuteLocator()
    second = mdates.SecondLocator(bysecond=np.arange(15, 60, 15))
    datefmt = mdates.DateFormatter("%H:%M")

    tim = np.datetime64('1970-01-01') + (time * 1000).astype('timedelta64[ms]')
    ind = np.logical_and(tim > sdate, tim < edate)

    #    pyplot.figure(figsize=(10,7))
    f, axarr = pyplot.subplots(1, 1, sharex=True)  # , gridspec_kw = {'height_ratios':[3,3,1]}
    pyplot.subplots_adjust(hspace=0.05)
    axarr.xaxis.set_major_locator(minut)
    axarr.xaxis.set_major_formatter(datefmt)
    axarr.xaxis.set_minor_locator(second)
    #    axarr.plot(tim[ind],k[-1,ind],label='k',color='b')#,linestyle='',marker='.')
    #    axarr.plot(tim[ind],ksmooth[ind],label='ksmooth',color='r')#,linestyle='',marker='.')
    #    axarr.plot(tim[ind],zen[ind],label='zen',color='b')#,linestyle='',marker='.')
    #    axarr.plot(tim[ind],beta[ind],label='beta',color='r')#,linestyle='',marker='.')
    axarr.plot(tim[ind], roll[ind], label='roll', color='b')  # ,linestyle='',marker='.')
    axarr.plot(tim[ind], pitch[ind], label='pitch', color='r')  # ,linestyle='',marker='.')
    axarr.grid(True)
    axarr.legend(loc=1)
    axarr.set_ylabel(r'k')
    f.autofmt_xdate()
    f.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    plot_nk(sdate=np.datetime64('2016-04-22T15:26:45'),
            edate=np.datetime64('2016-04-22T15:29:15'))
