#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modules.get_toa.py is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors/Copyright(2014-2018):
# -Jonas Witthuhn (witthuhn@tropos.de)


'''
modules.get_toa.py is a python-module for calculation of the spectral solar
irradiance (as seen from a sensor with a defined filter) at top of atmosphere
in the wavelength range of 200-1800nm. The calculations mainly based on scaling
to solar activity based on measurements from the SORCE satellite mission (see
http://lasp.colorado.edu/sorce/
http://earthobservatory.nasa.gov/Library/SORCE/ for more information), and 
scaling due to sun - earth geometrie (see algorithm from the Astronomical 
Alamanac. The solar spectrum used for scaling is 'NewGuey2003'.

Methods
-------
_assume_TSI(date,basepf='../lookuptables/',lvl=0)
    `_assume_TSI` is a helper function for `get_I0` to provide total solar
    irradiance on given date for scaling top of atmosphere irradiance.
    Extrapolate the total solar irradiance from previous runs on different
    dates from the dataset `basepf`+'TSI.dat', if SORCE data is not available
    or no internet connection is available.
_get_TSI(date,basepf='../lookuptables/',lvl=0)
    `_get_TSI` is a helper function for `get_I0` to provide total solar
    irradiance on given date for scaling top of atmosphere irradiance.
    If not already aquired in previous runs, this function downloads the total
    solar irradiance (24h mean) on given date and stores it for later use in 
    `basepf`+'TSI.dat'.
get_I0(date,wvls,basepf='../lookuptables/',
       dbresponse="calibration/spectralResponse.dat",assume='close',lvl=0)
    `get_I0` scales the 'NewGuey2003' spectrum to solar activity and earth-sun-
    distance and convolves it with the spectral response function at given
    wavelengths. Providing the top of atmosphere irradiance as seen from a
    sensor with a defined filter. The response function can also be assumed to
    be a gaussian function with a given bandwith using the `assume` statement.
    
See also
--------
modules.sunpos
modules.helpers

References
----------
.. [1] United States Naval Observatory, 1993: The Astronomical Almanac, 
       Nautical Almanac Office, Washington DC.
.. [2] World Meteorological Organization, 2014: Guide to Meteorological 
       Instruments and Methods of Observation. Geneva, Switzerland, World 
       Meteorological Organization, 1128p. (WMO-No.8, 2014).
.. [3] Christian A.Gueymard, 2014: The sun's total and spectral irradiance for
       solar energy applications and solar radiation models.,
       https://doi.org/10.1016/j.solener.2003.08.039

'''

import sys
import warnings

if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.signal import general_gaussian

import sunpos as sp
from helpers import print_status


def _assume_TSI(date,
                basepf='/../lookuptables/',
                lvl=0):
    r"""
    Assuming 24h mean Total Solar Irradiance (TSI) at 1AU of given date, if no
    internet connection available or SORCE server is down, from previouse
    aquired data in ../lookuptables/TSI.dat.
    
    Parameters
    ----------
    date : `numpy.datetime64` object
        `date` represents the actual date to look for in the dataset.
    basepf : str, optional
        `basepf` represents the path-string to the ShRad base directory.
        default: "../lookuptables/"
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default=0
            
    Returns
    -------
    tsi : float
        The total solar irradiance at 1 AU [Wm-2]
    etsi : float
        The measurement uncertainty of the total solar irradiance at 1AU [Wm-2]
    """

    ###Load previouse aquired data.
    fname = basepf + 'TSI.dat'
    dat = np.loadtxt(fname)[:, [0, 1, 5]]
    dat = dat[dat[:, 0].argsort()]
    print_status("Assuming TSI from interpolating dataset %s" % (fname), lvl)

    ###Interpolate or extrapolate TSI from the dataset.
    ###Julday is calculated for EPOCH_J2000_0 at 00:00AM.
    date = date.astype("datetime64[s]")
    jday = sp.datetime2julday(date.astype("datetime64[D]")) - 0.5

    z = np.polyfit(dat[:, 0], dat[:, 1], 3)
    f = np.poly1d(z)
    tsi = f(jday)
    etsi = np.std(dat[:, 1]) + np.median(dat[:, 2])

    ### Warn if there is a large data gap (>150days) in the dataset to the
    ### given date.
    if np.min(np.abs(dat[:, 0] - jday)) > 150:
        warnings.warn("Assuming TSI may not be stable with "
                      + "a datagap of %d days." % (np.min(np.abs(dat[:, 0] - jday)))
                      + "Dataset used for extrapolation: %s" % fname)
    print_status("...done!", lvl, style='g')
    print_status("", lvl - 1)
    return np.float(tsi), np.float(etsi)


def _get_TSI(date,
             basepf='/../lookuptables/',
             lvl=0):
    r"""
    Aquiring 24h mean Total Solar Irradiance (TSI) at 1AU of given date from
    SORCE server and save the data in ../lookuptables/TSI.dat.
    
    Parameters
    ----------
    date : `numpy.datetime64` object
        `date` represents the actual date to look for in the dataset.
    basepf : str, optional
        `basepf` represents the path-string to the ShRad base directory.
        default: "../lookuptables/"
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default=0    
            
    Returns
    -------
    tsi : float
        The total solar irradiance at 1 AU [Wm-2]
    etsi : float
        The measurement uncertainty of the total solar irradiance at 1AU [Wm-2]
    """
    print_status(str("Get +-12h (24h) mean of total solar irradiance (TSI)"
                     + "at 1AU for date: "
                     + pd.to_datetime(date).strftime("%Y-%m-%d 12:00 UTC")), lvl)

    ###Julday is calculated for EPOCH_J2000_0 at 00:00AM.
    date = date.astype("datetime64[s]")
    jday = sp.datetime2julday(date.astype("datetime64[D]")) - 0.5

    if (np.datetime64('today') - date).astype('timedelta64[D]').astype(int) <= 7:
        print_status("No SORCE data for this day. Data is published with a 7 day delay.", lvl, style='fail')
        tsi1au, etsi = _assume_TSI(date, basepf, lvl + 1)
        print_status("..done!", lvl, style='g')
        print_status("", lvl - 1)
        return np.float(tsi1au), np.float(etsi)

    ### Check if data is already in database, if not download it from SORCEdata
    ### and save it to the database (dat).    
    ### From database (dat) aquiring jday,tsi_1au,etsi_1au - later to be scaled
    ### earth_sun_distance.
    fname = basepf + 'TSI.dat'
    dat = np.loadtxt(fname)
    if len(dat) == 0:  # file is empty
        dat = np.zeros((1, 13))
    if np.int64(jday) not in np.array(dat[:, 0], dtype=np.int64):
        print_status("Download todays TSI data from lasp.colorado.edu", lvl)
        url = str("http://lasp.colorado.edu/lisird/latis/dap/"
                  + "sorce_tsi_24hr_l3.tab?&time>="
                  + pd.to_datetime(date).strftime(str("%Y-%m-%dT00:00:00&time"
                                                      + "<%Y-%m-%dT23:59:59")))
        try:
            u = urlopen(url)
        except:
            print_status("lasp.colorado.edu is not available. No Internet?",
                         lvl,
                         style='fail')
            tsi1au, etsi = _assume_TSI(date, basepf, lvl + 1)
            print_status("..done!", lvl, style='g')
            print_status("", lvl - 1)
            return np.float(tsi1au), np.float(etsi)
        ulines = u.readlines()

        u.close()
        if len(ulines) != 0:
            l = ulines[-1].decode()
            l = l.replace('\t', ' ')
            try:
                l.split(' ')  # check if the data is in proper shape to continue
            except:
                print_status("No SORCE data for this day.", lvl, style='fail')
                tsi1au, etsi = _assume_TSI(date, basepf, lvl + 1)
                print_status("..done!", lvl, style='g')
                print_status("", lvl - 1)
                return np.float(tsi1au), np.float(etsi)
        else:
            print_status("No SORCE data for this day.", lvl, style='fail')
            tsi1au, etsi = _assume_TSI(date, basepf, lvl + 1)
            print_status("..done!", lvl, style='g')
            print_status("", lvl - 1)
            return np.float(tsi1au), np.float(etsi)
        tsi_append = l.split(' ')

        tsi_append[0] = np.float(jday)
        np.savetxt(fname, np.array(np.vstack((dat, tsi_append)), dtype=np.float),
                   delimiter=' ',
                   header=str('jday '
                              + 'tsi_1au'
                              + 'instrument_accuracy_1au '
                              + 'instrument_precision_1au '
                              + 'solar_standard_deviation_1au '
                              + 'measurement_uncertainty_1au '
                              + 'tsi_true_earth '
                              + 'instrument_accuracy_true_earth '
                              + 'instrument_precision_true_earth '
                              + 'solar_standard_deviation_true_earth '
                              + 'measurement_uncertainty_true_earth '
                              + 'avg_measurement_date '
                              + 'std_dev_measurement_date\n'
                              + 'EPOCH_J2000_0+12h'
                              + 'Wm-2 ' + 'Wm-2 ' + 'Wm-2 ' + 'Wm-2 ' + 'Wm-2 ' + 'Wm-2 '
                              + 'Wm-2 ' + 'Wm-2 ' + 'Wm-2 ' + 'Wm-2 '
                              + 'sorce_julian_date ' + 'days'))
        print_status("..done!", lvl, style='g')
        print_status("", lvl - 1)
        tsi1au, etsi = tsi_append[1], tsi_append[5]
        return np.float(tsi1au), np.float(etsi)
    else:
        tsi1au, etsi = dat[np.array(dat[:, 0], dtype=np.int64) == np.int64(jday), [1, 5]]
        print_status("..done!", lvl, style='g')
        print_status("", lvl - 1)
        return np.float(tsi1au), np.float(etsi)


def get_I0(date,
           wvls,
           cwvls,
           basepf='../lookuptables/',
           dbresponse="calibration/spectralResponse.dat",
           assume='close',
           lvl=0):
    r"""
    Calculating the top of atmosphere (TOA) spectral irradiance for given
    wavelengths `wvls`. The irradiance is scaled to actual sun activity and
    spectral response function of channel filters for each wavelength. The
    spectral response should be defined in the file `dpresponse` or could also
    be assumed with the `assume` statement.
    
    Parameters
    ----------
    date : `numpy.datetime64` object
        `date` represents the actual date to look for in the dataset.
    wvls : iterable, float or int
        `wvls` represent the spectral wavelengths to calculate the irradiance
        at TOA for.
    cwvls : iterable, float ot int
        `cwvls` represents the centroid wavelength of the spectral response 
        funtion.
    basepf : str, optional
        `basepf` represents the path-string to the ShRad base directory.
        default: "../lookuptables/"
    dbresponse : str, optional
        `dpresponse` defines the location of the spectral response data relative
        from `basepf`. Default: "../lookuptables/calibration/spectralResponse.dat".
        The file should be in the the format: (1)line starts with 0 and than
        channels names corresponding to `wvls`. The following lines starting
        with calibration wavelength followed by the normalized (to maximum) 
        response of each channel. Separator =' ', Comments ='#'
    assume : str or int or float, optional
        If assignet with int or float and `wvls` are not found in file 
        `dpresponse` a default gaussian window is assigned as spectral response
        function with FWHM of `assume` [nm]. -- If assignet with 'close' and
        `wvls` are not found in file `dpresponse` the response function of the 
        closest channel is assignet to the given `wvls`. Default: 'close'
    lvl : int, optional
        `lvl` represents the intention level of the workflow for nicely printed
        status messages. Only nessesary for the looks. Default:0   
            
    Returns
    -------
    I0 : `numpy.array`, float, shape(wvls)
        `I0` is the calculated spectral solar irradiance at TOA
    eI0 : `numpy.array`, float, shape(wvls)
        `eI0` is the assumed error of `I0` calculated from spectrum variations
        and total solar irradiance measurement uncertainty.
    """
    wvls = np.array(wvls, dtype=np.float)
    print_status("Calculate spectral Irradiance at TOA (I0)", lvl)
    ### load NewGuey2003 spectrum and scale it with the actual total solar
    ### irradiance. This will serve as top of atmosphere incoming irradiance
    ### to calculate spektral optical depths. 
    tsi1au, etsi = _get_TSI(date, basepf, lvl + 1)
    Sc = 1366.1  # NewGuey2003 Solar constant
    I0dat = np.loadtxt(basepf + "NewGuey2003.dat",
                       converters={1: lambda s: float(s) / 1000.})
    I01 = griddata(I0dat[:, 0], I0dat[:, 1], np.arange(200, 1800))
    I0 = I01 * (tsi1au / Sc)  # scale I0 to solar activity

    ### The scaled spectrum have to be convolved with the spectral response
    ### function of each filter to calculate the true spectral irradiance at
    ### top of atmosphere as the sensor would measure through the filter at TOA.
    I0s = np.zeros(len(wvls))
    ### Standard deviation to other spectras vs NewGuey2003
    E_I0 = np.zeros(len(wvls))
    E_I0[wvls >= 1000] = 1.1
    E_I0[wvls < 1000] = 0.8
    E_I0[wvls < 700] = 1.2
    E_I0[wvls < 400] = 3.4
    try:
        response = np.loadtxt(basepf + dbresponse, delimiter=',')
        channels = response[0, 1:]
        response = response[1:, :]
    except:
        warnings.warn("There is no spectral calibration data.")
        channels = []
        response = []
    for i, w in enumerate(wvls):
        if np.int64(w) in channels:
            ### Aquire the spectral response function centered on the centroid
            ### wavelength. Centering is nessesary for convolving the spectrum
            res = griddata(response[:, 0], response[:, 1:][:, channels == np.int64(w)],
                           np.arange(int(np.round(cwvls[i], 0) - 30),
                                     int(np.round(cwvls[i], 0) + 30)), fill_value=0)
            res = res.flatten()
            I0i = np.convolve(I0, res / np.sum(res), 'same')
            I0s[i] = griddata(np.arange(200, 1800), I0i, np.array(w))
        else:
            if (assume == b'close' or assume == 'close') and len(channels) != 0:
                cw = channels[np.argmin(np.abs(channels - w))]
                warnings.warn('Requested wavelength %.2f is not in response ' % (w)
                              + 'dataset, calculations are performed now with '
                              + 'the next "close" wavelength ->%.2f in the ' % (cw)
                              + 'dataset! Large errors may arise if the '
                              + 'wavelengths are to far from each other. Please '
                              + 'be aware of this or choose FWHM value for the '
                              + 'assume parameter for a better approximation.')
                res = griddata(response[:, 0], response[:, 1:][:, channels == np.int64(cw)],
                               np.arange(int(np.round(cwvls[i], 0) - 30),
                                         int(np.round(cwvls[i], 0) + 30)), fill_value=0)
                res = res.flatten()
                I0i = np.convolve(I0, res / np.sum(res), 'same')
                I0s[i] = griddata(np.arange(200, 1800), I0i, np.array(w))
            else:
                try:
                    fwhm = np.float(assume)
                except:
                    # print(assume)
                    raise ValueError("Input of assume not understood! "
                                     + "Should be a number, or 'close'.")
                sig = fwhm / (2. * np.sqrt(2. * np.log(2)))
                res = np.trim_zeros(general_gaussian(np.int64(fwhm * 10), 1.5, sig),
                                    trim='fb')
                I0i = np.convolve(I0, res / np.sum(res), 'same')
                I0s[i] = griddata(np.arange(200, 1800), I0i, np.array(w))

    eI0 = I0s * etsi / Sc + E_I0 * I0s / 100.
    SED = sp.earth_sun_distance(sp.datetime2julday(date))
    I0s = I0s / (SED ** 2)
    eI0 = eI0 / (SED ** 2)
    print_status("..done!", lvl, style='g')
    print_status("", lvl - 1)
    return I0s, eI0
