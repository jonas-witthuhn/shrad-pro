#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:44:40 2018

@author: walther
"""
import numpy as np


def tilt_correction(tiltcfg):
    print(self.tab * '  |' + "correct motion and cosine response deviation ...")
    self.tab += 1
    angdat = np.loadtxt(self.fpath + "lookuptables/calibration/AngularResponse_GUV350_140129.csv", delimiter=',')
    angcor = np.zeros((len(angdat[1:, 0]), len(self.channels)))
    for i in range(10):
        angcor[:, i] = angdat[1:, 1]
    angcor[:, 10:-1] = angdat[1:, 3:-1]
    angcor[:, -1] = np.mean(angdat[1:, -2:], axis=1)  # factor of broadband

    angzen = angdat[1:, 0]
    INSdat = self.load_ins()
    raw = self.load_raw(mode='raw')
    tim = raw['time'][:]
    azi = raw['azi'][:]
    zen = raw['zen'][:]
    roll = raw['roll'][:]
    pitch = raw['pitch'][:]
    if INSdat != False:
        instim = INSdat['time'][:]
        roll = griddata(instim, INSdat['roll'][:], tim)
        pitch = griddata(instim, INSdat['pitch'][:], tim)
        yaw = griddata(instim, INSdat['yaw'][:], tim)
        raw['roll'][:] = roll
        raw['pitch'][:] = pitch
        raw.update({'yaw': yaw})
    if not self.noINS:
        if INSdat == False:
            raise ValueError("No inertial navigation data for %s!" % (self.date.strftime("%Y-%m-%d")))
        # correct angular response
        beta = self._calc_viewzen(roll, pitch, yaw, zen, azi, droll, dpitch, dyaw)
        angcors = np.zeros((len(tim), len(self.channels)))
        for i in range(len(self.channels)):
            angcors[:, i] = griddata(angzen, angcor[:, i], beta)
        raw['rad'][:, :] = raw['rad'][:, :] / angcors[:, :]
        angerror = self._E_motion(self.channels, zen,
                                  self._calc_viewzen(roll, pitch, yaw, zen, azi, droll, dpitch, dyaw))
        k = self._get_k(roll, pitch, yaw, zen, azi, droll, dpitch, dyaw)  # with rayleight
        for i in range(len(self.channels)):
            raw['rad'][:, i] = raw['rad'][:, i] * k[i, :]
    else:
        # correct angular response
        if yaw == None and dyaw == None:  # cant calculate the misaligned zenith angle
            beta = self._calc_viewzen(roll, pitch, 0, azi, zen, azi, droll, dpitch, 0)
            raw.update({'yaw': [np.NaN]})
            angcors = np.zeros((len(tim), len(self.channels)))
            for i in range(len(self.channels)):
                angcors[:, i] = griddata(angzen, angcor[:, i], zen)
            angcors2 = np.zeros(
                (len(tim), len(self.channels)))  # maximum correction factor -> radiometer is misaligned towards the sun
            for i in range(len(self.channels)):
                angcors2[:, i] = griddata(angzen, angcor[:, i], beta)
            drad = raw['rad'][:, :] / angcors2[:, :]
            raw['rad'][:, :] = raw['rad'][:, :] / angcors[:, :]
            angerror = np.abs(drad - raw['rad'][:, :]) * 100. / raw['rad'][:, :]  # [%]
            angerror += self._E_motion(self.channels, zen,
                                       self._get_gamma(roll, pitch, azi, zen, azi, droll, dpitch, dyaw))
        else:
            yaw = np.ones(len(tim)) * yaw
            raw.update({'yaw': yaw})
            beta = self._calc_viewzen(roll, pitch, yaw, zen, azi, 0, 0, dyaw)
            angcors = np.zeros((len(tim), len(self.channels)))
            for i in range(len(self.channels)):
                angcors[:, i] = griddata(angzen, angcor[:, i], beta)
            raw['rad'][:, :] = raw['rad'][:, :] / (angcors[:, :])
            angerror = self._E_motion(self.channels, zen, self._calc_viewzen(roll, pitch, yaw, zen, azi, 0, 0, dyaw))
            k = self._get_k(roll, pitch, yaw, zen, azi, 0, 0, dyaw)  # with rayleight
            for i in range(len(self.channels)):
                raw['rad'][:, i] = raw['rad'][:, i] * k[i, :]
    raw.update({'E_ali': angerror})
    self.tab -= 1
    print(self.tab * '  |' + "correction done: raw data -> tc " + str(~self.noINS))
    return raw


def _get_k(self, roll, pitch, yaw, zen, azi, droll=0., dpitch=0., dyaw=0.):
    gamma = self._calc_viewzen(roll, pitch, yaw, zen, azi, droll, dpitch, dyaw)
    k = []
    X, Y = np.meshgrid(np.arange(90), np.arange(90))
    X = X.flatten()
    Y = Y.flatten()
    for i, wvl in enumerate(self.channels):
        C3 = np.load(self.fpath + "lookuptables/motioncorrection/" + "C3lookup_%.1f.npy" % (wvl))
        vals = C3.flatten()
        gamma[np.isnan(gamma) == True] = -1
        C = griddata((X, Y), vals, (zen, gamma))
        if len(k) == 0:
            k = C
        else:
            k = np.vstack((k, C))
    return k


def _calc_viewzen(self, roll, pitch, yaw, zen, azi, droll=0., dpitch=0., dyaw=0.):
    # calculate the angle between radiometer normal to sun position vektor
    c = np.pi / 180.
    r = roll * c + droll * c
    p = pitch * c + dpitch * c
    y = yaw * c + dyaw * c
    z = zen * c
    a = azi * c
    g = a - y
    coszen = np.sin(z) * np.sin(r) * np.sin(g) - np.sin(z) * np.sin(p) * np.cos(r) * np.cos(g) + np.cos(z) * np.cos(
        p) * np.cos(r)
    zenX = np.arccos(coszen) / c
    zenX[zenX >= 89] = np.nan
    return zenX  # [degrees] angle between radiometer normal and solar vector


processData
