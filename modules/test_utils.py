from unittest import TestCase

import xarray as xr
import numpy as np
import pandas as pd

import trosat.sunpos as sp
from modules.utils import add_sun_position

class Test(TestCase):
    def setUp(self):
        times = pd.date_range('2014-04-01T00:00',
                              '2014-04-01T23:00',
                              freq='1H')
        times = times.to_numpy(dtype='datetime64')
        self.ds = xr.Dataset({}, coords={'time' : ('time', times)})
        self.test_coords = np.array([[45, -90],
                                     [45, 90],
                                     [-45, -90],
                                     [45, 90]])
    def test_add_sun_position(self):
        self.assertRaises(add_sun_position(self.ds),
                          ValueError,
                          'An Exception should be raised if no position information is available.')
        test_ds = self.ds.copy()

        test_ds.assign({'BioGpsLatitude' : ('time', [self.test_coords[0,0]]*test_ds.time.size )
        self.assertIsInstance()
