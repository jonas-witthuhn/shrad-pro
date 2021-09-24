import configparser

import xarray as xr

config = configparser.ConfigParser()
config.read("../ConfigFile.ini")
pf = config['PATHS']['datasets']


def create_ins_dataset(Fname, Data):
    ds = xr.Dataset({'pitch': ('time', Data['pitch']),
                     'roll': ('time', Data['roll']),
                     'yaw': ('time', Data['yaw']),
                     'lat': ('time', Data['lat']),
                     'lon': ('time', Data['lon'])},
                    coords={'time': ('time', Data['time'])}
