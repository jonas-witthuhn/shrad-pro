import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import circmean
from scipy.spatial.transform import Rotation as R

# from modules import circular as circ
from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints
from modules.helpers import print_warning as printw


def circ_corrcoef(x, y, deg=True):
    """
    Calculate circular correlation coefficient
    """
    if deg:
        x = np.deg2rad(x)
        y = np.deg2rad(y)
    sx = np.sin(x-circmean(x))
    sy = np.sin(y-circmean(y))

    r = np.sum(sx*sy) / np.sqrt(np.sum(sx**2)*np.sum(sy**2))
    return r


def xyz2rp(xyz):
    """
    Calculate pitch and roll angle of a Cartesian vector assuming heading aligned
    coordinate system
      * ships bow points along x-axis
      * pitch -> left-hand rotation around y-axis-> positive if bow is up
      * roll -> right-hand rotation around x-axis -> positive if starboard is down
      * yaw -> left-hand rotation around z-axis -> positive if bow moves clockwise
                or positive from north (if x-axis points towards north)
      * Coordinate system:
        X: North or ships bow,
        Y: West or ships port-side
        Z: upward

    Parameters
    ----------
    xyz: list of len(3), or numpy.array of shape(N,3)

    Returns
    -------
    rp: numpy.array of shape (N,2)
        axis [:,0] roll angle [degrees] (positive if starboard is down)
        axis [:,1] pitch angle [degrees] (positive if bow is up)
    """
    xyz = np.array(xyz)
    if len(xyz.shape) == 1:
        xyz = xyz[np.newaxis, :]
    # ensure normalized vector
    xyz /= np.linalg.norm(xyz, axis=1)[:, np.newaxis]

    # calculate roll and pitch as seen from the ship
    r = np.arctan2(-xyz[:, 1], xyz[:, 2])
    p = np.arctan2(-xyz[:, 0], xyz[:, 2])
    r = np.rad2deg(r)
    p = np.rad2deg(p)
    return np.vstack([r, p]).T


def rpy2xyz(rpy, degrees=True):
    """
    Calculate Cartesian coordinates of ships normal vector (x=0,y=0,z=1) if rotated
    by the angles roll, pitch and yaw.
      * ships bow points along x-axis
      * pitch -> left-hand rotation around y-axis-> positive if bow is up
      * roll -> right-hand rotation around x-axis -> positive if starboard is down
      * yaw -> left-hand rotation around z-axis -> positive if bow moves clockwise
                or positive from north (if x-axis points towards north)
      * Coordinate system:
        X: North or ships bow,
        Y: West or ships port-side
        Z: upward

    Parameters
    ----------
    rpy: list/tuple of len(3) or numpy.array of shape(N,3)
        Roll, pitch, and yaw angles in degrees [roll, pitch, yaw]
        or np.array([[roll1, pitch1, yaw1], ..., [rollN, pitchN, yawN]])
    degrees: bool
        If True, the unit of angles in rpy is assumed to be degrees, else rad.
        The default is True.

    Returns
    -------
    xyz: np.array of shape (N,3)
        Cartesian coordinates x,y,z
    """

    # ensure shape
    rpy = np.array(rpy)
    if len(rpy.shape) == 1:
        rpy = rpy[np.newaxis, :]

    # initialize normal vector
    vector = np.array([0, 0, 1])

    # sort to match order of rotation
    ypr = rpy[:, [2, 1, 0]]

    # as rotation will be applied as right-hand rotation later,
    # invert yaw and pitch, to achieve left-hand rotation of the same.
    ypr = ypr * np.array([-1, -1, 1])

    # calculate right-hand euler rotation matrix
    r = R.from_euler("ZYX", ypr, degrees=degrees)

    # apply rotation and return
    return r.apply([0, 0, 1])


def estimate_guv2ins_misalignment(ds, dyaw_assume=None , verbose=True, debug=False, lvl=0):
    """
    Derive offset alignment angles of GUVis instrument setup on a platform. The INS data
    corresponds to the platform alignment angles. Offset roll and pitch are defined as
    seen from the platform (from the platform to the guvis normal vector).
    The yaw angle of GUVis is the angle positive clockwise from north, as seen from the
    GUVis accelerometer.
    Parameters
    ----------
    ds: xarray.Dataset
        Dataset have to include:
            * EsRoll: GUVis Accelerometer Roll
            * EsPitch: GUVis Accelerometer Pitch
            * InsRoll: INS Roll
            * InsPitch: INS Pitch
            * InsYaw: INS Yaw
    dyaw_assume: float or None
        Offset of Heading of instrument to ship ins. If None, this will be assumed, but
        might be uncertain.
    verbose: bool
        enable verbose mode, default is True.
    debug: bool
        enable debug messages, default is False.
    lvl: int
        intend level of verbose messages

    Returns
    -------
    ds: xarray.Dataset
        input dataset with added Offset roll and pitch, and instrument yaw angle
    (OffsetRoll, OffsetPitch): (float, float)
        Offset roll and pitch angles from platform to guvis normal vector. [degrees]
    Yaw_Guvis: float
        Yaw Angle of the GUVis accelerometer. Positive clockwise from north. [degrees]
    """

    def _test_yaw(yaw_test, xyz_ship, roll_guvis, pitch_guvis, degrees=True):
        rpy_guvis = np.vstack([roll_guvis,
                               pitch_guvis,
                               np.ones(len(roll_guvis)) * yaw_test]).T
        xyz_guvis = rpy2xyz(rpy_guvis, degrees=degrees)

        dot = np.sum(xyz_ship * xyz_guvis, axis=1)
        dangles = np.arccos(dot)
        return np.mean(dangles)

    if verbose:
        prints("Calculate Platform to GUVis offset ...", lvl=lvl)

    # calculate mean sampling frequency
    freq = 1e3 / (np.diff(ds.time.data)).astype('timedelta64[ms]').astype(int)
    mean_freq = np.mean(freq)

    # smooth out ripples using 2sec rolling mean
    drop_var = [key for key in ds.keys() if ((key[:2] != 'Es') and (key[:2] != 'In'))]
    ds = ds.drop_vars(drop_var)
    ds = ds.rolling(time=int(np.round(mean_freq, 0))*2, center=True).mean().dropna("time")

    # ship in heading aligned coordinate system
    xyz_ship = rpy2xyz(np.vstack([ds.InsRoll.data,
                                  ds.InsPitch.data,
                                  np.zeros(ds.time.size)]).T)

    # find yaw offset, between ship and guvis
    if type(dyaw_assume) == type(None):
        bounds = [0, 360]
    else:
        bounds = [dyaw_assume-10, dyaw_assume+10]
    res = minimize_scalar(_test_yaw,
                          bounds=bounds,
                          args=(xyz_ship, ds.EsRoll.data, ds.EsPitch.data, True),
                          method='bounded')

    yaw_guvis = float(res.x)
    if debug:
        printd(f"Offset Yaw of GUVis: {yaw_guvis:.3f}")

    # transform guvis pitch/roll to ship heading aligned coord. system
    xyzship_guvis = rpy2xyz(np.vstack([ds.EsRoll.data,
                                       ds.EsPitch.data,
                                       np.ones(ds.time.size)*yaw_guvis]).T)
    rp_guvis = xyz2rp(xyzship_guvis)

    # calculate misalignment roll and pitch angles between
    # INS and GUVis
    # For this, we compare adjusted platform roll and pitch to
    # the GUVis angles, but avoiding peaks of roll and pitch angles
    # as they are erroneous due to the influence of acceleration force
    # 1. step: Find time index between peeks of roll or pitch
    # (width of peaks is assumed minimum 1 second)
    roll_peaks, roll_peaks_res = find_peaks(ds.EsRoll.data, width=[mean_freq])
    pitch_peaks, pitch_peaks_res = find_peaks(ds.EsPitch.data, width=[mean_freq])
    roll_left_ips = np.round(roll_peaks_res['left_ips'], 0).astype(int)
    roll_right_ips = np.round(roll_peaks_res['right_ips'], 0).astype(int)
    pitch_left_ips = np.round(pitch_peaks_res['left_ips'], 0).astype(int)
    pitch_right_ips = np.round(pitch_peaks_res['right_ips'], 0).astype(int)
    idx_half_peak_roll = np.unique(np.concatenate((roll_left_ips,
                                                   roll_right_ips), axis=0))
    idx_half_peak_pitch = np.unique(np.concatenate((pitch_left_ips,
                                                    pitch_right_ips), axis=0))

    delta_roll = np.mean(rp_guvis[idx_half_peak_roll, 0] - ds.InsRoll.data[idx_half_peak_roll])
    delta_pitch = np.mean(rp_guvis[idx_half_peak_pitch, 1] - ds.InsPitch.data[idx_half_peak_pitch])
    delta_roll = float(delta_roll)
    delta_pitch = float(delta_pitch)

    if debug:
        printd(f"OffsetRoll: {delta_roll:.2f}")
        printd(f"OffsetPitch: {delta_pitch:.2f}")
    if verbose:
        prints("... done", lvl=lvl)
    return (delta_roll, delta_pitch), yaw_guvis


def calc_apparent_szen(rpy, sun_angles, drdpdy):
    """
    Calculate apparent solar zenith angle, which is the angle between the platform
    normal and the sun position vector.
      * pitch -> left-hand rotation around y-axis-> positive if bow is up
      * roll -> right-hand rotation around x-axis -> positive if starboard is down
      * yaw -> left-hand rotation around z-axis -> positive if bow moves clockwise
                or positive from north (if x-axis points towards north)

    Parameters
    ----------
    rpy:  tuple(3) or numpy.array(N,3)
        Roll, pitch, and yaw angle of the platform [degrees]
    sun_angles: tuple(2) or numpy.array(N,2)
        solar zenith and azimuth angle [degrees]
    drdpdy: tuple or numpy.array, shape as rpy
        Offset roll, pitch, and yaw angle of the radiometer on the platform [degrees]

    Returns
    -------

    """
    # platform angles
    rpy = np.deg2rad(np.array(rpy) + np.array(drdpdy))


    if len(rpy.shape) == 1:
        rpy = rpy[np.newaxis, :]
    r, p, y = rpy.T

    # sun angles
    sun_angles = np.deg2rad(np.array(sun_angles))
    if len(sun_angles.shape) == 1:
        sun_angles = sun_angles[np.newaxis, :]
    z, a = sun_angles.T

    # apparent azimuth
    g = a - y

    # apparent zenith
    coszen = np.sin(z) * np.sin(r) * np.sin(g) \
        - np.sin(z) * np.sin(p) * np.cos(r) * np.cos(g) \
        + np.cos(z) * np.cos(p) * np.cos(r)

    apparent_szen = np.rad2deg(np.arccos(coszen))
    apparent_szen[apparent_szen >= 89] = np.nan

    return apparent_szen  # [degrees] angle between radiometer normal and solar vector
