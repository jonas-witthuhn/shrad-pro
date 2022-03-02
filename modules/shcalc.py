import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from modules import circular as circ
from modules.helpers import print_debug as printd
from modules.helpers import print_status as prints
from modules.helpers import print_warning as printw


def rpy2xyz(rpy):
    """
    Calculate Cartesian coordinates of ships normal vector (x=0,y=0,z=1) if rotated
    by the angles roll, pitch and yaw.
      * ships bow points along x-axis
      * pitch -> right-hand rotation around y-axis-> positive if bow is down
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

    Returns
    -------
    xyz: np.array of shape (N,3)
        Cartesian coordinates x,y,z
    """
    rpy = np.deg2rad(np.array(rpy))
    if len(rpy.shape) == 1:
        rpy = rpy[np.newaxis, :]
    r, p, y = rpy.T
    x = np.sin(p) * np.cos(r) * np.cos(y) - np.sin(r) * np.sin(y)
    y = - np.sin(p) * np.cos(r) * np.sin(y) - np.sin(r) * np.cos(y)
    z = np.cos(p) * np.cos(r)
    xyz = np.vstack((x, y, z)).T
    xyz /= np.linalg.norm(xyz, axis=1)[:, np.newaxis]
    return xyz


def xyz2rp(xyz, dyaw=0):
    """
    Calculate pitch and yaw angle of a Cartesian vector and an azimuth offset
      * ships bow points along x-axis
      * pitch -> right-hand rotation around y-axis-> positive if bow is down
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
    dyaw: float
        yaw (azimuth) offset angle (clockwise from north or ships fore (bow)) [degrees],
        the default is 0.

    Returns
    -------
    r: numpy.array of shape (N,)
        roll angle [degrees] (positive if starboard is down)
    p: numpy.array of shape (N,)
        pitch angle [degrees] (positive if fore (bow) is down)
    """
    xyz = np.array(xyz)
    if len(xyz.shape) == 1:
        xyz = xyz[np.newaxis, :]
    # ensure normalized vector
    xyz /= np.linalg.norm(xyz, axis=1)[:, np.newaxis]
    x, y, z = xyz.T
    dyaw = np.deg2rad(dyaw)

    # transform coordinates so that ship is now on x-axis
    # (vector rotates counter clockwise from north)
    x1 = x * np.cos(dyaw) - y * np.sin(dyaw)
    y1 = x * np.sin(dyaw) + y * np.cos(dyaw)

    # calculate roll and pitch as seen from the ship
    r = np.arctan2(-y1, z)
    p = np.arctan2(x1, z)
    r = np.rad2deg(r)
    p = np.rad2deg(p)
    return r, p


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

    def _angle_correlation(a0, a1, b0, b1):
        """ Mean circular correlation of two angle pairs (a0,a1), (b0,b1) in [degrees]
        """
        #     print(a0,a1,b0,b1)
        a_test = circ.corrcoef(a0, a1)
        b_test = circ.corrcoef(b0, b1)
        return np.mean([1-a_test, 1-b_test])

    def _test_yaw(yaw_test, xyz_platform, roll_guvis, pitch_guvis):
        roll_test, pitch_test = xyz2rp(xyz_platform, yaw_test)
        ncorrelation = _angle_correlation(roll_test, roll_guvis,
                                          pitch_test, pitch_guvis)
        return ncorrelation

    if verbose:
        prints("Calculate Platform to GUVis offset ...", lvl=lvl)

    # smooth out ripples using 2sec rolling mean
    drop_var = [key for key in ds.keys() if ((key[:2] != 'Es') and (key[:2] != 'In'))]
    ds = ds.drop_vars(drop_var)
    ds = ds.rolling(time=15 * 2, center=True).mean().dropna("time")

    # roll, pitch, yaw of the ship:
    rpy_ship = np.vstack((ds.InsRoll.data,
                          ds.InsPitch.data,
                          ds.InsYaw.data)).T
    # ships normal in cartesian coordinates
    xyz_ship = rpy2xyz(rpy_ship)
    # find yaw offset, between ship and guvis
    if type(dyaw_assume) == type(None):
        bounds = [0, 360]
    else:
        bounds = [dyaw_assume-10, dyaw_assume+10]
    res = minimize_scalar(_test_yaw,
                          bounds=bounds,
                          args=(xyz_ship, ds.EsRoll.data, ds.EsPitch.data),
                          method='bounded')

    yaw_guvis = float(res.x)
    if debug:
        printd(f"Offset Yaw of GUVis: {yaw_guvis:.3f}")

    # calculate roll and pitch if ship has a yaw like guvis
    roll_platform, pitch_platform = xyz2rp(xyz_ship, yaw_guvis)

    # calculate misalignment roll and pitch angles between
    # INS and GUVis
    # For this, we compare adjusted platform roll and pitch to
    # the GUVis angles, but avoiding peaks of roll and pitch angles
    # as they are erroneous due to the influence of acceleration force
    # 1. step: Find time index between peeks of roll or pitch
    # (width of peaks is assumed minimum 1 second)
    freq = 1e3 / (np.diff(ds.time.data)).astype('timedelta64[ms]').astype(int)
    roll_peaks, roll_peaks_res = find_peaks(ds.EsRoll.data, width=[np.mean(freq)])
    pitch_peaks, pitch_peaks_res = find_peaks(ds.EsPitch.data, width=[np.mean(freq)])

    roll_left_ips = np.round(roll_peaks_res['left_ips'], 0).astype(int)
    roll_right_ips = np.round(roll_peaks_res['right_ips'], 0).astype(int)
    pitch_left_ips = np.round(pitch_peaks_res['left_ips'], 0).astype(int)
    pitch_right_ips = np.round(pitch_peaks_res['right_ips'], 0).astype(int)
    idx_half_peak_roll = np.unique(np.concatenate((roll_left_ips,
                                                   roll_right_ips), axis=0))
    idx_half_peak_pitch = np.unique(np.concatenate((pitch_left_ips,
                                                    pitch_right_ips), axis=0))
    delta_roll = np.mean(ds.EsRoll[idx_half_peak_roll] - roll_platform[idx_half_peak_roll])
    delta_pitch = np.mean(ds.EsPitch[idx_half_peak_pitch] - pitch_platform[idx_half_peak_pitch])
    delta_roll = float(delta_roll.data)
    delta_pitch = float(delta_pitch.data)

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
      * pitch -> right-hand rotation around y-axis-> positive if bow is down
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
