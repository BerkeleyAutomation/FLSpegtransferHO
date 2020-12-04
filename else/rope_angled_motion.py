from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np

# ZYZ euler angle to quaternion
def inclined_orientation(axis_rot, latitude, longitude=0):
    theta_z1 = longitude
    theta_y = latitude
    theta_z2 = axis_rot
    R = U.Rz(theta_z1).dot(U.Ry(theta_y)).dot(U.Rz(theta_z2))
    return U.R_to_quaternion(R)

dvrk = dvrkDualArm()
pos1 = [0.0, 0.0, -0.14]
rot1 = [30, 40, 0]
quat1 = inclined_orientation(axis_rot=np.deg2rad(20), latitude=np.deg2rad(-30))
dvrk.set_pose(pos1=pos1, rot1=quat1)
