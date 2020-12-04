import numpy as np
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.deprecated.dvrkMotionBridgeP import dvrkMotionBridgeP

dvrk_model = dvrkKinematics()
dvrk_motion = dvrkMotionBridgeP()

root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
dir = 'experiment/0_trajectory_extraction/PSM1/'
filename = 'verification_peg_transfer.npy'
joint_traj = np.load(root+dir+filename)
j1 = joint_traj[:, 0]
j2 = joint_traj[:, 1]
j3 = joint_traj[:, 2]
j4 = joint_traj[:, 3]
j5 = joint_traj[:, 4]
j6 = joint_traj[:, 5]

joint_traj_new = []
for joints in joint_traj:
    # get pose of PSM1
    pos = dvrk_model.fk_position(joints=joints, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4)
    R = dvrk_model.fk_orientation(joints=joints)
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, -1] = pos
    T = np.matrix(T)

    # mirror around yz-plane
    T_mirror = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_new = T_mirror.dot(T).dot(T_mirror)

    # import pdb; pdb.set_trace()
    # new joint angles for PSM2
    joints_new = dvrk_model.inverse_kinematics(T_new)
    joint_traj_new.append(joints_new)
    # dvrk_motion.set_joint(joint1=joints, joint2=joints_new)
    print('index: ', len(joint_traj_new), '/', len(joint_traj))
    print('PSM1: ', joints)
    print('PSM2: ', np.array(joints_new))
    print(' ')

np.save(filename+"_new", joint_traj_new)
