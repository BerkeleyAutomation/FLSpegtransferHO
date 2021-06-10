import numpy as np
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.path import *

dvrk_model = dvrkKinematics()

dir = 'experiment/0_trajectory_extraction/PSM1/'
filename = '210404_random_traj_PSM1.npy'
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
    T = dvrk_model.fk(joints=joints, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4)

    # mirror around yz-plane
    T_mirror = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_new = T_mirror.dot(T).dot(T_mirror)

    # new joint angles for PSM2
    joints_new = dvrk_model.ik(T_new)[0]
    joint_traj_new.append(joints_new)
    print('index: ', len(joint_traj_new), '/', len(joint_traj))
    print('PSM1: ', joints)
    print('PSM2: ', np.array(joints_new))
    print(' ')

np.save(filename+"_new", joint_traj_new)
