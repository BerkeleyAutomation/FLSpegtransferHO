import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2
from FLSpegtransfer.path import *

bag = rosbag.Bag('/home/davinci/pycharmprojects/210405_cal_random_traj_PSM2.bag')
topics = ['/dvrk/PSM1/state_joint_current', '/dvrk/PSM2/state_joint_current']

joint_ang1 = []
joint_ang2 = []

# async extraction
for topic, msg, t in bag.read_messages(topics=topics):
    if topic == topics[0]:
        joint_ang1.append(list(msg.position))
    elif topic == topics[1]:
        joint_ang2.append(list(msg.position))
    print('length (joint): ', len(joint_ang1))
bag.close()

# post-process data
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
joint_ang1 = np.array(joint_ang1)[::8]
joint_ang2 = np.array(joint_ang2)[::8]
joint_ang1 = joint_ang1[100:]
joint_ang2 = joint_ang2[100:]
pose1 = []
pose2 = []
joint_ang1_new = []
joint_ang2_new = []
for q in joint_ang1:
    pos, rot = dvrkKinematics.joint_to_pose(q)
    pos[2] += 0.005
    joint_ang1_new.append(dvrkKinematics.pose_to_joint(pos, rot)[0])

for q in joint_ang2:
    pos, rot = dvrkKinematics.joint_to_pose(q)
    pos[2] += 0.005
    joint_ang2_new.append(dvrkKinematics.pose_to_joint(pos, rot)[0])

print (len(joint_ang1_new))
print (len(joint_ang2_new))
# np.save('210405_cal_random_traj_PSM1', joint_ang1_new)
np.save('210405_cal_random_traj_PSM2', joint_ang2_new)
