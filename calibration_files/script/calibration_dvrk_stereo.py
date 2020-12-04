from FLSpegtransfer.path import *
from FLSpegtransfer.vision.BallDetectionStereo import BallDetectionStereo
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.utils.ImgUtils import ImgUtils
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2


def plot_trajectory(joint, pos):
    joint = np.array(joint)
    pos  = np.array(pos)
    q1 = joint[:, 0]
    q2 = joint[:, 1]
    q3 = joint[:, 2]
    q4 = joint[:, 3]
    q5 = joint[:, 4]
    q6 = joint[:, 5]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b.-')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    print('data length: ', len(joint))
    plt.show()

    # Create 2D plot for joint angles
    plt.subplot(611)
    plt.plot(q1 * 180. / np.pi, 'b-')
    plt.ylabel('q0 ($^\circ$)')
    plt.subplot(612)
    plt.plot(q2 * 180. / np.pi, 'b-')
    plt.ylabel('q2 ($^\circ$)')
    plt.subplot(613)
    plt.plot(q3, 'b-')
    plt.ylabel('q3 (mm)')
    plt.subplot(614)
    plt.plot(q4 * 180. / np.pi, 'b-')
    plt.ylabel('q4 ($^\circ$)')
    plt.subplot(615)
    plt.plot(q5 * 180. / np.pi, 'b-')
    plt.ylabel('q5 ($^\circ$)')
    plt.subplot(616)
    plt.plot(q6 * 180. / np.pi, 'b-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(step)')
    plt.show()


def pose_estimation(pbr, pbg, pbb, pby, use_Trc):    # Find tool position, joint angles
    pt = []
    q_phy = []
    if len(pbr) < 2:
        pass
    else:
        pt = bd.find_tool_pitch(pbr[0], pbr[1])  # tool position of pitch axis
        pt = np.array(pt)  # (m)
        if use_Trc:
            pt = bd.Rrc.dot(pt) + bd.trc
            qp1, qp2, qp3 = dvrkKinematics.ik_position_straight(pt, L3=0, L4=0)     # position of pitch axis

            # Find tool orientation, joint angles, and overlay
            temp = [pbr[2], pbg, pbb, pby]
            if len(pbr) < 3:
                qp4 = 0.0; qp5 = 0.0; qp6 = 0.0
            elif temp.count([]) > 2:
                qp4=0.0; qp5=0.0; qp6=0.0
            else:
                Rm = bd.find_tool_orientation(pbr[2], pbg, pbb, pby)  # orientation of the marker
                qp4, qp5, qp6 = dvrkKinematics.ik_orientation(qp1, qp2, Rm)
            q_phy = [qp1, qp2, qp3, qp4, qp5, qp6]
        else:
            q_phy = []
    return pt, q_phy

# configurations
use_Trc = False
which_arm = 'PSM1'

# define instances
av = AlliedVisionCapture()
dvrk = dvrkDualArm()
if use_Trc:
    Trc = np.load(root+'calibration_files/Trc_stereo.npy')
else:
    Trc = []
bd = BallDetectionStereo(Trc=Trc)

# Load & plot trajectory
filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/short_random_stereo.npy'
joint_traj = np.load(filename)
pos_traj = []
for q in joint_traj:
    fk_pos = dvrkKinematics.fk([q[0], q[1], q[2], 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0)
    fk_pos = np.array(fk_pos)[:3,-1]
    pos_traj.append(fk_pos)
plot_trajectory(joint_traj, pos_traj)

# collect data
time_st = time.time()   # (sec)
time_stamp = []
q_cmd_ = []
q_phy_ = []
pos_cmd_ = []
pos_phy_ = []
pt = []
for q_cmd in joint_traj:
    if use_Trc:
        pass
    else:
        q_cmd[3] = 0.0
        q_cmd[4] = 0.0
        q_cmd[5] = 0.0
    if which_arm == 'PSM1':
        dvrk.set_joint(joint1=q_cmd)
        dvrk.set_jaw(jaw1=[0.0])
    elif which_arm == 'PSM2':
        dvrk.set_joint(joint2=q_cmd)
        dvrk.set_jaw(jaw2=[0.0])

    # Capture image from stereo cameras
    img_left, img_right = av.capture(which='rectified')

    # Find balls
    pbr = bd.find_balls3D(img_left, img_right, color='red', visualize=False)
    # pbg = bd.find_balls3D(img_left, img_right, color='green', visualize=False)
    pbg = []; pbb = []; pby = []

    # pose estimation
    pt, q_phy = pose_estimation(pbr, pbg, pbb, pby, use_Trc)

    # Overlay
    img_left = bd.overlay_ball(img_left, pbr, which='left')
    img_right = bd.overlay_ball(img_right, pbr, which='right')
    img_left = bd.overlay_dot(img_left, pt, text='pitch', which='left')
    img_right = bd.overlay_dot(img_right, pt, text='pitch', which='right')
    if use_Trc:
        img_left = bd.overlay_tool(img_left, q_phy, (0, 255, 0))
        img_right = bd.overlay_tool(img_right, q_phy, (0, 255, 0))

    # Append data pairs
    if use_Trc:
        # joint angles
        q_cmd_.append(q_cmd)
        q_phy_.append(q_phy)
        time_stamp.append(time.time() - time_st)
        print('index: ', len(q_cmd),'/',len(joint_traj))
        print('t_stamp: ', time.time() - time_st)
        print('q_cmd: ', q_cmd)
        print('q_phy: ', q_phy)
        print(' ')
    else:
        # positions of pitch axis
        pos_cmd_temp = dvrkKinematics.fk([q_cmd[0], q_cmd[1], q_cmd[2], 0, 0, 0],L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0)[:3,-1]
        pos_cmd_.append(pos_cmd_temp)
        pos_phy_.append(pt)
        print('index: ', len(pos_cmd_), '/', len(joint_traj))
        print('pos_des: ', pos_cmd_temp)
        print('pos_act: ', pt)
        print(' ')

    # Visualize
    img_stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
    cv2.imshow("spheres_detected", img_stacked)
    cv2.waitKey(1) & 0xFF
    # cv2.waitKey(0)


# Save data to a file
if use_Trc:
    np.save('q_cmd_raw', q_cmd_)
    np.save('q_phy_raw', q_phy_)
else:
    # Get transform from robot to camera
    np.save('pos_cmd', pos_cmd_)
    np.save('pos_phy', pos_phy_)
    T = U.get_rigid_transform(np.array(pos_phy_), np.array(pos_cmd_))
    np.save('Trc', T)
np.save('t_stamp_raw', time_stamp)
print("Data is successfully saved")