import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
# from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar


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

# define objects
use_rc_transform = False
dvrk = dvrkDualArm()
zivid = ZividCapture()
zivid.start()
bd = BallDetectionRGBD(use_rc_transform=False)

# Load & plot trajectory
root = '/home/davinci/pycharmprojects/FLSpegtransfer/'
filename = root + 'experiment/0_trajectory_extraction/PSM1/short_random.npy'
# filename = root + 'experiment/0_trajectory_extraction/training_peg_transfer_400_sampled.npy'
joint_traj = np.load(filename)
pos_traj = []
for q in joint_traj:
    fk_pos = dvrkKinematics.fk([q[0], q[1], q[2], 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0)
    fk_pos = np.array(fk_pos)[:3,-1]
    pos_traj.append(fk_pos)
plot_trajectory(joint_traj, pos_traj)
j1 = joint_traj[:, 0]
j2 = joint_traj[:, 1]
j3 = joint_traj[:, 2]
j4 = joint_traj[:, 3]
j5 = joint_traj[:, 4]
j6 = joint_traj[:, 5]

# collect data
q_msd = []
pos_msd = []
dvrk.set_jaw(jaw1=[0.*np.pi/180.])
assert len(j1)==len(j2)==len(j3)==len(j4)==len(j5)==len(j6)
for i,(qd1,qd2,qd3,qd4,qd5,qd6) in enumerate(zip(j1,j2,j3,j4,j5,j6)):
    # Set joint
    jaw1 = [0. * np.pi / 180.]
    joint1 = [qd1, qd2, qd3, qd4, qd5, qd6]
    dvrk.set_joint(joint1=joint1)
    dvrk.set_jaw(jaw1=jaw1)
    time.sleep(0.3)

    # Capture image from Zivid
    color, _, point = zivid.capture_3Dimage(color='BGR')
    color_org = np.copy(color)

    # # Find balls
    # pbr = bd.find_balls(color, point, 'red')
    # pbg = bd.find_balls(color, point, 'green')
    # pbb = bd.find_balls(color, point, 'blue')
    # pby = bd.find_balls(color, point, 'yellow')
    #
    # # overlay
    # color = bd.overlay_ball(color, pbr)
    # color = bd.overlay_ball(color, [pbg])
    # color = bd.overlay_ball(color, [pbb])
    # color = bd.overlay_ball(color, [pby])
    #
    # # Find tool position, joint angles, and overlay
    # if pbr[0] == [] or pbr[1] == []:
    #     qa1=0.0; qa2=0.0; qa3=0.0; qa4=0.0; qa5=0.0; qa6=0.0
    # else:
    #     pt = bd.find_tool_position(pbr[0], pbr[1])  # tool position of pitch axis
    #     pt = np.array(pt) * 0.001  # (m)
    #     if use_rc_transform:
    #         pt = bd.Rrc.dot(pt) + bd.trc
    #         qa1, qa2, qa3 = dvrkKinematics.ik_position(pt)
    #
    #         # Find tool orientation, joint angles, and overlay
    #         if len(pbr) < 3:
    #             qa4 = 0.0; qa5 = 0.0; qa6 = 0.0
    #         elif [pbr[2], pbg, pbb, pby].count([]) > 1:
    #             qa4=0.0; qa5=0.0; qa6=0.0
    #         else:
    #             Rm = bd.find_tool_orientation(pbr[2], pbg, pbb, pby)  # orientation of the marker
    #             qa4, qa5, qa6 = dvrkKinematics.ik_orientation(qa1, qa2, Rm)
    #             color = bd.overlay_tool(color, [qa1, qa2, qa3, qa4, qa5, qa6], (0, 255, 0))

    # Append data pairs
    # joint angles
    joint1 = dvrk.arm1.get_current_joint(wait_callback=True)
    pose1 = dvrk.arm1.get_current_pose(wait_callback=True)
    q_msd.append(joint1)
    pos_msd.append(pose1)
    cv2.imwrite("../../images/color"+str(i)+".png", color)
    np.save("../../images/point"+str(i), point)

    print('index: ', len(q_msd),'/',len(j1))
    print('joint: ', joint1)
    print('pose: ', pose1)
    print(' ')

    # Visualize
    cv2.imshow("images", color)
    cv2.waitKey(1) & 0xFF
    # cv2.waitKey(0)

# Save data to a file
np.save('joint', q_msd)
np.save('pose', pos_msd)
print("Data is successfully saved")