import sys
sys.path.insert(0, '/home/davinci/pycharmprojects')
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.path import *
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
np.set_printoptions(formatter={'float_kind': "{:.3f}".format})

plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title


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
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
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


def pose_estimation(pbr, pbg, pbb, pby, use_Trc, which_arm):    # Find tool position, joint angles
    pt = []
    q_phy = []
    if len(pbr) < 2:
        pass
    else:
        pt = bd.find_tool_pitch(pbr[0], pbr[1])  # tool position of pitch axis
        pt = np.array(pt) * 0.001  # (m)
        if use_Trc:
            pt = bd.Rrc.dot(pt) + bd.trc
            qp1, qp2, qp3 = dvrkKinematics.ik_position_straight(pt, L3=0, L4=0)     # position of pitch axis

            # Find tool orientation, joint angles, and overlay
            temp = [pbr[2], pbg, pbb, pby]
            if len(pbr) < 3:
                qp4 = 0.0; qp5 = 0.0; qp6 = 0.0
            elif temp.count([]) >= 2:
                qp4=0.0; qp5=0.0; qp6=0.0
            else:
                Rm = bd.find_tool_orientation(pbr[2], pbg, pbb, pby, which_arm)  # orientation of the marker
                qp4, qp5, qp6 = dvrkKinematics.ik_orientation(qp1, qp2, Rm)
            q_phy = [qp1, qp2, qp3, qp4, qp5, qp6]
        else:
            q_phy = []
    return pt, q_phy


def plot_position(pos_des):
    # plot trajectory of des & act position
    pos_des = np.array(pos_des)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b.--')
    # plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r.-')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # plt.title('Trajectory of tool position')
    plt.legend(['desired', 'actual'])
    plt.show()

def ik_position(pos):  # (m)
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Inverse Kinematics
    q1 = np.arctan2(x, -z)  # (rad)
    q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
    q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + dvrkVar.L1 - dvrkVar.L2  # (m)
    return q1, q2, q3

def random_sampling(sample_number, pos_min, pos_max):
    q_target = []
    pos_target = []
    for i in range(sample_number):
        pos_rand = np.random.uniform(pos_min, pos_max)
        pos_target.append(pos_rand)
        q1, q2, q3 = ik_position(pos_rand)
        q_target.append([q1, q2, q3, 0.0, 0.0, 0.0])
    return q_target, pos_target

# configurations
use_Trc = False
which_camera = 'inclined'
which_arm = 'PSM2'

# define objects
zivid = ZividCapture(which_camera=which_camera)
zivid.start()
dvrk = dvrkArm('/'+which_arm)
if use_Trc:
    Trc = np.load(root+'calibration_files/Trc_' + which_camera + '_' + which_arm + '.npy')
else:
    Trc = []
Tpc = np.load(root+'calibration_files/Tpc_' + which_camera + '.npy')
bd = BallDetectionRGBD(Trc=Trc, Tpc=Tpc, which_camera=which_camera)

# Load & plot trajectory
if use_Trc:
    # filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_insertion_sampled_2000.npy'
    filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_random.npy'
    # filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/210404_random_traj_PSM2.npy'
    joint_traj = np.load(filename)
else:
    # filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/short_random.npy'
    #filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/random_sampled_500.npy'
    #filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/210405_cal_random_traj_PSM2.npy'
    if which_arm == 'PSM1':     # workspace of new conf.
        pos_min = [0.076, 0.094, -0.124]
        pos_max = [0.014, 0.038, -0.107]
    elif which_arm == 'PSM2':   # workspace of new conf.
        pos_min = [-0.029, 0.044, -0.102]
        pos_max = [-0.090, 0.093, -0.141]
    joint_traj, _ = random_sampling(100, pos_min, pos_max)

# joint_traj = joint_traj[100:]
# joint_traj = joint_traj[::2]
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
try:
    for q_cmd in joint_traj:
        if q_cmd[2] < 0.103:
            continue
        if use_Trc:
            pass
        else:
            q_cmd[3] = 0.0
            q_cmd[4] = 0.0
            q_cmd[5] = 0.0
        dvrk.set_jaw_interpolate(jaw=np.deg2rad([30]))
        dvrk.set_joint_interpolate(joint=q_cmd)
        time.sleep(0.3)

        # Capture image from Zivid
        color, _, point = zivid.capture_3Dimage(color='BGR')

        # Find balls
        if use_Trc:
            pbr = bd.find_balls(color, point, 'red', nb_sphere=3, visualize=False)
            pbg = bd.find_balls(color, point, 'green', nb_sphere=1, visualize=False)
            pbb = bd.find_balls(color, point, 'blue', nb_sphere=1, visualize=False)
            pby = bd.find_balls(color, point, 'yellow', nb_sphere=1, visualize=False)
        else:
            pbr = bd.find_balls(color, point, 'red', nb_sphere=2, visualize=False)
            pbg = []
            pbb = []
            pby = []

        # pose estimation
        pt, q_phy = pose_estimation(pbr, pbg, pbb, pby, use_Trc, which_arm)

        # overlay
        color = bd.overlay_ball(color, pbr)
        color = bd.overlay_ball(color, [pbg])
        color = bd.overlay_ball(color, [pbb])
        color = bd.overlay_ball(color, [pby])
        color = bd.overlay_dot(color, pt, 'pitch')
        if use_Trc:
            color = bd.overlay_tool(color, q_phy, (0, 0, 255))  # measured on red
            color = bd.overlay_tool(color, q_cmd, (0, 255, 0))  # commanded on green

        # Append data pairs
        if use_Trc:
            # joint angles
            q_cmd_.append(q_cmd)
            q_phy_.append(q_phy)
            time_stamp.append(time.time() - time_st)
            print('index: ', len(q_cmd_),'/',len(joint_traj))
            print('t_stamp: ', time.time() - time_st)
            print('q_cmd: ', np.array(q_cmd))
            print('q_phy: ', np.array(q_phy))
            print(' ')
        else:
            if len(pt) != 0:
                # positions of pitch axis
                pos_cmd_temp = dvrkKinematics.fk([q_cmd[0], q_cmd[1], q_cmd[2], 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0)[:3,-1]
                pos_cmd_.append(pos_cmd_temp)
                pos_phy_.append(pt)
                print('index: ', len(pos_cmd_), '/', len(joint_traj))
                print('pos_cmd: ', pos_cmd_temp)
                print('pos_phy: ', pt)
                print(' ')

        # Visualize
        scale = 0.9
        w = int(color.shape[1] * scale)
        h = int(color.shape[0] * scale)
        dim = (w, h)
        color = cv2.resize(color, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("", color)
        cv2.waitKey(1) & 0xFF
        # cv2.waitKey(0)

finally:
    # Save data to a file
    if use_Trc:
        np.save('q_cmd', q_cmd_)
        np.save('q_phy', q_phy_)
    else:
        # Get transform from robot to camera
        np.save('pos_cmd', pos_cmd_)
        np.save('pos_phy', pos_phy_)
        T = U.get_rigid_transform(np.array(pos_phy_), np.array(pos_cmd_))
        np.save('Trc_' + which_camera + '_' + which_arm, T)
    np.save('t_stamp', time_stamp)
    print("Data is successfully saved")
