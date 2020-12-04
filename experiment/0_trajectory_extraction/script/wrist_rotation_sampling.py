import numpy as np
import matplotlib.pyplot as plt
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.utils.CmnUtil as U


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

def plot_joint(q_des):
    q_des = np.array(q_des)
    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(611)
    plt.plot(q_des[:,0]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 0] * 180. / np.pi, 'r-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    ax.set_xticklabels([])
    plt.ylabel('q0 ($^\circ$)')

    ax = plt.subplot(612)
    plt.plot(q_des[:,1]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 1] * 180. / np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(q_des[:, 2], 'b-')
    # plt.plot(q_act[:, 2], 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(q_des[:, 3]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 3]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(q_des[:, 4]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 4]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(q_des[:, 5]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 5]*180./np.pi, 'r-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(sample)')
    plt.show()

def ik_position(pos):  # (m)
    x = pos[0]
    y = pos[1]
    z = pos[2]
    L1 = 0.4318  # Rcc (m)
    L2 = 0.4162  # tool
    # L3 = 0.0091  # pitch ~ yaw (m)
    # L4 = 0.0102  # yaw ~ tip (m)

    # Inverse Kinematics
    q1 = np.arctan2(x, -z)  # (rad)
    q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
    q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + L1 - L2  # (m)
    return q1, q2, q3

def sampling(nb_pos, nb_rot):
    pos_min = [0.180, 0.017, -0.150]    # workspace for peg transfer
    pos_max = [0.087, -0.040, -0.117]
    pos_traj = []
    joint_traj = []
    roll_st = 0.0
    for i in range(nb_pos):
        pos_rand = np.random.uniform(pos_min, pos_max)
        for j in range(nb_rot):
            # select random roll angle
            roll_ed = np.random.uniform(-90, 90)
            if roll_st > roll_ed:
                roll = np.arange(roll_st, roll_ed, step=-10)
            else:
                roll = np.arange(roll_st, roll_ed, step=10)
            rot = [[r, 0.0, 0.0] for r in roll]
            quat = [U.euler_to_quaternion(r, 'deg') for r in rot]
            jaw = [0 * np.pi / 180.]
            for q in quat:
                joints = dvrk_model.pose_to_joint(pos_rand, q)
                # dvrk.set_joint(joint1=joints, jaw1=jaw)
                joints_deg = [joint * 180. / np.pi if i != 2 else joint for i, joint in enumerate(joints)]
                print(i, roll_ed, pos_rand, joints_deg)
                joint_traj.append(joints)
                pos_traj.append(pos_rand)
    return joint_traj, pos_traj

# dvrk = dvrkMotionBridgeP()
dvrk_model = dvrkKinematics()
joint_traj, pos_traj = sampling(nb_pos=60, nb_rot=5)
plot_position(pos_traj)
plot_joint(joint_traj)
np.save('joint_sampled', joint_traj)
print (len(joint_traj), " sampled trajectories saved.")