import numpy as np
import matplotlib.pyplot as plt
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
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

# def ik_position(pos):  # (m)
#     x = pos[0]
#     y = pos[1]
#     z = pos[2]
#     L1 = 0.4318  # Rcc (m)
#     L2 = 0.4162  # tool
#     # L3 = 0.0091  # pitch ~ yaw (m)
#     # L4 = 0.0102  # yaw ~ tip (m)
#
#     # Inverse Kinematics
#     q0 = np.arctan2(x, -z)  # (rad)
#     q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
#     q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + L1 - L2  # (m)
#     return q0, q2, q3

def insertion_sampling(dvrk_model, insertion_number, dvrk_motion):
    pos_org = [0.080, 0.0, -0.095]
    rot_org = [0.0, 0.0, 0.0, 1.0]
    pos_min = [0.172, 0.036, -0.150]
    pos_max = [0.080, -0.051, -0.110]
    height_ready = -0.120
    height_block = [-0.155, -0.150]
    traj_pos = []
    traj_joint = []
    # dvrk_motion.set_jaw(jaw1=[0.0])
    for i in range(insertion_number):
        # pick
        pos_above = np.random.uniform(pos_min, pos_max)
        pos_above[2] = height_ready
        pos_grasp = [pos_above[0], pos_above[1], np.random.uniform(height_block[0], height_block[1])]

        # action 1
        traj_pos.append(pos_org)
        q1, q2, q3, q4, q5, q6 = dvrk_model.pose_to_joint(pos_org, rot_org)
        traj_joint.append([q1,q2,q3,q4,q5,q6])
        # dvrk_motion.set_joint(joint1=[q1,q2,q3,q4,q5,q6])

        # action 2
        traj_pos.append(pos_above)
        q4_rand = np.random.uniform(-np.pi / 2, np.pi / 2)
        rot_q = U.euler_to_quaternion([q4_rand, 0.0, 0.0])
        q1, q2, q3, q4, q5, q6 = dvrk_model.pose_to_joint(pos_above, rot_q)
        traj_joint.append([q1,q2,q3,q4,q5,q6])
        # dvrk_motion.set_joint(joint1=[q1, q2, q3, q4, q5, q6])

        # action 3
        traj_pos.append(pos_grasp)
        q1, q2, q3, q4, q5, q6 = dvrk_model.pose_to_joint(pos_grasp, rot_q)
        traj_joint.append([q1, q2, q3, q4, q5, q6])
        # dvrk_motion.set_joint(joint1=[q1, q2, q3, q4, q5, q6])

        # action 4
        traj_pos.append(pos_above)
        q1, q2, q3, q4, q5, q6 = dvrk_model.pose_to_joint(pos_above, rot_q)
        traj_joint.append([q1, q2, q3, q4, q5, q6])
        # dvrk_motion.set_joint(joint1=[q1, q2, q3, q4, q5, q6])

        # place
        pos_above = np.random.uniform(pos_min, pos_max)
        pos_above[2] = height_ready

        # action 5
        traj_pos.append(pos_above)
        rot_q = U.euler_to_quaternion([0.0, 0.0, 0.0])
        q1, q2, q3, q4, q5, q6 = dvrk_model.pose_to_joint(pos_above, rot_q)
        traj_joint.append([q1, q2, q3, q4, q5, q6])
        # dvrk_motion.set_joint(joint1=[q1, q2, q3, q4, q5, q6])
        print (i, "th insertion traj. sampled")
    return traj_pos, traj_joint

if __name__ == "__main__":
    dvrk_model = dvrkKinematics()
    dvrk_motion = dvrkDualArm()
    traj_pos, traj_joint = insertion_sampling(dvrk_model, 400, dvrk_motion)
    plot_position(traj_pos)
    plot_joint(traj_joint)
    np.save('insertion_sampled', traj_joint)