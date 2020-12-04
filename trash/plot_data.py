import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use('seaborn-whitegrid')
plt.rc('legend', fontsize=13)    # legend fontsize

def plot_joint(q_des1, q_act1, q_des2, q_act2):
    q_des1 = np.array(q_des1)
    q_act1 = np.array(q_act1)
    q_des2 = np.array(q_des2)
    q_act2 = np.array(q_act2)

    RMSE1 = []
    RMSE2 = []
    for i in range(6):
        RMSE1.append(np.sqrt(np.sum((q_des1[:,i] - q_act1[:,i]) ** 2) / len(q_des1[:,i])))
        RMSE2.append(np.sqrt(np.sum((q_des2[:,i] - q_act2[:,i]) ** 2) / len(q_des2[:,i])))

    RMSE1 = [np.rad2deg(q) if i!=2 else q for i,q in enumerate(RMSE1)]
    RMSE2 = [np.rad2deg(q) if i != 2 else q for i, q in enumerate(RMSE2)]
    print("RMSE1=", RMSE1, "(deg or mm)")
    print("RMSE2=", RMSE2, "(deg or mm)")

    # Create plot
    t1 = range(len(q_des1))
    t2 = range(len(q_des2))
    plt.figure()

    ax = plt.subplot(611)
    plt.plot(t1, np.rad2deg(q_des1[:,0]), 'b-')
    plt.plot(t1, np.rad2deg(q_act1[:,0]), 'g-')
    plt.plot(t2, np.rad2deg(q_act2[:,0]), 'r-')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    # plt.ylim([35, 62])
    ax.set_xticklabels([])
    plt.ylabel('q0 ($^\circ$)')
    plt.legend(['q_des', 'q_grey', 'q_red'], ncol=3, bbox_to_anchor=(0.5, 1), loc="lower center")

    ax = plt.subplot(612)
    plt.plot(t1, np.rad2deg(q_des1[:, 1]), 'b-')
    plt.plot(t1, np.rad2deg(q_act1[:, 1]), 'g-')
    plt.plot(t2, np.rad2deg(q_act2[:, 1]), 'r-')
    # plt.ylim([-10, 12])
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(t1, q_des1[:, 2], 'b-')
    plt.plot(t1, q_act1[:, 2], 'g-')
    plt.plot(t2, q_act2[:, 2], 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(t1, np.rad2deg(q_des1[:, 3]), 'b-')
    plt.plot(t1, np.rad2deg(q_act1[:, 3]), 'g-')
    plt.plot(t2, np.rad2deg(q_act2[:, 3]), 'r-')
    # plt.ylim([-90, 70])
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(t1, np.rad2deg(q_des1[:, 4]), 'b-')
    plt.plot(t1, np.rad2deg(q_act1[:, 4]), 'g-')
    plt.plot(t2, np.rad2deg(q_act2[:, 4]), 'r-')
    # plt.ylim([-60, 60])
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(t1, np.rad2deg(q_des1[:, 5]), 'b-')
    plt.plot(t1, np.rad2deg(q_act1[:, 5]), 'g-')
    plt.plot(t2, np.rad2deg(q_act2[:, 5]), 'r-')
    # plt.ylim([-60, 60])
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('sample number')
    plt.show()


# Trajectory load for training data
q_des_grey = np.load('training/random_user/q_des_grey.npy')    # desired joint angles: [q0, ..., q6]
q_act_grey = np.load('training/random_user/q_act_grey.npy')    # actual joint angles: [q0, ..., q6]
q_des_red = np.load('training/random_user/q_des_red.npy')    # desired joint angles: [q0, ..., q6]
q_act_red = np.load('training/random_user/q_act_red.npy')    # actual joint angles: [q0, ..., q6]
print('data length: ', len(q_des_red))
plot_joint(q_des_grey, q_act_grey, q_des_red, q_act_red)

q_des_grey = np.load('training/peg_user/q_des_grey.npy')    # desired joint angles: [q0, ..., q6]
q_act_grey = np.load('training/peg_user/q_act_grey.npy')    # actual joint angles: [q0, ..., q6]
q_des_red = np.load('training/peg_user/q_des_red.npy')    # desired joint angles: [q0, ..., q6]
q_act_red = np.load('training/peg_user/q_act_red.npy')    # actual joint angles: [q0, ..., q6]
print('data length: ', len(q_des_red))
plot_joint(q_des_grey, q_act_grey, q_des_red, q_act_red)
