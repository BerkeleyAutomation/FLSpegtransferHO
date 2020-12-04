import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use('seaborn-whitegrid')


def plot_joint(q_des, q_act, q_est, show_window=True):
    q_des = np.array(q_des)
    q_act = np.array(q_act)
    q_est = np.array(q_est)

    RMSE = []
    MSE = []
    for i in range(6):
        RMSE.append(np.sqrt(np.sum((q_des[:,i] - q_act[:,i]) ** 2) / len(q_act[:,i])))
        # RMSE.append(np.sqrt(np.sum((q_act[:,i] - q_est[:, i]) ** 2) / len(q_act[:, i])))
        MSE.append(np.sum((q_act[:,i] - q_est[:,i]) ** 2) / len(q_act[:,i]))

    RMSE = [np.rad2deg(q) if i!=2 else q for i,q in enumerate(RMSE)]
    print("RMSE=", RMSE, "(deg or mm)")
    print("MSE=", MSE)

    # Create plot
    t = range(len(q_des))
    # plt.figure()
    ax = plt.subplot(611)
    plt.plot(t, np.rad2deg(q_des[:,0]), 'b-')
    plt.plot(t, np.rad2deg(q_act[:,0]), 'g-')
    plt.plot(t, np.rad2deg(q_est[:,0]), 'r-')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    # plt.ylim([35, 62])
    ax.set_xticklabels([])
    plt.ylabel('q0 ($^\circ$)')

    ax = plt.subplot(612)
    plt.plot(t, np.rad2deg(q_des[:, 1]), 'b-')
    plt.plot(t, np.rad2deg(q_act[:, 1]), 'g-')
    plt.plot(t, np.rad2deg(q_est[:, 1]), 'r-')
    # plt.ylim([-10, 12])
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(t, q_des[:, 2], 'b-')
    plt.plot(t, q_act[:, 2], 'g-')
    plt.plot(t, q_est[:, 2], 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(t, np.rad2deg(q_des[:, 3]), 'b-')
    plt.plot(t, np.rad2deg(q_act[:, 3]), 'g-')
    plt.plot(t, np.rad2deg(q_est[:, 3]), 'r-')
    # plt.ylim([-90, 70])
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(t, np.rad2deg(q_des[:, 4]), 'b-')
    plt.plot(t, np.rad2deg(q_act[:, 4]), 'g-')
    plt.plot(t, np.rad2deg(q_est[:, 4]), 'r-')
    # plt.ylim([-60, 60])
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(t, np.rad2deg(q_des[:, 5]), 'b-')
    plt.plot(t, np.rad2deg(q_act[:, 5]), 'g-')
    plt.plot(t, np.rad2deg(q_est[:, 5]), 'r-')
    # plt.ylim([-60, 60])
    # plt.legend(['desired', 'actual'])
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('sample number')
    if show_window:
        plt.show()

def plot_hysteresis(q_des, q_act, show_window=True):
    q_des = np.array(q_des)
    q_act = np.array(q_act)

    plt.figure()
    plt.subplot(231)
    plt.plot(np.rad2deg(q_des[:, 0]), np.rad2deg(q_act[:, 0]), 'b-')
    plt.title('q0')
    plt.xlabel('q$_{des}$ ($^\circ$)')
    plt.ylabel('q$_{act}$ ($^\circ$)')

    plt.subplot(232)
    plt.plot(np.rad2deg(q_des[:, 1]), np.rad2deg(q_act[:, 1]), 'b-')
    plt.title('q2')
    plt.xlabel('q$_{des}$ ($^\circ$)')
    plt.ylabel('q$_{act}$ ($^\circ$)')

    plt.subplot(233)
    plt.plot(q_des[:, 2], q_act[:, 2], 'b-')
    plt.title('q3')
    plt.xlabel('q$_{des}$ (mm)')
    plt.ylabel('q$_{act}$ (mm)')

    plt.subplot(234)
    plt.plot(np.rad2deg(q_des[:, 3]), np.rad2deg(q_act[:, 3]), 'b-')
    plt.title('q4')
    plt.xlabel('q$_{des}$ ($^\circ$)')
    plt.ylabel('q$_{act}$ ($^\circ$)')

    plt.subplot(235)
    plt.plot(np.rad2deg(q_des[:, 4]), np.rad2deg(q_act[:, 4]), 'b-')
    plt.title('q5')
    plt.xlabel('q$_{des}$ ($^\circ$)')
    plt.ylabel('q$_{act}$ ($^\circ$)')

    plt.subplot(236)
    plt.plot(np.rad2deg(q_des[:, 5]), np.rad2deg(q_act[:, 5]), 'b-')
    plt.title('q6')
    plt.xlabel('q$_{des}$ ($^\circ$)')
    plt.ylabel('q$_{act}$ ($^\circ$)')

    plt.xlim([-90, 90])
    plt.ylim([-90, 90])
    if show_window:
        plt.show()