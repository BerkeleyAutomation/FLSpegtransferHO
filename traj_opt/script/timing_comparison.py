from FLSpegtransfer.motion.dvrkArm import dvrkArm
import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=20)     # fontsize of the axes title
# plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
# plt.rc('legend', fontsize=17)    # legend fontsize
# plt.rc('figure', titlesize=10)  # fontsize of the figure title


def plot_joint(t1, q1, q1_dot, q1_ddot, t2, q2, q2_dot, q2_ddot):
    # Create plot
    plt.figure(1)
    plt.title('Joint angle')
    ylabel = ['q1 (rad)', 'q2 (rad)', 'q3 (m)', 'q4 (rad)', 'q5 (rad)', 'q6 (rad)']
    for i in range(6):
        plt.subplot(610+i+1)
        plt.plot(t1, q1[:,i], 'b-', t2, q2[:,i], 'r-')
        plt.ylabel(ylabel[i])
    plt.xlabel('time (s)')

    plt.figure(2)
    plt.title('Joint velocity')
    ylabel = ['q1 (rad/s)', 'q2 (rad/s)', 'q3 (m/s)', 'q4 (rad/s)', 'q5 (rad/s)', 'q6 (rad/s)']
    for i in range(6):
        plt.subplot(610 + i + 1)
        plt.plot(t1, q1_dot[:, i], 'b-', t2, q2_dot[:, i], 'r-')
        plt.ylabel(ylabel[i])
    plt.xlabel('time (s)')

    plt.figure(3)
    plt.title('Joint acceleration')
    ylabel = ['q1 (rad/s^2)', 'q2 (rad/s^2)', 'q3 (m/s^2)', 'q4 (rad/s^2)', 'q5 (rad/s^2)', 'q6 (rad/s^2)']
    for i in range(6):
        plt.subplot(610 + i + 1)
        plt.plot(t1, q1_ddot[:, i], 'b-', t2, q2_ddot[:, i], 'r-')
        plt.ylabel(ylabel[i])
    plt.xlabel('time (s)')
    plt.show()


def calculate_dot(q_pos, dt):
    q_pos_prev = np.insert(q_pos, 0, q_pos[0], axis=0)
    q_pos_prev = np.delete(q_pos_prev, -1, axis=0)
    q_vel = (q_pos - q_pos_prev) / dt
    q_vel_prev = np.insert(q_vel, 0, q_vel[0], axis=0)
    q_vel_prev = np.delete(q_vel_prev, -1, axis=0)
    q_acc = (q_vel - q_vel_prev) / dt
    return q_vel, q_acc

dvrk = dvrkArm('/PSM1')
traj = np.load("traj.npy", allow_pickle=True)
traj_opt = np.load("traj_opt_cubic.npy", allow_pickle=True)

# # run motion (linear)
# for tt in traj:
#     for qs in tt:
#         dvrk.set_joint(qs[0], wait_callback=True)
#         for q in qs:
#             dvrk.set_joint(q, wait_callback=False)
#             time.sleep(0.01)
#         dvrk.set_joint(qs[-1], wait_callback=True)

# # run motion (optimized)
# for qs in traj_opt:
#     dvrk.set_joint(qs[0], wait_callback=True)
#     for q in qs:
#         dvrk.set_joint(q, wait_callback=False)
#         time.sleep(0.01)
#     dvrk.set_joint(qs[-1], wait_callback=True)

# plot
time_diff = 0.0
dt = 0.01
for qs, qs_opt in zip(traj, traj_opt):
    # for q in qs:
    #     print (len(q))
    qs = np.concatenate(qs)
    print (len(qs), len(qs_opt), round((len(qs)-len(qs_opt))*0.01, 2))
    time_diff += (len(qs)-len(qs_opt))*0.01

    # plot
    t = np.arange(start=0, stop=len(qs))*dt
    t_opt = np.arange(start=0, stop=len(qs_opt))*dt
    qs_vel, qs_acc = calculate_dot(qs, dt)
    qs_opt_vel, qs_opt_acc = calculate_dot(qs_opt, dt)
    # print (np.max(qs_vel, axis=0))
    # print (np.max(qs_acc, axis=0))
    # print (np.max(qs_opt_vel, axis=0))
    # print (np.max(qs_opt_acc, axis=0))
    # print ("")
    plot_joint(t, qs, qs_vel, qs_acc, t_opt, qs_opt, qs_opt_vel, qs_opt_acc)
    # plot_pose(t, ps, ps_vel, ps_acc, t_opt, ps_opt, ps_opt_vel, ps_opt_acc)
print (time_diff)