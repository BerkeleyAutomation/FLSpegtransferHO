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


from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.traj_opt.CubicOptimizer_2wp import CubicOptimizer_2wp
from FLSpegtransfer.traj_opt.PegMotionOptimizer import PegMotionOptimizerV2b
from FLSpegtransfer.traj_opt.SQPMotionOptimizer import MTSQPMotionOptimizer
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics

# instances
# dvrk = dvrkArm('/PSM1')
cubic_opt = CubicOptimizer_2wp(max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, t_step=0.01, print_out=False, visualize=False)
qp_opt = PegMotionOptimizerV2b(max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, t_step=0.01, minimize='j', max_jerk=dvrkVar.a_max*15)
sqp_opt = MTSQPMotionOptimizer(dim=6, H=12, t_step=0.1, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, objective='v') # max_jerk=dvrkVar.a_max*15, objective='v')
# load waypoints
waypoints = np.load("ref_waypoints.npy")

# calculate traj & recording
traj_cubic = []
traj_qp = []
traj_sqp = []
len_cubic = []
len_qp = []
len_sqp = []
for wp in waypoints:
    q0, qw1, qw2, qf = wp

    # cubic-based trajectory
    q_pos_cubic, _ = cubic_opt.optimize(q0, [qw1], [qw2], [qf])
    traj_cubic.append(q_pos_cubic)
    len_cubic.append(len(q_pos_cubic))
    print ("len(cubic_opt)=", len(q_pos_cubic))

    # qp trajectory
    x, H = qp_opt.optimize_motion(*wp)
    dim = 6
    q_pos_qp = np.array([x[t * dim:(t + 1) * dim] for t in range(H + 1)])
    traj_qp.append(q_pos_qp)
    len_qp.append(len(q_pos_qp))
    print("len(qp_opt)=", len(q_pos_qp))
    x, H = sqp_opt.optimize_motion(*wp)
    q_pos_sqp = np.array([x[t * dim:(t + 1) * dim] for t in range(H + 1)])
    traj_sqp.append(q_pos_sqp)
    len_sqp.append(len(q_pos_sqp))
    print("len(sqp_opt)=", H)
    print("")

np.save("traj_opt_cubic", traj_cubic)
np.save("traj_opt_qp", traj_qp)
np.save("len_cubic", len_cubic)
np.save("len_qp", len_qp)

time_diff = np.array(len_cubic) - np.array(len_sqp)  # unit=10(ms)
print(time_diff)
print(np.sum(time_diff))

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

# # plot
# time_diff = 0.0
# dt = 0.01
# time_transfer = []
# time_transfer_opt = []
# for qs, qs_opt in zip(traj, traj_opt):
#     # for q in qs:
#     #     print (len(q))
#     time_transfer.append(len(qs) * 0.01)
#     time_transfer_opt.append(len(qs_opt)*0.01)
#     print ("Transfer time: ", len(qs)*0.01, len(qs_opt)*0.01)
#
#     # plot
#     t = np.arange(start=0, stop=len(qs))*dt
#     t_opt = np.arange(start=0, stop=len(qs_opt))*dt
#     qs_vel, qs_acc = calculate_dot(qs, dt)
#     qs_opt_vel, qs_opt_acc = calculate_dot(qs_opt, dt)
#     # print (np.max(qs_vel, axis=0))
#     # print (np.max(qs_acc, axis=0))
#     # print (np.max(qs_opt_vel, axis=0))
#     # print (np.max(qs_opt_acc, axis=0))
#     # print ("")
#     # plot_joint(t, qs, qs_vel, qs_acc, t_opt, qs_opt, qs_opt_vel, qs_opt_acc)
#     # plot_pose(t, ps, ps_vel, ps_acc, t_opt, ps_opt, ps_opt_vel, ps_opt_acc)
# print("Ave. cal. time(s): ", np.average(cal_time), np.average(cal_time_opt_cubic))
# print("Std. cal. time(s): ", np.std(cal_time), np.std(cal_time_opt_cubic))
# print("Ave. transfer. time(s): ", np.average(time_transfer), np.average(time_transfer_opt))
# print("Std. transfer. time(s): ", np.std(time_transfer), np.std(time_transfer_opt))
# print("Diff. of completion time(s): ", (np.average(time_transfer)-np.average(time_transfer_opt))*len(traj))
