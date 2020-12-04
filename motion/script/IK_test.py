import numpy as np

# # Data collection
# motion = dvrkMotionBridgeP()
# model = dvrkKinematics()
# pos_min = [-0.1, -0.1, -0.15]
# pos_max = [0.1, 0.1, -0.10]
# rot_min = [-40, -40, -40]
# rot_max = [40, 40, 40]
# t_analytic_record = []
# t_numerical_record = []
# error_analytic_record = []
# error_numerical_record = []
# for i in range(1000):
#     pos_rand = np.random.uniform(pos_min, pos_max)
#     rot_rand = np.random.uniform(rot_min, rot_max)
#     quat_rand = U.euler_to_quaternion(rot_rand, unit='deg')
#     motion.set_pose(pos1=[0.0, 0.0, -0.08], rot1=[0.0, 0.0, 0.0, 1.0])  # move neutral posture
#     motion.set_pose(pos1=pos_rand, rot1=quat_rand)
#
#     t0 = time.time()
#     joint_analytic = model.pose_to_joint(pos_rand, quat_rand, method='analytic')
#     t_analytic = time.time() - t0
#     t0 = time.time()
#     joint_numerical = model.pose_to_joint(pos_rand, quat_rand, method='numerical')
#     t_numerical = time.time() - t0
#     joint_GT = motion.act_joint1
#
#     # print(list(np.array(joint_analytic) * 180 / np.pi))
#     # print(list(np.array(joint_numerical) * 180 / np.pi))
#     # print(list(np.array(joint_GT) * 180 / np.pi))
#
#     error_analytic = np.array(joint_analytic) - np.array(joint_GT)
#     error_numerical = np.array(joint_numerical) - np.array(joint_GT)
#
#     print(i+1)
#     print ("Error_analytic=", list(error_analytic))
#     print("Error_numerical=", list(error_numerical))
#     print ("Time_analytic=", t_analytic)
#     print ("Time_numerical=", t_numerical)
#     print ()
#     t_analytic_record.append(t_analytic)
#     t_numerical_record.append(t_numerical)
#     error_analytic_record.append(error_analytic)
#     error_numerical_record.append(error_numerical)
#
# np.save("t_analytic.npy", t_analytic_record)
# np.save("t_numerical.npy", t_numerical_record)
# np.save("error_analytic.npy", error_analytic_record)
# np.save("error_numerical.npy", error_numerical_record)


# Evaluation
t_analytic = np.load('t_analytic.npy')
t_numerical = np.load('t_numerical.npy')
error_analytic = np.load('error_analytic.npy')
error_numerical = np.load('error_numerical.npy')

import matplotlib.pyplot as plt

plt.plot(t_numerical*1000, 'b-', t_analytic*1000, 'r-')
plt.ylabel('Computing time (ms)')
plt.xlabel('Number of samples')
plt.legend(['Numerical', 'Analytic'])
plt.show()

plt.plot(error_numerical[:,5]*180/np.pi, 'b-', error_analytic[:,5]*180/np.pi, 'r-')
plt.ylabel('Calculation error, q6 (deg)')
plt.xlabel('Number of samples')
plt.legend(['Numerical', 'Analytic'])
plt.show()

print("Computing time (s)")
print("Average_analytic=", np.average(t_analytic))
print("Average_numerical=", np.average(t_numerical))
print("SD_analytic=", np.std(t_analytic))
print("SD_numerical=", np.std(t_numerical))
print()
print("Joint error (rad) or (m)")
print("RMSE_analytic=", np.sqrt(np.sum(error_analytic, axis=0)**2)/len(error_analytic))
print("RMSE_numerical=", np.sqrt(np.sum(error_numerical, axis=0)**2)/len(error_numerical))
print("Max_analytic=", np.max(abs(error_analytic), axis=0))
print("Max_numerical=", np.max(abs(error_numerical), axis=0))