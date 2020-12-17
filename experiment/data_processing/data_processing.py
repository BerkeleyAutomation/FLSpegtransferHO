import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use('seaborn-whitegrid')

def index_outlier(trajectory):
    index = []
    for i,joints in enumerate(trajectory):
        if joints == [] or joints[3]==joints[4]==joints[5]==0.0:
            print('faulted data: ', i)
            index.append(i)
    return index

# Trajectory check
q_cmd = np.load('q_cmd.npy', allow_pickle=True)    # desired joint angles: [q0, ..., q6]
q_phy = np.load('q_phy.npy', allow_pickle=True)    # actual joint angles: [q0, ..., q6]

t_stamp = np.load('t_stamp.npy')    # measured time (sec)
print('data length: ', len(q_cmd))

# find and delete the outlier
index = index_outlier(q_phy)
print ("number of outlier: ", index)
q_cmd = np.delete(q_cmd, index, axis=0)
q_phy = np.delete(q_phy, index, axis=0)
t_stamp = np.delete(t_stamp, index, axis=0)
print ("outlier has been deleted.")
# type conversion: in case of allow_pickle = True
q_phy = np.array([np.array(p) for p in q_phy])

# plot joint angles
RMSE = []
for i in range(6):
    RMSE.append(np.sqrt(np.sum((q_cmd[:,i] - q_phy[:,i]) ** 2)/len(q_phy)))
print("RMSE=", RMSE)
t = range(len(q_cmd))
plt.title('joint angle')
plt.subplot(611)
plt.plot(t, q_cmd[:,0]*180./np.pi, 'b-', t, q_phy[:,0]*180./np.pi, 'r-')
plt.ylabel('q0 ($^\circ$)')
plt.subplot(612)
plt.plot(t, q_cmd[:,1]*180./np.pi, 'b-', t, q_phy[:, 1] * 180. / np.pi, 'r-')
plt.ylabel('q2 ($^\circ$)')
plt.subplot(613)
plt.plot(t, q_cmd[:, 2], 'b-', t, q_phy[:, 2], 'r-')
plt.ylabel('q3 (mm)')
plt.subplot(614)
plt.plot(t, q_cmd[:, 3]*180./np.pi, 'b-', t, q_phy[:, 3]*180./np.pi, 'r-')
plt.ylabel('q4 ($^\circ$)')
plt.subplot(615)
plt.plot(t, q_cmd[:, 4]*180./np.pi, 'b-', t, q_phy[:, 4]*180./np.pi, 'r-')
plt.ylabel('q5 ($^\circ$)')
plt.subplot(616)
plt.plot(t, q_cmd[:, 5]*180./np.pi, 'b-', t, q_phy[:, 5]*180./np.pi, 'r-')
plt.ylabel('q6 ($^\circ$)')
plt.xlabel('(step)')
plt.show()


# save as a new data
np.save('q_cmd_new.npy', q_cmd)
np.save('q_phy_new.npy', q_phy)
np.save('t_stamp_new.npy', t_stamp)
print("new data has been saved")