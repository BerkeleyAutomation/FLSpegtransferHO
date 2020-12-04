import numpy as np
import PyKDL
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics


def index_outlier(trajectory):
    index = []
    for i,joints in enumerate(trajectory):
        if joints[3]==joints[4]==joints[5]==0.0:
            print ('faulted data: ', i)
            index.append(i)
    return index

# file_path = 'pick_place/2/'
file_path = 'random_sampled/'
q_des = np.load(file_path + 'q_des_raw.npy')    # desired joint angles: [q0, ..., q6]
q_act = np.load(file_path + 'q_act_raw.npy')    # actual joint angles: [q0, ..., q6]
# t_stamp = np.load(file_path + 't_stamp_raw.npy')    # measured time (sec)
print('data length: ', len(q_des))

# find and delete the outlier
index = index_outlier(q_act)
print ("number of outlier: ", index)
q_des = np.delete(q_des, index, axis=0)
q_act = np.delete(q_act, index, axis=0)
# t_stamp = np.delete(t_stamp, index, axis=0)

L1 = 0.4318  # Rcc (m)
L2 = 0.4162  # tool
L3 = 0.0091  # pitch ~ yaw (m)
L4 = 0.0102  # yaw ~ tip (m)
pos_des = []
pos_act = []
quat_des = []
quat_act = []
for q in q_des:
    pos_des.append(dvrkKinematics.fk_position(q1=q[0], q2=q[1], q3=q[2], q4=0, q5=0, q6=0, L1=L1, L2=L2, L3=0, L4=0))
    R = dvrkKinematics.fk_orientation(q[0], q[1], q[2], q[3], q[4], q[5])
    R_matrix = PyKDL.Rotation(R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2])
    quat_des.append(list(R_matrix.GetQuaternion()))

for q in q_act:
    pos_act.append(dvrkKinematics.fk_position(q1=q[0], q2=q[1], q3=q[2], q4=0, q5=0, q6=0, L1=L1, L2=L2, L3=0, L4=0))
    R = dvrkKinematics.fk_orientation(q[0], q[1], q[2], q[3], q[4], q[5])
    R_matrix = PyKDL.Rotation(R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2])
    quat_des.append(list(R_matrix.GetQuaternion()))

np.save('q_des', q_des)
np.save('q_act', q_act)
# np.save('t_stamp', t_stamp)
np.save('pos_des', pos_des)
np.save('pos_act', pos_act)
np.save('quat_des', quat_des)
np.save('quat_act', quat_act)
print ("pose saved")