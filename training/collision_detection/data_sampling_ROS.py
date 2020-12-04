from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
import numpy as np

root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
dir = "training/collision_detection/dataset/"

joint_traj = np.load(root+dir+"random_sampled_3000.npy")
joint_traj[:,3:] = 0.0
dvrk = dvrkDualArm()
dvrk.set_jaw(jaw1=[0.0])
joint_des_record = []
joint_msd_record = []
curr_record = []
print ("traj_length = ", len(joint_traj))
for i,q in enumerate(joint_traj):
    joint_des, joint_msd, curr = dvrk.set_joint_interpolate(joint1=q, method='cubic', record=True)
    joint_des_record.append(joint_des)
    joint_msd_record.append(joint_msd)
    curr_record.append(curr)
    print (i+1, "th trajectory completed")

# joint_des_record = np.concatenate(joint_des_record, axis=0)
# joint_msd_record = np.concatenate(joint_msd_record, axis=0)
# curr_record = np.concatenate(curr_record, axis=0)
np.save("joint_des", joint_des_record)
np.save("joint_msd", joint_msd_record)
np.save("mot_curr", curr_record)
print("Data is successfully saved.")