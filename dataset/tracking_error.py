import numpy as np
from FLSpegtransfer.path import *
from sklearn.metrics import mean_squared_error
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar

# Load experimental data
q_cmd = np.load(root+"dataset/insertion_sampled_grey1/q_cmd.npy")
q_phy = np.load(root+"dataset/insertion_sampled_grey1/q_phy.npy")

# cartesian position
pos_cmd = [dvrkKinematics.fk(q, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4)[:3, 3] for q in q_cmd]
pos_phy = [dvrkKinematics.fk(q, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=dvrkVar.L3, L4=dvrkVar.L4)[:3, 3] for q in q_phy]
pos_cmd = np.array(pos_cmd)
pos_phy = np.array(pos_phy)

# Evaluate data
RMSE = np.sqrt(np.sum((pos_cmd - pos_phy)**2)/len(pos_cmd))
RMSE_axis = np.sqrt(np.sum((pos_cmd - pos_phy)**2, axis=0)/len(pos_cmd))
print ("Number of samples: ", len(pos_cmd))
print("RMSE_total=", RMSE, '(m)')
print("RMSE_each_axis=", RMSE_axis, '(m)')

# positioning error (total)
pos_error = np.sqrt(np.sum((pos_cmd - pos_phy)**2, axis=1))  # position dist between phy & cmd
avr_total = np.average(pos_error)
SD_total = np.std(pos_error)
print("Average_total=", avr_total, '(m)')
print("SD_total=", SD_total, '(m)')

# positioning error (each axis)
pos_error_axis = abs(pos_cmd - pos_phy)
avr_axis = np.average(pos_error_axis, axis=0)
SD_axis = np.std(pos_error_axis, axis=0)
print("Average_axis=", avr_axis, '(m)')
print("SD_axis=", SD_axis, '(m)')
