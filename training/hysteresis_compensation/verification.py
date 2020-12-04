import numpy as np
from plot import *

from FLSpegtransfer.trash.dvrkMotionControllerDNN import dvrkNNController

# load model
data_in = 15
data_out = 3

root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
dir = "calibration_files/"
model = dvrkNNController(root+dir+"model_hysteresis.out", 10)

# verification data
dir = "experiment/4_verification/dataset/peg_sampled/"
q_des_ver = np.load(root + dir + "q_des.npy")
q_act_ver = np.load(root + dir + "q_act.npy")
print("data shape: ", np.shape(q_des_ver))

# prediction from model
q_pred = []
for q in q_des_ver:
    q_pred.append(model.cal_interpolate(q, mode='predict'))

# verification result
plot_joint(q_des_ver, q_act_ver, q_pred, show_window=False)
plot_hysteresis(q_des_ver, q_pred, show_window=True)