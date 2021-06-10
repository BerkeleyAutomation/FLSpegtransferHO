from FLSpegtransfer.training.hysteresis_compensation.plot import *
from FLSpegtransfer.training.dvrkHystCompDNN import NNModel
from FLSpegtransfer.path import *
import numpy as np

def format_dataset(q_des, q_act, H):
    # Training only using wrist angles
    # q_des = q_des[:, 3:]
    # q_act = q_act[:, 3:]

    # History of joint angles
    # x = [q_(t), q_(t-1), q_(t-2), ..., q_(t-H+1)]
    #     [q_(t+1), q_(t), q_(t-1), ..., q_(t-H+2)]
    #     [ ... ]
    x = []
    for i in range(H):
        x.append(q_des[i:len(q_des)-H+1+i])
    x = x[::-1]
    x = np.hstack(x)
    y = q_act[H-1:]
    print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
    return x, y

# load data
q_cmd = np.load("q_cmd.npy")
q_phy = np.load("q_phy.npy")
print ("data shape: ", np.shape(q_cmd))

# plot training data
# plot_joint(q_cmd, q_phy, q_phy, show_window=False)
# plot_hysteresis(q_cmd, q_phy, show_window=True)

# training dataset
# H_upper = [np.deg2rad(-7), np.deg2rad(17), np.deg2rad(2)]
# H_lower = [np.deg2rad(-13), np.deg2rad(-10), np.deg2rad(-30)]
x, y = format_dataset(q_cmd[:, 4:], q_phy[:, 4:], H=6)

# train model
input_dim = np.shape(x)[1]
output_dim = np.shape(y)[1]
model = NNModel(input_dim, output_dim)
model.train(x, y, batch_size=len(x), num_workers=[], num_epoch=3000)
y_pred = []
for input in x:
    output = model.model_out(input)
    y_pred.append(output)

# training result
dummy = np.zeros((len(x),4))
x = np.concatenate((dummy, np.array(x)[:,:3]), axis=1)
y = np.concatenate((dummy, y), axis=1)
y_pred = np.concatenate((dummy, y_pred), axis=1)
# plot_joint(x[:200], y[:200], y_pred[:200], show_window=False)
# plot_hysteresis(x, y_pred, show_window=True)

i=9
model.save('model_random_smooth_q56.out'+str(i))