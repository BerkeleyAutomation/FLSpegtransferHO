from FLSpegtransfer.training.hysteresis_compensation.plot import *
from FLSpegtransfer.training.dvrkHystCompDNN import NNModel
from FLSpegtransfer.path import *
import numpy as np

# load data
dir = "dataset/insertion_sampled_grey2/"
q_cmd = np.load(root + dir + "q_cmd.npy")
q_phy = np.load(root + dir + "q_phy.npy")
print ("data shape: ", np.shape(q_cmd))

# plot training data
# plot_joint(q_cmd, q_phy, q_phy, show_window=False)
# plot_hysteresis(q_cmd, q_phy, show_window=True)

# training dataset
x = q_cmd[:, :3]
y = q_phy[:, :3]

# train model
input_dim = np.shape(x)[1]
output_dim = np.shape(y)[1]
model = NNModel(input_dim, output_dim)
model.train(x, y, batch_size=len(x), num_workers=[], num_epoch=2000)
y_pred = []
for input in x:
    output = model.model_out(input)
    y_pred.append(output)

# training result
dummy = np.zeros((len(x),3))
x = np.concatenate((x, dummy), axis=1)
y = np.concatenate((y, dummy), axis=1)
y_pred = np.concatenate((y_pred, dummy), axis=1)
# plot_joint(x[:100], y[:100], y_pred[:100], show_window=False)
# plot_hysteresis(x, y_pred, show_window=True)

i=4
model.save('model_grey_peg_sampled_q123.out'+str(i))