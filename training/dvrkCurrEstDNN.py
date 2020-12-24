from FLSpegtransferHO.path import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_input, data_output):     # pre-proceessing
        self.x = data_input
        self.y = data_output

    def __len__(self):    # return numbers of total samples when using len(dataset)
        return len(self.x)

    def __getitem__(self, index):   # return dataset[i]
        x = self.x[index]
        y = self.y[index]
        return x, y


class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        if torch.cuda.device_count() > 1:
            self.layer = nn.DataParallel(self.layer)
        self.layer.to(self.device)

        # Set hyperparameters
        # self.batch_size = 30
        self.learning_rate = 0.001

        # Loss func & Optimizer
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        out = self.layer(x)
        return out

    def train(self, data_in, data_out, batch_size, history, num_workers, num_epoch):
        x,y = self.format_dataset(data_in, data_out, history)
        x = Variable(torch.FloatTensor(x)).to(self.device)
        y = Variable(torch.FloatTensor(y)).to(self.device)
        # dataset = CustomDataset(x, y)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for epoch in range(num_epoch):
            # for i, data in enumerate(dataloader):
            self.optimizer.zero_grad()
            output = self.forward(x)
            loss = self.loss_func(output, y)
            loss.backward()
            self.optimizer.step()
            print('Epoch: {}/{}, Cost: {:.6f}'.format(epoch+1, num_epoch, loss))

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("trained model has been saved as '", filename, "'" )

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

    def model_out(self, x):
        if type(x) is not torch.Tensor:
            x = Variable(torch.FloatTensor(x))
        x = x.to(self.device)
        y_pred = self.layer(x)
        return y_pred.tolist()

    def format_dataset(self, data_in, data_out, history):
        H = history
        data_in = np.array(data_in)
        data_out = np.array(data_out)

        # History of joint angles
        # x = [q0(t), q2(t), q3(t), ..., q0(t-H+1), q2(t-H+1), q3(t-H+1)]
        #     [q0(t+1), q2(t+1), q3(t+1), ..., q0(t-H+2), q2(t-H+2), q3(t-H+2)]
        #     [ ... ]
        x = []
        for i in range(H):
            x.append(data_in[i:len(data_in)-H+1+i])
        x = x[::-1]
        x = np.hstack(x)
        y = data_out[H-1:]
        print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
        return x, y


class dvrkCurrEstDNN():
    def __init__(self, history, nb_ensemble, nb_axis):
        # model inform.
        self.H = history
        self.nb_axis = nb_axis
        dir = "training/collision_detection/models/DNN/high_acc_3Hz/"
        filename = root + dir + "model_mot_curr_est.out"
        input_dim = self.nb_axis*history
        output_dim = self.nb_axis

        # load model
        self.models = []
        for i in range(nb_ensemble):
            model = NNModel(input_dim, output_dim)
            model.load_model(filename+str(i))
            self.models.append(model)

        # data members
        self.q_cmd_hist = []

    def model_out(self, x):
        preds = []
        for model in self.models:
            preds.append(model.model_out(x))
        return np.median(preds, axis=0)

    def predict(self, q_cmd):
        if len(self.q_cmd_hist) < self.H * self.nb_axis:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd)
            q_pred = [0.0]*self.nb_axis
        else:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd)
            self.q_cmd_hist = np.delete(self.q_cmd_hist, np.s_[-self.nb_axis:])  # slice
            q_pred = self.model_out(self.q_cmd_hist)
        return q_pred