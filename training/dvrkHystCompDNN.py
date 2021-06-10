from FLSpegtransfer.path import *
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

    def train(self, data_in, data_out, batch_size, num_workers, num_epoch):
        x = Variable(torch.FloatTensor(data_in)).to(self.device)
        y = Variable(torch.FloatTensor(data_out)).to(self.device)
        dataset = CustomDataset(x, y)
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

    def reset_func(self, model):
        if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
            model.reset_parameters()

    def reset_weight(self):
        self.apply(self.reset_func)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

    def model_out(self, x):
        if type(x) is not torch.Tensor:
            x = Variable(torch.FloatTensor(x))
        x = x.to(self.device)
        y_pred = self.layer(x)
        return y_pred.tolist()


class dvrkHystCompDNN:
    def __init__(self, history, arm_name):
        self.H = history
        self.models_q123, self.models_q4, self.models_q56 = self.load_models(arm_name)

        # data members
        self.q_cmd_hist = []

        # data member for interpolation
        self.v_max = 2.0  # maximum velocity
        self.dt = 0.01  # sampling time interval
        self.q_cmd_int_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def load_models(self, arm_name):
        models_q123 = []
        if arm_name == '/PSM1':
            dir = "models/grey1_PSM1/"
        elif arm_name == '/PSM2':
            dir = "models/grey2_PSM2/"
        else:
            raise ValueError
        filename = root + dir + "model_random_smooth_q123.out"
        for i in range(5):
            model = NNModel(input_dim=3, output_dim=3)
            model.load_model(filename+str(i))
            models_q123.append(model)

        models_q4 = []
        filename = root + dir + "model_random_smooth_q4.out"
        for i in range(5):
            model = NNModel(input_dim=1, output_dim=1)
            model.load_model(filename + str(i))
            models_q4.append(model)

        models_q56 = []
        filename = root + dir + "model_random_smooth_q56.out"
        for i in range(10):
            model = NNModel(input_dim=self.H*2, output_dim=2)
            model.load_model(filename + str(i))
            models_q56.append(model)
        return models_q123, models_q4, models_q56

    def model_out(self, x, models):
        preds = [model.model_out(x) for model in models]
        return np.median(preds, axis=0)

    # predict q_phy for all joints
    def predict(self, q_cmd):
        if q_cmd == []:
            return []
        # predict q1, q2, q3, q4
        # q_pred123 = self.model_out(q_cmd[:3], self.models_q123)
        q_pred123 = q_cmd[:3]
        q_pred4 = self.model_out([q_cmd[3]], self.models_q4)

        # predict q5,q6
        if len(self.q_cmd_hist) < self.H*2:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd[4:])
            q_pred56 = q_cmd[4:]
        else:
            self.q_cmd_hist = np.insert(self.q_cmd_hist, 0, q_cmd[4:])
            self.q_cmd_hist = np.delete(self.q_cmd_hist, np.s_[-2:])
            q_pred56 = self.model_out(self.q_cmd_hist, self.models_q56)
        q_pred = np.concatenate((q_pred123, q_pred4, q_pred56), axis=0)
        return q_pred

    # motion controller (calibrate new cmd)
    def step(self, q_cmd):
        if q_cmd == []:
            return []
        q_pred = self.predict(q_cmd)
        error = np.array(q_cmd) - np.array(q_pred)
        q_cmd_new = np.array(q_cmd) + error
        return q_cmd_new

    # predict & calibrate points
    def cal_interpolate(self, q_cmd, mode):
        if q_cmd == []:
            return []
        if all(np.isclose(q_cmd, self.q_cmd_int_prev)):
            return q_cmd
        else:
            q_cmd = np.array(q_cmd)
            tf = abs((q_cmd - self.q_cmd_int_prev) / self.v_max)
            n = np.ceil(max(tf / self.dt))  # number of points
            dq = (q_cmd - self.q_cmd_int_prev) / n
            for i in range(int(n)):
                if mode=='predict':
                    q_cmd_new = self.predict(self.q_cmd_int_prev + (i+1)*dq)    # predict
                elif mode=='calibrate':
                    q_cmd_new = self.step(self.q_cmd_int_prev + (i+1)*dq)    # calibrate
            self.q_cmd_int_prev = q_cmd
            return q_cmd_new    # return the last one