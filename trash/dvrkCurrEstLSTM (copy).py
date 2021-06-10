import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from FLSpegtransfer.trash.LSTMModel import LSTMModel


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_out = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
        if torch.cuda.device_count() > 1:
            self.lstm = nn.DataParallel(self.lstm)
            self.layer_out = nn.DataParallel(self.layer_out)
        self.lstm.to(self.device)
        self.layer_out.to(self.device)

        # Hyper parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Loss func & Optimizer
        self.learning_rate = 0.001
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x, hidden, cell):
        # input shape = (batch_size, time_steps, input_size)
        # output shape = (batch_size, time_steps, hidden_size)
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.layer_out(hidden)
        return out[self.num_layers-1], hidden, cell

    def init_hidden_cell(self, batch_size):
        # hidden shape = (num_layers, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return hidden, cell

    def init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_size).zero_()

    def train(self, data_in, data_out, history, num_epoch, batch_size):
        x, y = self.format_dataset(data_in, data_out, history, batch_size)
        # x = Variable(torch.FloatTensor(x)).to(self.device)
        # y = Variable(torch.FloatTensor(y)).to(self.device)
        for epoch in range(num_epoch):
            hidden, cell = self.init_hidden_cell(batch_size)
            loss = 0.0
            for batch in range(len(x)):
                x_batch = Variable(torch.FloatTensor(x[batch])).to(self.device)
                y_batch = Variable(torch.FloatTensor(y[batch])).to(self.device)
                self.optimizer.zero_grad()
                output, hidden, cell = self.forward(x_batch, hidden, cell)
                loss += self.loss_func(output, y_batch[0])
                print (batch)
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
        x = x.unsqueeze(0)  # same as np.expand_dims(x, axis=0)
        batch_size = np.shape(x)[0]
        hidden, cell = self.init_hidden_cell(batch_size)
        out, hidden, cell = self.forward(x, hidden, cell)
        return out.squeeze().tolist()

    def format_dataset(self, data_in, data_out, history, batch_size):
        H = history
        data_in = np.array(data_in)
        data_out = np.array(data_out)

        # History of joint angles
        # x = [[[q0(0), q2(0), q3(0)]
        #       [q0(1), q2(1), q3(1)]
        #               ...
        #       [q0(H-1), q2(H-1), q3(H-1)]],
        #
        #      [[q0(1), q2(1), q3(1)]
        #       [q0(2), q2(2), q3(2)]
        #               ...
        #       [q0(H), q2(H), q3(H)]],
        #
        #               ...
        #
        #      [[q0(t-H+1), q2(t-H+1), q3(t-H+1)]
        #               ...
        #       [q0(t), q2(t), q3(t)]]]
        x = []
        for i in range(H):
            x.append(data_in[i:len(data_in) - H + 1 + i])
        x = np.hstack(x)
        x = x[:len(x)//batch_size * batch_size]     # throw out the remainder
        x = x.reshape(len(x)//batch_size, batch_size, H, np.shape(data_in)[1])
        y = data_out[H-1:]
        y = y[:len(y)//batch_size * batch_size]     # throw out the remainder
        y = y.reshape(len(y)//batch_size, batch_size, np.shape(data_out)[1])
        y = np.expand_dims(y, axis=1)
        print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
        print("x shape = (batch_size, time_steps, input_size)")
        print("y shape = (num_layers, batch_size, hidden_size)")
        return x, y


class dvrkCurrEstLSTM():
    def __init__(self, history, nb_ensemble, nb_axis):
        # model inform.
        self.H = history
        self.nb_axis = nb_axis
        root = "/home/hwangmh/pycharmprojects/FLSpegtransfer/"
        dir = "training/collision_detection/models/LSTM/"
        filename = root + dir + "model_mot_curr_est.out"
        input_dim = self.nb_axis
        output_dim = self.nb_axis
        hidden_dim = 128
        nb_layers = 1

        # load model
        self.models = []
        for i in range(nb_ensemble):
            model = LSTMModel(input_dim, hidden_dim, output_dim, nb_layers)
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
        if len(self.q_cmd_hist) < self.H:
            self.q_cmd_hist.append(q_cmd)
            q_pred = [0.0 ,0.0, 0.0]
        else:
            self.q_cmd_hist.pop(0)
            self.q_cmd_hist.append(q_cmd)
            q_pred = self.model_out(self.q_cmd_hist)
        return q_pred