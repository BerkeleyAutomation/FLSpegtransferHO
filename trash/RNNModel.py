import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn = nn.DataParallel(self.rnn)
        self.rnn.to(self.device)


        # Hyper parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Loss func & Optimizer
        self.learning_rate = 0.001
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x, hidden):
        # x shape = (batch_size, time_steps, input_size)
        x = x.unsqueeze(0)  # same as np.expand_dims(x, axis=0)
        out, hidden = self.rnn(x, hidden)
        return out, hidden

    def init_hidden(self):
        # hidden shape = (num_layers, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(self.device)
        return hidden

    def train(self, data_in, data_out, history, num_epoch):
        x,y = self.format_dataset(data_in, data_out, history)
        x = Variable(torch.FloatTensor(x)).to(self.device)
        y = Variable(torch.FloatTensor(y)).to(self.device)
        batch_size = np.shape(x)[0]
        for epoch in range(num_epoch):
            hidden = self.init_hidden()
            loss = 0
            for b in range(batch_size):
                self.optimizer.zero_grad()
                output, hidden = self.forward(x[b], hidden)
                loss += self.loss_func(hidden, y[:,b].unsqueeze(0))
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
        out, hidden = self.rnn(x, self.init_hidden())
        return hidden.tolist()

    def format_dataset(self, data_in, data_out, history):
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
        x = x.reshape(len(data_in)-H+1, H, -1)
        y = data_out[H-1:]
        y = np.expand_dims(y, axis=0)
        print("x.shape= ", np.shape(x), "y.shape= ", np.shape(y))
        print("x shape = (batch_size, time_steps, input_size)")
        print("y shape = (num_layers, batch_size, hidden_size)")
        return x, y