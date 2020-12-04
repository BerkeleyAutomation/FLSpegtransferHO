import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


# Preprocessing string data
# alphabet(0-25), others(26~32), start(33), end(34) -> 35 chars

n_hidden = 50
lr = 0.01
epochs = 2000

string = "hello pytorch.how long can a rnn cell remember?"# show us your limit!"
chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
char_list = [i for i in chars]
char_len = len(char_list)
n_letters = len(char_list)


# String to onehot vector
# a -> [1 0 0 ... 0 0]

def string_to_onehot(string):
    start = np.zeros(shape=len(char_list) ,dtype=int)
    end = np.zeros(shape=len(char_list) ,dtype=int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=char_len ,dtype=int)
        zero[idx]=1
        start = np.vstack([start,zero])
    output = np.vstack([start,end])
    return output


# Onehot vector to word
# [1 0 0 ... 0 0] -> a

def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]


# RNN with 1 hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.act_fn = nn.Tanh()

    def forward(self, input, hidden):
        hidden = self.act_fn(self.i2h(input) + self.h2h(hidden))
        output = self.i2o(hidden)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


rnn = RNN(n_letters, n_hidden, n_letters)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())

for i in range(epochs):
    rnn.zero_grad()
    total_loss = 0
    hidden = rnn.init_hidden()
    for j in range(one_hot.size()[0]-1):
        import pdb; pdb.set_trace()
        input = Variable(one_hot[j:j+1,:])
        output, hidden = rnn.forward(input, hidden)
        target = Variable(one_hot[j+1])
        loss = loss_func(output.view(-1),target.view(-1))
        total_loss += loss
        input = output

    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(total_loss)


hidden = rnn.init_hidden()
input = Variable(one_hot[0:1,:])

for i in range(len(string)):
    output, hidden = rnn.forward(input, hidden)
    print(onehot_to_word(output.data),end="")
    input = output
