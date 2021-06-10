import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
char_list = [i for i in chars]
char_len = len(char_list)


batch_size = 1
seq_len = 1
num_layers = 1
input_size = char_len
hidden_size = 35
lr = 0.01
num_epochs = 1000


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
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        return output, hidden, cell

    def init_hidden_cell(self):
        hidden = Variable(torch.zeros(num_layers, seq_len * batch_size, hidden_size))
        cell = Variable(torch.zeros(num_layers, seq_len * batch_size, hidden_size))

        return hidden, cell


rnn = RNN(input_size, hidden_size, num_layers)


one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())
print(one_hot.size())

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)


j=0
input_data = Variable(one_hot[j:j+seq_len].view(batch_size,seq_len,-1))
print(input_data.size())

hidden,cell = rnn.init_hidden_cell()
print(hidden.size(),cell.size())

output, hidden,cell = rnn(input_data,hidden,cell)
print(output.size(),hidden.size(),cell.size())

unroll_len = one_hot.size()[0] // seq_len - 1

input_data2 = []
label2 = []
for j in range(unroll_len):
    input_data2.append(one_hot[j:j + seq_len].numpy())
    label2.append(one_hot[j + 1:j + seq_len + 1].numpy())

input_data2 = np.array(input_data2)
label2 = np.array(label2)
input_data2 = torch.from_numpy(input_data2).type_as(torch.FloatTensor())
label2 = torch.from_numpy(label2).type_as(torch.FloatTensor())

for i in range(num_epochs):
    hidden, cell = rnn.init_hidden_cell()
    loss = 0

    optimizer.zero_grad()
    output, hidden, cell = rnn(input_data2, hidden, cell)
    loss += loss_func(output.view(1, -1), label2.view(1, -1))

    import pdb; pdb.set_trace()

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss)

hidden, cell = rnn.init_hidden_cell()

for j in range(unroll_len - 1):
    input_data = Variable(one_hot[j:j + seq_len].view(batch_size, seq_len, -1))
    label = Variable(one_hot[j + 1:j + seq_len + 1].view(batch_size, seq_len, -1))

    output, hidden, cell = rnn(input_data, hidden, cell)
    print(onehot_to_word(output.data), end="")