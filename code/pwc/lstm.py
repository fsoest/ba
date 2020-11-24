import numpy as np
import torch
import torch.nn as nn

# %%
device = torch.device('cpu')


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, nn_hidden):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size

        self.LSTM = nn.LSTM(input_size, hidden_size, self.n_layers)
        self.fc1 = nn.Linear(self.hidden_size, nn_hidden)
        self.fc2 = nn.Linear(nn_hidden, output_size)

    def forward(self, x):
        batch_size = x.size(1)

        hidden = self.initial_hidden_state(batch_size)

        # Pass data into LSTM
        out, hidden = self.LSTM(x, hidden)

        # Output through Linear nets
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)

        return out, hidden

    def initial_hidden_state(self, batch_size):
        # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        # return hidden
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device), weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

# %%
n_layers = 1
seq_len = 10
batch_size = 5
input_size = 4
output_size = 4
hidden_size = 8
nn_hidden = 20

# %%

model = Model(input_size, output_size, hidden_size, n_layers, nn_hidden)

x = torch.randn(seq_len, batch_size, input_size)

model.forward(x)
