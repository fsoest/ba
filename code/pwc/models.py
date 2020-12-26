import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiproc.data_preprocessing import rev_angle_embedding
from dataset import WorkDataset
from multiproc.pwc_helpers import wrapper


def encoder_mask(data_size, N):
    mask = torch.full((data_size, N, 2), np.pi)
    mask[:, :, 1] = torch.full(mask[:, :,1].shape, 2*np.pi)
    return mask

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(LSTMNetwork, self).__init__()

        self.N = N
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden_input = hidden_input
        self.lstm = nn.LSTM(input_size=hidden_input[1], hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bi, dropout=dropout)

        if self.lstm.bidirectional == True:
            self.n_directions = 2
        else:
            self.n_directions = 1
        self.fc1 = nn.Linear(input_size, hidden_input[0])
        self.fc2 = nn.Linear(hidden_input[0], hidden_input[1])
        self.fc3 = nn.Linear(self.n_directions * hidden_size, output_hidden)
        self.fc4 = nn.Linear(output_hidden, output_size)
        self.relu = nn.ReLU()


        self.criterion = torch.nn.MSELoss()

    def forward(self, input, hidden, cell):
        input = self.fc1(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.relu(input)
        output, internals = self.lstm(input, (hidden, cell))
        hidden, cell = internals
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        return output, (hidden, cell)

    def initHiddenCell(self):
        return torch.zeros(self.n_directions * self.n_layers, self.batch_size, self.hidden_size, dtype=torch.double), torch.zeros(self.n_directions * self.n_layers, self.batch_size, self.hidden_size, dtype=torch.double)

    def HiddenCellTest(self, test_size):
        return torch.zeros(self.n_directions * self.n_layers, test_size, self.hidden_size, dtype=torch.double), torch.zeros(self.n_directions * self.n_layers, test_size, self.hidden_size, dtype=torch.double)

    def learn(self, train_set, valid_set, optimiser, scheduler, max_epoch=1000, patience=30):
        # Parameters for early stopping
        loss_high = 1e3
        count = 0
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(max_epoch):
            for i, batch in enumerate(dataloader):
                print('Batch {} of {}'.format(i, len(train_set) // self.batch_size), end='\r')
                hidden, cell = self.initHiddenCell()
                with torch.no_grad():
                    x = torch.squeeze(batch['x'])
                    y = torch.squeeze(batch['y'])[:, :, :4]
                optimiser.zero_grad()
                output, internals = self.forward(x, hidden, cell)
                loss = self.criterion(output, y)
                loss.backward()
                optimiser.step()
            self.eval()
            valid_loss = self.calc_loss(valid_set).item()
            scheduler.step(valid_loss)
            if valid_loss < loss_high:
                loss_high = valid_loss
                count = 0
                torch.save(self, 'best_model')
                print('Now! Loss: {}'.format(self.calc_loss(valid_set)))
            else:
                count += 1
            self.train()
            if count > patience:
                break
            print('Training loss: {}, Validation loss: {}'.format(loss, valid_loss))
        print('Validation loss of best model: {}'.format(self.calc_loss(valid_set)))
        return epoch

    def calc_loss(self, dataset):
        with torch.no_grad():
            X = dataset.__getitem__(range(len(dataset)))['x']
            y = dataset.__getitem__(range(len(dataset)))['y'][:, :, :4]
            hidden, cell = self.HiddenCellTest(len(X))
            y_pred, internals = self.forward(X, hidden, cell)
            loss = self.criterion(y_pred, y)
        return loss

    def work_ratio(self, data, dt):
        dataset = WorkDataset(data, self.N, 'lstm')
        with torch.no_grad():
            X = dataset.__getitem__(range(len(dataset)))['x']
            hidden, cell = self.HiddenCellTest(len(X))
            y_pred, internals = self.forward(X, hidden, cell)

        trans_pred = rev_angle_embedding(y_pred, self.N)
        E_pred = np.zeros(len(y_pred))

        for i in range(len(E_pred)):
            E_pred[i] = wrapper(trans_pred[i], data[i, 0][:self.N], data[i, 0][self.N:], dt, data[i, 3], self.N)

        return np.mean(E_pred / data[:, 2])


class single_layer_fcANN(nn.Module):
    def __init__(self, hidden_neuron, N):
        super(single_layer_fcANN, self).__init__()

        self.fc1 = nn.Linear(4 * N, hidden_neuron)
        self.fc2 = nn.Linear(hidden_neuron, 4 * N)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class two_layer_fcANN(nn.Module):
    def __init__(self, neurons, N, batch_size):
        super(two_layer_fcANN, self).__init__()

        self.criterion = torch.nn.MSELoss()
        self.N = N
        self.batch_size = batch_size
        self.fc1 = nn.Linear(4 * N, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], 4 * N)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def learn(self, train_set, valid_set, optimiser, scheduler, max_epoch=1000, patience=30):
        # Parameters for early stopping
        loss_high = 1e3
        count = 0
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(max_epoch):
            for i, batch in enumerate(dataloader):
                print('Batch {} of {}'.format(i, len(train_set) // self.batch_size), end='\r')
                with torch.no_grad():
                    x = torch.squeeze(batch['x'])
                    y = torch.squeeze(batch['y'])
                optimiser.zero_grad()
                output = self.forward(x)
                loss = self.criterion(output, y)
                loss.backward()
                optimiser.step()
            self.eval()
            valid_loss = self.calc_loss(valid_set).item()
            scheduler.step(valid_loss)
            if valid_loss < loss_high:
                loss_high = valid_loss
                count = 0
                torch.save(self, 'best_model')
                print('Now! Loss: {}'.format(self.calc_loss(valid_set)))
            else:
                count += 1
            self.train()
            if count > patience:
                break
            print('Training loss: {}, Validation loss: {}'.format(loss, valid_loss))
        print('Validation loss of best model: {}'.format(self.calc_loss(valid_set)))
        return epoch

    def calc_loss(self, dataset):
        with torch.no_grad():
            X = dataset.__getitem__(range(len(dataset)))['x']
            y = dataset.__getitem__(range(len(dataset)))['y']
            y_pred = self.forward(X)
            loss = self.criterion(y_pred, y)
        return loss

    def work_ratio(self, data, dt):
        dataset = WorkDataset(data, self.N, 'ann')
        with torch.no_grad():
            X = dataset.__getitem__(range(len(dataset)))['x']
            y_pred = self.forward(X)

        trans_pred = rev_angle_embedding(y_pred, self.N, reshape=True)
        E_pred = np.zeros(len(y_pred))

        for i in range(len(E_pred)):
            E_pred[i] = wrapper(trans_pred[i], data[i, 0][:self.N], data[i, 0][self.N:], dt, data[i, 3], self.N)

        return np.mean(E_pred / data[:, 2])
