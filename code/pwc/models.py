import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiproc.data_preprocessing import rev_angle_embedding, rev_mult_embedding, rev_out_embedding
from dataset import WorkDataset
from multiproc.pwc_helpers import wrapper


def encoder_mask(data_size, N):
    mask = torch.full((data_size, N, 2), np.pi)
    mask[:, :, 1] = torch.full(mask[:, :,1].shape, 2*np.pi)
    return mask

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        """
        input_size = Size of input embedding
        output_size = size of output embedding
        hidden_size = hidden size of LSTM
        hidden_input = list, amount of neurons in each dense layer
        output_hidden = neurons in dense output layer
        batch_size
        n_layers
        N
        bi
        dropout
        """
        super(LSTMNetwork, self).__init__()

        self.N = N
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden_input = hidden_input
        self.dropout = dropout
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
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
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
                torch.save(self, 'best_model_{}'.format(self.dropout))
                print('Now! Loss: {}'.format(self.calc_loss(valid_set)))
            else:
                count += 1
            self.train()
            if count > patience:
                break
            print('Training loss: {}, Validation loss: {}'.format(loss, valid_loss))
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
    def __init__(self, neurons, N, batch_size, dropout):
        super(two_layer_fcANN, self).__init__()

        self.criterion = torch.nn.MSELoss()
        self.N = N
        self.batch_size = batch_size
        self.fc1 = nn.Linear(4 * N, neurons[0])
        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.fc3 = nn.Linear(neurons[1], neurons[2])
        self.fc4 = nn.Linear(neurons[2], 4 * N)
        self.relu = nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout_layer(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.dropout_layer(x)
        x = self.relu(x)
        x = self.fc4(x)
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
                torch.save(self, 'models/N_5_ann')
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


class LSTM_total_dropout(LSTMNetwork):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(LSTM_total_dropout, self).__init__(input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, input, hidden, cell):
        input = self.fc1(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        output, internals = self.lstm(input, (hidden, cell))
        hidden, cell = internals
        output = self.dropout_layer(output)
        output = self.fc3(output)
        output = self.dropout_layer(output)
        output = self.relu(output)
        output = self.fc4(output)
        return output, (hidden, cell)


class FCNN_LSTM(LSTMNetwork):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(FCNN_LSTM, self).__init__(input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout)

        self.lstm = nn.LSTM(input_size=hidden_input[1], hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bi, dropout=dropout)

        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(5 * N, hidden_input[0])
        self.fc2 = nn.Linear(hidden_input[0], N * hidden_input[1])
        self.fc3 = nn.Linear(self.n_directions * hidden_size, output_hidden)
        self.fc4 = nn.Linear(output_hidden, output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden, cell):
        size = len(input)
        input = torch.reshape(input, (size, 5 * self.N))
        input = self.fc1(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = torch.reshape(input, (size, self.N, self.hidden_input[1]))
        output, internals = self.lstm(input, (hidden, cell))
        hidden, cell = internals
        output = self.fc3(output)
        input = self.dropout_layer(input)
        output = self.relu(output)
        output = self.fc4(output)
        return output, (hidden, cell)


class C_LSTM(LSTMNetwork):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(C_LSTM, self).__init__(input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout)


        self.lstm = nn.LSTM(input_size=hidden_input[2], hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bi, dropout=dropout)
        self.fc1 = nn.Linear(input_size, hidden_input[0])
        self.fc2 = nn.Linear(hidden_input[0], hidden_input[1])
        self.fc3 = nn.Linear(hidden_input[1], hidden_input[2])
        self.fc4 = nn.Linear(self.n_directions * hidden_size, output_hidden[0])
        self.fc5 = nn.Linear(output_hidden[0], output_hidden[1])
        self.fc6 = nn.Linear(output_hidden[1], output_hidden[2])
        self.fc7 = nn.Linear(output_hidden[2], output_size)
        self.relu = nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(self.dropout)


    def forward(self, input, hidden, cell):
        input = self.fc1(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = self.fc3(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        output, internals = self.lstm(input, (hidden, cell))
        hidden, cell = internals
        output = self.fc4(output)
        input = self.dropout_layer(input)
        output = self.relu(output)
        output = self.fc5(output)
        input = self.dropout_layer(input)
        output = self.relu(output)
        output = self.fc6(output)
        input = self.dropout_layer(input)
        output = self.relu(output)
        output = self.fc7(output)
        return output, (hidden, cell)


class LastLSTM(LSTMNetwork):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(LastLSTM, self).__init__(input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(input_size=hidden_input[1], hidden_size=hidden_size, num_layers=n_layers, batch_first=True, bidirectional=bi, dropout=dropout)

        self.fc1 = nn.Linear(input_size, hidden_input[0])
        self.fc2 = nn.Linear(hidden_input[0], hidden_input[1])
        self.fc3 = nn.Linear(self.n_directions * hidden_size, output_hidden)
        self.fc4 = nn.Linear(output_hidden, self.N * output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden, cell):
        size = len(input)
        input = self.fc1(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        output, internals = self.lstm(input, (hidden, cell))
        hidden, cell = internals
        output = output[:, -1, :]
        output = self.dropout_layer(output)
        output = self.fc3(output)
        output = self.dropout_layer(output)
        output = self.relu(output)
        output = self.fc4(output)
        output = torch.reshape(output, (size, self.N, self.output_size))
        return output, (hidden, cell)


class LSTM_batch_norm(LSTMNetwork):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(LSTM_batch_norm, self).__init__(input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.norm1 = torch.nn.BatchNorm1d(N)
        # self.norm1 = torch.nn.BatchNorm1d(hidden_input[0])
        # self.norm2 = torch.nn.BatchNorm1d(hidden_input[1])
        # self.norm3 = torch.nn.BatchNorm1d(output_hidden)

    def forward(self, input, hidden, cell):
        input = self.fc1(input)
        input = self.norm1(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.norm1(input)
        input = self.dropout_layer(input)
        input = self.relu(input)
        output, internals = self.lstm(input, (hidden, cell))
        hidden, cell = internals
        output = self.dropout_layer(output)
        output = self.fc3(output)
        output = self.norm1(output)
        output = self.dropout_layer(output)
        output = self.relu(output)
        output = self.fc4(output)
        return output, (hidden, cell)
