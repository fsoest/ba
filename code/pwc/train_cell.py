from models import LSTM_total_dropout
import torch
import numpy as np
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding, angle_embedding
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from torch.utils.data import DataLoader
from multiproc.pwc_helpers import wrapper, rho_path
import matplotlib.pyplot as plt


class LSTM_train_cell_state(LSTM_total_dropout):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout):
        super(LSTM_train_cell_state, self).__init__(input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout)
        self.h_layer = torch.nn.Linear(input_size, self.n_directions * self.n_layers * self.hidden_size)
        self.c_layer = torch.nn.Linear(input_size, self.n_directions * self.n_layers * self.hidden_size)


    def initHiddenCell(self, x_0):
        return self.h_layer(x_0).reshape(self.n_directions * self.n_layers, x_0.shape[0], self.hidden_size), self.c_layer(x_0).reshape(self.n_directions * self.n_layers, x_0.shape[0], self.hidden_size)

    def learn(self, train_set, valid_set, optimiser, scheduler, max_epoch=1000, patience=30):
        # Parameters for early stopping
        loss_high = 1e3
        count = 0
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
        for epoch in range(max_epoch):
            for i, batch in enumerate(dataloader):
                print('Batch {} of {}'.format(i, len(train_set) // self.batch_size), end='\r')
                with torch.no_grad():
                    x = torch.squeeze(batch['x'])
                    y = torch.squeeze(batch['y'])[:, :, :4]
                hidden, cell = self.initHiddenCell(x[:, 0])
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
                hidden, cell = self.initHiddenCell(X[:, 0])
                y_pred, internals = self.forward(X, hidden, cell)
                loss = self.criterion(y_pred, y)
            return loss

# %%
N = 5
seed = 42
batch_size = 44
dt = 1
rho = 'eigen'
N_sobol = 45
runs = range(21)

# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)


x = train_set.__getitem__(range(10))['x']

model = LSTM_train_cell_state(5, 4, 324, [793, 488], 228, batch_size, 3, N, True, 0.32).double()

# %%
learning_rate = 1e-2
sched_factor = 0.25
pat_drop = 2
patience = 10

optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)


model.learn(train_set, valid_set, optimiser, scheduler)
