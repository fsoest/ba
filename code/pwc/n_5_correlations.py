"""
Document to check for correlations between cell state and current rho
"""
import torch
import numpy as np
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding, angle_embedding
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from torch.utils.data import DataLoader
from multiproc.pwc_helpers import wrapper, rho_path
import matplotlib.pyplot as plt
from rho_vis import exp_xyz, data_wrapper
from colour_bloch import Bloch as cBloch
from torch.utils.data import Dataset
from custom_loss import real_matmul


N = 5
seed = 42
batch_size = 44
dt = 1
rho = 'eigen'
N_sobol = 45
runs = range(21)

# %%
model = torch.load('models/dt_1_bi').eval()

# %%
class CellDataset(Dataset):
    """
    Dataset to process cell states, takes model as well as raw data as inputs
    """
    def __init__(self, inpcell, rhos):
        self.rhos = rhos
        self.cell = inpcell.permute(1, 0, 2).reshape(inpcell.shape[1], 6 * 324)

    def __len__(self):
        return len(self.rhos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rho = torch.from_numpy(self.rhos[idx, -1])

        return {'x': self.cell[idx], 'y': rho}


class correl_model(torch.nn.Module):
    """
    ML model to predict system states from cells states
    """
    def __init__(self, input_size, batch_size, hidden_size):
        """
        Output size 8, 4 complex numbers
        """
        super(correl_model, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 8)
        self.relu = torch.nn.ReLU()
        self.batch_size = batch_size
        self.dropout = 0.3
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, input):
        output = self.layer1(input)
        output = self.dropout_layer(output)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.dropout_layer(output)
        output = self.relu(output)
        output = self.layer3(output)
        return output

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
                    y = torch.squeeze(batch['y'])
                optimiser.zero_grad()
                output = self.forward(x)
                loss = self.loss(output, y)
                loss.backward()
                optimiser.step()
            self.eval()
            valid_loss = self.calc_loss(valid_set).item()
            scheduler.step(valid_loss)
            if valid_loss < loss_high:
                loss_high = valid_loss
                count = 0
                torch.save(self, 'cellmodel')
                print('Now! Loss: {}'.format(self.calc_loss(valid_set)))
            else:
                count += 1
            self.train()
            if count > patience:
                break
            print('Training loss: {}, Validation loss: {}'.format(loss, valid_loss))
        return epoch

    def loss(self, pred, y):
        return torch.mean(self.dist(pred, y))

    def calc_loss(self, data_set):
        with torch.no_grad():
            x = data_set.__getitem__(range(len(data_set)))['x']
            y = data_set.__getitem__(range(len(data_set)))['y']
        output = self.forward(x)
        return self.loss(output, y)

    def dist(self, pred, y):
        """
        pred shape = [batch, 8]
        """
        pred_real = torch.reshape(pred[:, :4], (pred.shape[0], 2, 2))
        pred_imag = torch.reshape(pred[:, 4:], (pred.shape[0], 2, 2))
        y_real = torch.reshape(torch.real(y), (y.shape[0], 4))
        y_imag = torch.reshape(torch.imag(y), (y.shape[0], 4))

        y_col = torch.zeros((y.shape[0], 8))
        y_col[:, :4] = y_real
        y_col[:, 4:] = y_imag

        rho_unnormed_real, rho_unnormed_imag = real_matmul(pred_real, pred_imag, torch.transpose(pred_real, 1, 2), -1 * torch.transpose(pred_imag, 1, 2))

        trace = rho_unnormed_real[:, 0, 0] + rho_unnormed_real[:, 1, 1]

        trace_mat = torch.zeros(rho_unnormed_real.shape)
        trace_mat[:, 0, 0] = trace
        trace_mat[:, 0, 1] = trace
        trace_mat[:, 1, 0] = trace
        trace_mat[:, 1, 1] = trace

        rho_real = torch.div(rho_unnormed_real, trace_mat)
        rho_imag = torch.div(rho_unnormed_imag, trace_mat)

        rho_col = torch.zeros((rho_real.shape[0], 8))
        rho_col[:, :4] = torch.reshape(rho_real, (rho_real.shape[0], 4))
        rho_col[:, 4:] = torch.reshape(rho_imag, (rho_imag.shape[0], 4))

        dist = torch.linalg.norm(rho_col - y_col, dim=1)
        return dist

# %%
learning_rate = 1e-2
sched_factor = 0.25
pat_drop = 2
patience = 10
qub = 3
# %%
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
# %%
rhos_train = np.zeros((len(data_train), N, 2, 2), dtype=np.complex128)
for i, data in enumerate(data_train[:len(data_train)]):
    a, b, c = data_wrapper(data, dt, 1)
    rhos_train[i] = a

with torch.no_grad():
    x = train_set.__getitem__(range(len(data_train)))
    hidden, cell = model.HiddenCellTest(len(data_train))
    y, internals_train = model(x['x'], hidden, cell)
train_set_cell = CellDataset(internals_train[1], rhos_train)
# %%
rhos_valid = np.zeros((len(data_valid), N, 2, 2), dtype=np.complex128)
for i, data in enumerate(data_valid):
    a, b, c = data_wrapper(data, dt, 1)
    rhos_valid[i] = a

with torch.no_grad():
    x = valid_set.__getitem__(range(len(data_valid)))
    hidden, cell = model.HiddenCellTest(len(data_valid))
    y, internals_valid = model(x['x'], hidden, cell)
valid_set_cell = CellDataset(internals_valid[1], rhos_valid)

# %%
rhos_test = np.zeros((len(data_test), N, 2, 2), dtype=np.complex128)
for i, data in enumerate(data_test):
    a, b, c = data_wrapper(data, dt, 1)
    rhos_test[i] = a

with torch.no_grad():
    x = test_set.__getitem__(range(len(data_test)))
    hidden, cell = model.HiddenCellTest(len(data_test))
    y, internals_test = model(x['x'], hidden, cell)
test_set_cell = CellDataset(internals_test[1], rhos_test)
# %%
cellmodel = correl_model(6*324, batch_size, 500).double()

optimiser = torch.optim.SGD(cellmodel.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)


cellmodel.learn(train_set_cell, valid_set_cell, optimiser, scheduler)

# %%
cellmodelload = torch.load('cellmodel').eval()
y_test = test_set_cell.__getitem__(range(len(test_set_cell)))['y']
with torch.no_grad():
    pred_test = cellmodelload(test_set_cell.__getitem__(range(len(test_set_cell)))['x'])

dist_test = cellmodelload.dist(pred_test, y_test)
