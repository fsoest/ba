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
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

# %%
model = torch.load('models/dt_1_bi').eval()

# %%
rhos = np.zeros((len(test_set), N, 2, 2), dtype=np.complex128)
for i, data in enumerate(data_test):
    a, b, c = data_wrapper(data, dt, 1)
    rhos[i] = a

x = test_set.__getitem__(range(len(test_set)))
hidden, cell = model.HiddenCellTest(len(test_set))
y, internals = model(x['x'], hidden, cell)


# %%
class CellDataset(Dataset):
    """
    Dataset to process cell states, takes model as well as raw data as inputs
    """
    def __init__(self, data, model, N, dt, qub, cell, rhos):
        self.data = data
        self.model = model
        self.N = N
        self.dt = dt
        self.qub = qub
        self.cell = cell
        self.rhos = rhos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        ith qubit
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        drives = test_set.__getitem__(range(len(self.data)))['x']
        hidden, cell = model.HiddenCellTest(len(self.data))
        y, internals = model(drives, hidden, cell)
        # Get cell state from LSTM
        hidden, x = internals

        rhos = np.zeros((len(self.data), N, 2, 2), dtype=np.complex128)
        for i, data in enumerate(self.data):
            a, b, c = data_wrapper(data, dt, 1)
            rhos[i] = a
        ith_rho = torch.from_numpy(rhos[:, self.qub])

        return {'x': x[self.qub+1], 'y': ith_rho}


class correl_model(torch.nn.Module):
    """
    ML model to predict system states from cells states
    """
    def __init__(self, input_size, batch_size):
        """
        Output size 8, 4 complex numbers
        """
        super(correl_model, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.layer1 = torch.nn.Linear(input_size, 8)
        self.batch_size = batch_size

    def forward(self, input):
        output = self.layer1(input)
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
                torch.save(self, 'best_model_{}'.format(self.dropout))
                print('Now! Loss: {}'.format(self.calc_loss(valid_set)))
            else:
                count += 1
            self.train()
            if count > patience:
                break
            print('Training loss: {}, Validation loss: {}'.format(loss, valid_loss))
        return epoch

    def loss(self, pred, y):
        """
        pred shape = [batch, 8]
        """
        pred_real = torch.reshape(pred[:, :4], (pred.shape[0], 2, 2))
        pred_imag = torch.reshape(pred[:, 4:], (pred.shape[0], 2, 2))

        rho_unnormed_real, rho_unnormed_imag = real_matmul(pred_real, pred_imag, torch.transpose(pred_real, 1, 2), -1 * torch.transpose(pred_imag, 1, 2))
        rho_real = torch.div(rho_unnormed_real, rho_unnormed_real[:, 0, 0] + rho_unnormed_real[:, 1, 1])
        rho_imag = torch.div(rho_unnormed_imag, rho_unnormed_real[:, 0, 0] + rho_unnormed_real[:, 1, 1])

        # Trace distance between pred and y
        dist = torch.mean(torch.linalg.norm(rho_real + 1j * rho_imag - y, ord='nuc'))

    def calc_loss(self, data_set):
        with torch.no_grad():
            x = data_set.__getitem__(len(data_set))['x']
            y = data_set.__getitem__(len(data_set))['y']
        output = self.forward(x)
        return self.loss(output, y)


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
train_set = CellDataset(data_train, model, N, dt, qub)
test_set = CellDataset(data_test, model, N, dt, qub)
valid_set = CellDataset(data_valid, model, N, dt, qub)

cellmodel = correl_model(324, batch_size)

optimiser = torch.optim.SGD(cellmodel.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)

cellmodel.learn(train_set, valid_set, optimiser, scheduler)
