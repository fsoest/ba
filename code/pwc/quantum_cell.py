import torch
import math
from torch.utils.data import DataLoader

class Quantum_cell(torch.nn.Module):
    def __init__(self, drive_size, trans_size, input_size, output_size, cell_size, batch_size, dropout):
        super(Quantum_cell, self).__init__()

        # Initialise matrices
        self.w_input = torch.nn.Linear(drive_size, input_size).double()
        self.w_output = torch.nn.Linear(output_size, trans_size).double()

        self.create_cell = torch.nn.Linear(drive_size, cell_size).double()

        self.td = torch.nn.Linear(input_size, output_size).double()
        self.ts = torch.nn.Linear(cell_size, output_size).double()
        # self.w_td = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        # self.w_ts = torch.nn.Parameter(torch.Tensor(output_size, cell_size))
        # self.b_td = torch.nn.Parameter(torch.Tensor(output_size))
        # self.b_ts = torch.nn.Parameter(torch.Tensor(output_size))

        self.ud = torch.nn.Linear(input_size, cell_size**2).double()
        self.ut = torch.nn.Linear(output_size, cell_size**2).double()
        # self.w_ud = torch.nn.Parameter(torch.Tensor(cell_size, input_size))
        # self.w_ut = torch.nn.Parameter(torch.Tensor(cell_size, output_size))
        # self.b_ud = torch.nn.Parameter(torch.Tensor(cell_size))
        # self.b_ut = torch.nn.Parameter(torch.Tensor(cell_size))
        self.dropout_layer = torch.nn.Dropout(dropout)

        self.cell_size = cell_size
        self.batch_size = batch_size

        self.criterion = torch.nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.cell_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def q_cell(self, drive, cell):
        input = self.w_input(drive)
        T = torch.tanh(self.td(input) + self.ts(cell))
        U = torch.tanh(self.ud(input) + self.ut(T))
        T = self.dropout_layer(T)
        U = self.dropout_layer(U)
        U = torch.reshape(U, (drive.shape[0], self.cell_size, self.cell_size))
        cell = torch.bmm(U, cell.unsqueeze(-1)).squeeze(-1)
        trans = self.w_output(T)
        return trans, cell

    def forward(self, drive, cell):
        # drive: [batch, time, features]
        trans = torch.zeros((drive.shape[0], drive.shape[1], drive.shape[2] - 1))
        for t in range(drive.shape[1]):
            trans[:, t], cell = self.q_cell(drive[:, t], cell)
        return trans, cell

    def init_cell(self, d_0):
        return self.create_cell(d_0)

    def learn(self, train_set, valid_set, optimiser, scheduler, max_epoch=1000, patience=30):
        loss_high = 1e3
        count = 0
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
        for epoch in range(max_epoch):
            for i, batch in enumerate(dataloader):
                print('Batch {} of {}'.format(i, len(train_set) // self.batch_size), end='\r')
                with torch.no_grad():
                    x = torch.squeeze(batch['x'])
                    y = torch.squeeze(batch['y'])[:, :, :4]
                optimiser.zero_grad()
                cell = self.init_cell(x[:, 0])
                output, cell = self.forward(x, cell)
                loss = self.criterion(output.double(), y.double())
                loss.backward()
                optimiser.step()
            self.eval()
            valid_loss = self.calc_loss(valid_set).item()
            scheduler.step(valid_loss)
            if valid_loss < loss_high:
                loss_high = valid_loss
                count = 0
                torch.save(self, 'q_cell_best')
                print('Now! Loss: {}'.format(self.calc_loss(valid_set)))
            else:
                count += 1
            self.train()
            if count > patience:
                break
            print('Training loss: {}, Validation loss: {}'.format(loss, valid_loss))
        return epoch, valid_loss

    def calc_loss(self, dataset):
        with torch.no_grad():
            X = dataset.__getitem__(range(len(dataset)))['x']
            y = dataset.__getitem__(range(len(dataset)))['y'][:, :, :4]
            cell = self.init_cell(X[:, 0])
            y_pred, cell = self.forward(X, cell)
            loss = self.criterion(y_pred.double(), y.double())
        return loss
