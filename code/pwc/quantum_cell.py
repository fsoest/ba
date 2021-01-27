import torch

class Quantum_cell(torch.nn.Module):
    def __init__(self, drive_size, trans_size, input_size, output_size, cell_size, batch_size):
        super(Quantum_cell, self).__init__()

        # Initialise matrices
        self.w_input = torch.nn.Parameter(torch.tensor(input_size, drive_size))
        self.w_output = torch.nn.Parameter(torch.tensor(trans_size, output_size))

        self.w_td = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.w_ts = torch.nn.Parameter(torch.Tensor(output_size, cell_size))
        self.b_td = torch.nn.Parameter(torch.Tensor(output_size))
        self.b_ts = torch.nn.Parameter(torch.Tensor(output_size))

        self.w_ud = torch.nn.Parameter(torch.Tensor(cell_size, input_size))
        self.w_ut = torch.nn.Parameter(torch.Tensor(cell_size, output_size))
        self.b_ud = torch.nn.Parameter(torch.Tensor(cell_size))
        self.b_ut = torch.nn.Parameter(torch.Tensor(cell_size))

        self.U = torch.nn.Parameter(torch.Tensor(cell_size, cell_size))

        self.cell_size = cell_size
        self.batch_size = batch_size

        self.criterion = torch.nn.MSELoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.cell_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, drive, cell):
        input = torch.mm(self.w_input, drive)
        T = torch.tanh(torch.mm(self.w_td, input) + self.b_td + torch.mm(self.w_ts, cell) + self.b_ts)
        U = torch.tanh(torch.mm(self.w_ud, input) + self.b_ud + torch.mm(self.w_ut, T) + self.b_ut)
        cell = torch.mm(U, cell)
        trans = torch.mm(self.w_output, T)
        return trans, cell

    def init_cell(self):
        return torch.zeros(batch_size, self.cell_size, dtype=torch.double)

    def learn(self, train_set, valid_set, optimiser, scheduler, max_epoch=1000):
        loss_high = 1e3
        count = 0
        dataloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
        for epoch in range(max_epoch):
            for i, batch in enumerate(dataloader):
                print('Batch {} of {}'.format(i, len(train_set) // self.batch_size), end='\r')
                cell = self.init_cell()
                with torch.no_grad():
                    x = torch.squeeze(batch['x'])
                    y = torch.squeeze(batch['y'])[:, :, :4]
                optimiser.zero_grad()
                output, internals = self.forward(x, cell)
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
