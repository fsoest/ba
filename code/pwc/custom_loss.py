import torch
import torch.nn as nn
import numpy
import argparse
from qutip import rand_ket_haar as rkh
import numpy as np
from multiproc.pwc_helpers import state_to_angles, get_eigen_rho, wrapper
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
from torch.utils.data import DataLoader
from multiproc.data_preprocessing import rev_angle_embedding, import_datasets
import wandb


def make_rho_0(rho, theta_d_0, phi_d_0):
    """
    Creates correct initial system state, depending on method
    """
    if rho == 'haar':
        psi_0 = rkh(2)
        rho_0 = (psi_0 * psi_0.dag()).full()
        return rho_0
    elif rho == '0':
        zero_state = np.matrix([1, 0], dtype=np.complex128)
        rho_0 = zero_state.H @ zero_state
        return rho_0
    elif rho == 'eigen':
        # get_eigen_rho returns array of rho_0, therefore [0]
        rho_0 = get_eigen_rho(np.array([theta_d_0]), np.array([phi_d_0]))[0]
        return rho_0


def matexp(x, dt):
    """
    Calculates the matrix exponentiation for matrix of type -j * dt * [[0, tau*], [tau, 0]]
    """
    exp = torch.zeros(x.shape, dtype=torch.cdouble)
    taus = x[:, 1, 0]
    exp[:, 0, 0] = torch.cos(dt * torch.abs(taus))
    exp[:, 0, 1] = -1j * torch.conj(taus) * torch.sin(dt * torch.abs(taus)) / torch.abs(taus)
    exp[:, 1, 0] = -1j * torch.abs(taus) * torch.sin(dt * torch.abs(taus)) / torch.conj(taus)
    exp[:, 1, 1] = torch.cos(dt * torch.abs(taus))
    return exp


def real_matmul(A_real, A_imag, B_real, B_imag):
    return A_real @ B_real - A_imag @ B_imag, A_imag @ B_real + A_real @ B_imag


class work_loss():
    """
    Class for custom loss function calculating work output directly
    """
    def __init__(self, dt, N):
        self.dt = dt
        self.N = N

    def forward(self, y, x):
        # Calculate rho_0
        with torch.no_grad():
            # Dimensions: [Batch, Time, Embedding]
            sin_phi_D = x[:, :self.N - 1, 1]
            cos_phi_D = x[:, :self.N - 1, 3]
            # exp_phi_D = cos_phi_D + 1j * sin_phi_D
            rho_0_real = torch.full((x.shape[0], 2, 2), 0.5)
            rho_0_imag = torch.zeros((x.shape[0], 2, 2))
            rho_0_real[:, 1, 0] *= cos_phi_D[:, 0]
            rho_0_real[:, 0, 1] *= cos_phi_D[:, 0]
            rho_0_imag[:, 1, 0] = 0.5 * sin_phi_D[:, 0]
            rho_0_imag[:, 0, 1] = -0.5 * sin_phi_D[:, 0]

        # Calculate unitary evolution operators

        # No grad for drive side
        with torch.no_grad():
            # H_D: [batch, time, 2x2 matrix]
            sin_theta_D = x[:, :self.N - 1, 0]

            # H_D = torch.zeros((x.shape[0], self.N - 1, 2, 2), dtype=torch.cdouble)

            # H_D[:, :, 1, 0] = exp_phi_D * sin_theta_D / 2
            # H_D[:, :, 0, 1] = torch.conj(exp_phi_D) * sin_theta_D / 2

            H_D_real = torch.zeros((x.shape[0], self.N - 1, 2, 2))
            H_D_imag = torch.zeros((x.shape[0], self.N -1, 2, 2))

            H_D_real[:, :, 1, 0] = 0.5 * sin_theta_D * cos_phi_D
            H_D_real[:, :, 0, 1] = 0.5 * sin_theta_D * cos_phi_D

            H_D_imag[:, :, 1, 0] = 0.5 * sin_theta_D * sin_phi_D
            H_D_imag[:, :, 0, 1] = -0.5 * sin_theta_D * sin_phi_D


        # H_T: [batch, time, 2x2]
        # H_T = torch.zeros((y.shape[0], self.N, 2, 2), dtype=torch.cdouble)
        theta_T = torch.atan2(y[:, :, 0], y[:, :, 2])
        phi_T = torch.atan2(y[:, :, 1], y[:, :, 3])

        # H_T[:, :, 1, 0] = (torch.cos(phi_T) + 1j * torch.sin(phi_T)) * torch.sin(theta_T) / 2
        # H_T[:, :, 0, 1] = (torch.cos(phi_T) - 1j * torch.sin(phi_T)) * torch.sin(theta_T) / 2

        H_T_real = torch.zeros((y.shape[0], self.N, 2, 2))
        H_T_imag = torch.zeros((y.shape[0], self.N, 2, 2))

        H_T_real[:, :, 1, 0] = torch.cos(phi_T) * torch.sin(theta_T) / 2
        H_T_real[:, :, 0, 1] = torch.cos(phi_T) * torch.sin(theta_T) / 2

        H_T_imag[:, :, 1, 0] = torch.sin(phi_T) * torch.sin(theta_T) / 2
        H_T_imag[:, :, 0, 1] = -1 * torch.sin(phi_T) * torch.sin(theta_T) / 2


        # Abs value of alpha = delta + tau
        # shape: [batch, 2]
        abs_alpha = torch.sqrt(torch.pow(torch.sin(theta_T[:, :self.N - 1]), 2) + torch.pow(sin_theta_D[:, :self.N - 1], 2) + 2 * torch.sin(theta_T[:, :self.N - 1]) * sin_theta_D[:, :self.N - 1] * (torch.cos(phi_T[:, :self.N - 1]) * cos_phi_D[:, :self.N - 1] + torch.sin(phi_T[:, :self.N - 1]) * sin_phi_D[:, :self.N - 1])) / 2

        alpha_real = 0.5 * (sin_theta_D[:, :self.N - 1] * cos_phi_D[:, :self.N - 1] + torch.sin(theta_T[:, :self.N - 1]) * torch.cos(phi_T[:, :self.N - 1]))
        alpha_imag = 0.5 * (sin_theta_D[:, :self.N - 1] * sin_phi_D[:, :self.N - 1] + torch.sin(theta_T[:, :self.N - 1]) * torch.sin(phi_T[:, :self.N - 1]))

        # Unitary evolution operator
        # Shape: [batch, N-1, 2, 2]

        U_real = torch.zeros((x.shape[0], self.N - 1, 2, 2))
        U_imag = torch.zeros((x.shape[0], self.N - 1, 2, 2))

        # Helpers
        c = torch.cos(abs_alpha * self.dt)
        s = torch.div(torch.sin(abs_alpha * self.dt), abs_alpha)

        U_real[:, :, 0, 0] = c
        U_real[:, :, 1, 1] = c
        U_real[:, :, 0, 1] = torch.mul(-1 * alpha_imag, s)
        U_real[:, :, 1, 0] = torch.mul(alpha_imag, s)

        U_imag[:, :, 0, 1] = torch.mul(-1 * alpha_real, s)
        U_imag[:, :, 1, 0] = torch.mul(-1 * alpha_real, s)


        # U_1 = matexp(H_D[:, 0] + H_T[:, 0], self.dt)
        # U_2 = matexp(H_D[:, 1] + H_T[:, 1], self.dt)


        # helper_real, helper_imag = real_matmul(U_real[:, 0], U_imag[:, 0], rho_0_real, rho_0_imag)
        # rho_1_real, rho_1_imag = real_matmul(helper_real, helper_imag, torch.transpose(U_real[:, 0], 1, 2), -1 * torch.transpose(U_imag[:, 0], 1, 2))
        #
        # A_1_real, A_1_imag = real_matmul(rho_1_real, rho_1_imag, (H_T_real[:, 1] - H_T_real[:, 0]), (H_T_imag[:, 1] - H_T_imag[:, 0]))
        # W_1 = A_1_real[:, 0, 0] + A_1_real[:, 1, 1]
        #
        # help2_real, help2_imag = real_matmul(U_real[:, 1], U_imag[:, 1], rho_1_real, rho_1_imag)
        # rho_2_real, rho_2_imag = real_matmul(help2_real, help2_imag, torch.transpose(U_real[:, 1], 1, 2), torch.transpose(-1 * U_imag[:, 1], 1, 2))
        #
        # A_2_real, A_2_imag = real_matmul(rho_2_real, rho_2_imag, H_T_real[:, 2] - H_T_real[:, 1], H_T_imag[:, 2] - H_T_imag[:, 1])
        # W_2 = A_2_real[:, 0, 0] + A_2_real[:, 1, 1]

        W = torch.zeros((x.shape[0]))

        rho_real = rho_0_real
        rho_imag = rho_0_imag

        for i in range(self.N - 1):
            helper_real, helper_imag = real_matmul(U_real[:, i], U_imag[:, i], rho_real, rho_imag)
            rho_real, rho_imag = real_matmul(helper_real, helper_imag, torch.transpose(U_real[:, i], 1, 2), -1 * torch.transpose(U_imag[:, i], 1, 2))
            A_real, A_imag = real_matmul(rho_real, rho_imag, (H_T_real[:, i+1] - H_T_real[:, i]), (H_T_imag[:, i+1] - H_T_imag[:, i]))
            W += A_real[:, 0, 0] + A_real[:, 1, 1]


        return torch.mean(W)


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_input, output_hidden, batch_size, n_layers, N, bi, dropout, dt):
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
        self.dt = dt
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
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.criterion = work_loss(self.dt, self.N)

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
                optimiser.zero_grad()
                output, internals = self.forward(x, hidden, cell)
                loss = self.criterion.forward(output, x)
                loss.backward()
                optimiser.step()
            self.eval()
            valid_loss = self.calc_loss(valid_set).item()
            scheduler.step(valid_loss)
            if valid_loss < loss_high:
                loss_high = valid_loss
                count = 0
                torch.save(self, 'best_model_custom_loss')
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
            hidden, cell = self.HiddenCellTest(len(X))
            y_pred, internals = self.forward(X, hidden, cell)
            loss = self.criterion.forward(y_pred, X)
        return loss

    def work_ratio(self, data, dt):
        dataset = WorkDataset(data[:, 0], self.N, 'custom_loss')
        with torch.no_grad():
            X = dataset.__getitem__(range(len(dataset)))['x']
            hidden, cell = self.HiddenCellTest(len(X))
            y_pred, internals = self.forward(X, hidden, cell)

        trans_pred = rev_angle_embedding(y_pred, self.N)
        E_pred = np.zeros(len(y_pred))

        for i in range(len(E_pred)):
            E_pred[i] = wrapper(trans_pred[i], data[i, 0][:self.N], data[i, 0][self.N:], dt, data[i, 3], self.N)

        return np.mean(E_pred), np.mean(E_pred / data[:, 2])



def train_lstm_total_dropout(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, rho, net):

    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train[:, 0], N, net=net)
    test_set = WorkDataset(data_test[:, 0], N, net=net)
    valid_set = WorkDataset(data_valid[:, 0], N, net=net)

    torch.manual_seed(seed)
    model = LSTMNetwork(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout, dt).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    elif optimiser == 'adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_custom_loss').eval()
    vloss = model.calc_loss(valid_set)
    # vwork = model.work_ratio(valid_set, dt)

    return epoch, vloss


opt_args = {
    'dropout': 'float',
    'learning_rate': 'float',
    'patience': 'int',
    'batch_size': 'int',
    'n_layers': 'int',
    'bidirectional': 'bool',
    'optimiser': 'string',
    'hidden_size': 'int',
    'hidden_input_1': 'int',
    'hidden_input_2': 'int',
    'hidden_output': 'int',
    'pat_drop': 'float',
    'sched_factor': 'float',
 }

N = 5
dt = 1
seed = 42
rho = 'eigen'
net = 'custom_loss'
N_sobol = 45
runs = range(21)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for k, v in opt_args.items():
        parser.add_argument('--' + k)

    args = parser.parse_args()

    for k, v, in vars(args).items():
        if opt_args[k] == 'float':
            opt_args[k] = float(v)
        elif opt_args[k] == 'int':
            opt_args[k] = int(v)
        elif opt_args[k] == 'bool':
            opt_args[k] = eval(v.capitalize())
        elif opt_args[k] == 'string':
            opt_args[k] = v

    wandb.init(project='custom_loss_dt_{}'.format(dt), config=opt_args)

    epoch, vloss = train_lstm_total_dropout(opt_args['dropout'], opt_args['learning_rate'],opt_args['patience'], opt_args['batch_size'], opt_args['n_layers'], opt_args['bidirectional'], opt_args['hidden_size'], opt_args['hidden_input_1'], opt_args['hidden_input_2'], opt_args['hidden_output'], opt_args['optimiser'], opt_args['pat_drop'], opt_args['sched_factor'], N, dt, rho, net)


    metrics = {
        'n_epochs': epoch,
        'validation_loss': vloss,
        # 'validation_work_ratio': vwork,
    }

    wandb.log(metrics)
