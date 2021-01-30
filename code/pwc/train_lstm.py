import numpy as np
from models import LSTMNetwork, LSTM_total_dropout, FCNN_LSTM, C_LSTM, LastLSTM, LSTM_batch_norm
from dataset import WorkDataset
import torch
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding
from sklearn.model_selection import train_test_split
from multiproc.pwc_helpers import wrapper, angles_to_states
from scipy.linalg import expm


seed = 42

# dropout = 0.2
# learning_rate = 6e-3
# patience = 10
# batch_size = 50
# n_layers = 2
# bidirectional = True
# optimiser = 'adam'
# hidden_size = 20
# hidden_input_1 = 20
# hidden_input_2 = 20
# hidden_output = 20

def get_angles(kets):
    N = kets.shape[1]
    angles = np.zeros((kets.shape[0], 2 * N))
    angles[:, :N] = 2 * np.arctan2(np.abs(kets[:, :, 1]),np.abs(kets[:, :, 0]))
    angles[:, N:] = (np.angle(kets[:, :, 1]) - np.angle(kets[:, :, 0])) % (2 * np.pi)
    return angles


def rand_herm():
    rand = np.random.uniform(low=-1, high=1, size=8)
    rand /= np.linalg.norm(rand)
    comp = np.zeros(4, dtype=np.complex128)
    for i in range(len(comp)):
        comp[i] = rand[i] + 1j * rand[i + 4]
    H = comp.reshape(2, 2)

    H = (H + H.T.conj())/2
    return H

def rand_unit(tau):
    H = rand_herm()
    return expm(-1j * H * tau)


device = torch.device('cpu')
def train_lstm(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LSTMNetwork(5, 2, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=patience/2)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model').eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork

def train_lstm_total_dropout(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LSTM_total_dropout(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    elif optimiser == 'adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


def train_local_opt(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, net):
    data = np.load('local_opt_data/N_{0}_dt_{1}.npy'.format(N, dt))
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LSTM_total_dropout(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    elif optimiser == 'adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


def train_fnn_lstm(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = FCNN_LSTM(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


def train_c_lstm(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output_1, hidden_output_2, hidden_output_3, optimiser, pat_drop, sched_factor, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = C_LSTM(5, 4, hidden_size, [hidden_input_1, hidden_input_2, hidden_input_3], [hidden_output_1, hidden_output_2,hidden_output_3], batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


def train_last_lstm(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LastLSTM(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


def train_lstm_noisy(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)

    # Add noise to training data
    N_data_init = 3000
    data_train = data_train[:N_data_init]

    n_noise = 3
    for n in range(n_noise):
        data_train = np.concatenate((data_train, data_train))
    # Array of tau values
    taus = np.linspace(0, 1, 8)
    for i in range(N_data_init):
        # Kets from angles
        kets = angles_to_states(data[i, 0])
        for j, tau in enumerate(taus[1:]):
            noisy = np.zeros((N, 2), dtype=np.complex128)
            for n in range(N):
                # Random unitary
                U = rand_unit(tau)
                noisy[n] = U @ kets[n]
            noisy_angles = get_angles(noisy[np.newaxis])
            data_train[j * N_data_init + i, 0] = noisy_angles[0]

    # Datesets
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LSTM_total_dropout(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


def train_batch_lstm(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, pat_drop, sched_factor, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LSTM_batch_norm(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

    if optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
    elif optimiser == 'adagrad':
        optimiser = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model_{}'.format(dropout)).eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork
