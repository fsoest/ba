import numpy as np
from models import LSTMNetwork, LSTM_total_dropout
from dataset import WorkDataset
import torch
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding
from sklearn.model_selection import train_test_split
from multiproc.pwc_helpers import wrapper


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
device = torch.device('cpu')
def train_lstm(dropout, learning_rate, patience, batch_size, n_layers, bidirectional, hidden_size, hidden_input_1, hidden_input_2, hidden_output, optimiser, N, dt, N_sobol, rho, runs, net):
    data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
    data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
    data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
    train_set = WorkDataset(data_train, N, net)
    test_set = WorkDataset(data_test, N, net)
    valid_set = WorkDataset(data_valid, N, net)

    torch.manual_seed(seed)
    model = LSTMNetwork(5, 4, hidden_size, [hidden_input_1, hidden_input_2], hidden_output, batch_size, n_layers, N, bidirectional, dropout).double()

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
    epoch = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

    model = torch.load('best_model').eval()
    vloss = model.calc_loss(valid_set)
    vwork = model.work_ratio(data_test, dt)
    return epoch, vloss, vwork


# %%
# import matplotlib.pyplot as plt
# x = test_set.__getitem__(range(len(test_set)))['x']
# h, c = model.HiddenCellTest(len(test_set))
# y = model(x, h, c)[0]
# res = rev_angle_embedding(y.detach().numpy(), 5)#.transpose(0, 2, 1).reshape(len(test_set), 2*5)
# res[:, 5:] = res[:, 5:] % (2*np.pi)
# plt.boxplot(res)
#
# dt = np.zeros((len(data_test), 10))
# for i in range(len(data_test)):
#     dt[i] = data_test[:, 1][i]
#
# plt.boxplot(dt)
