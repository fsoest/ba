from quantum_cell import Quantum_cell
from multiproc.data_preprocessing import import_datasets
from sklearn.model_selection import train_test_split
from dataset import WorkDataset
import torch
import numpy as np
import argparse
import wandb


opt_args = {
    'dropout': 'float',
    'learning_rate': 'float',
    'patience': 'int',
    'batch_size': 'int',
    'optimiser': 'string',
    'input_size': 'int',
    'output_size': 'int',
    'cell_size': 'int',
    'pat_drop': 'float',
    'sched_factor': 'float',
}

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
seed = 42
dt = 5
N = 5

data = np.load('local_opt_data/N_5_dt_{0}.npy'.format(dt), allow_pickle=True)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)
train_set = WorkDataset(data_train, N, net='lstm')
test_set = WorkDataset(data_test, N, net='lstm')
valid_set = WorkDataset(data_valid, N, net='lstm')
# %%
learning_rate =  opt_args['learning_rate']
sched_factor = opt_args['sched_factor']
patience = opt_args['patience']
pat_drop = opt_args['pat_drop']
dropout = opt_args['dropout']
batch_size = opt_args['batch_size']

input_size = opt_args['input_size']
output_size = opt_args['output_size']
cell_size = opt_args['output_size']

wandb.init(project='quantum_cell_dt_{0}'.format(dt), config=opt_args)

# %%
model = Quantum_cell(5, 4, input_size, output_size, cell_size, batch_size, dropout).double()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=sched_factor, patience=patience/pat_drop)
epoch, valid_loss = model.learn(train_set, valid_set, optimiser, scheduler, patience=patience)

metrics = {
    'epoch': epoch,
    'vloss': valid_loss,
}

wandb.log(metrics)
