import wandb
from train_lstm import train_last_lstm as train_lstm
import argparse
import os


# If you don't want your script to sync to the cloud
# os.environ['WANDB_MODE'] = 'dryrun'

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

 # Default hyperparameters
hyperparameter_defaults = dict(
     dropout = 0.5,
     learning_rate = 5e-3,
     patience = 30,
     batch_size = 30,
     n_layers = 2,
     bidirectional = True,
     optimiser = 'sgd',
     hidden_size = 10,
     hidden_input_1 = 20,
     hidden_input_2 = 20,
     hidden_output = 20,
     pat_drop = 2,
     sched_factor = 0.1,
 )

N = 5
dt = 5
N_sobol = 45
rho = 'eigen'
runs = range(30)
net = 'lstm'

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


    # wandb.init(project='full-drop-lstm-eigen-n-5', config=hyperparameter_defaults)
    # config = wandb.config

    epoch, vloss, vwork = train_lstm(opt_args['dropout'], opt_args['learning_rate'],opt_args['patience'], opt_args['batch_size'], opt_args['n_layers'], opt_args['bidirectional'], opt_args['hidden_size'], opt_args['hidden_input_1'], opt_args['hidden_input_2'], opt_args['hidden_output'], opt_args['optimiser'], opt_args['pat_drop'], opt_args['sched_factor'], N, dt, N_sobol, rho, runs, net)

metrics = {
    'n_epochs': epoch,
    'validation_loss': vloss,
    'validation_work_ratio': vwork,
}

# wandb.log(metrics)
