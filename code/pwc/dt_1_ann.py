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
from models import two_layer_fcANN


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
train_set = WorkDataset(data_train, N, net='ann')
test_set = WorkDataset(data_test, N, net='ann')
valid_set = WorkDataset(data_valid, N, net='ann')
dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
# %%
model = two_layer_fcANN([2000, 2000, 2000], N, batch_size, dropout=0.3179732914255167).double()
optimiser = torch.optim.SGD(model.parameters(), lr=0.02847, momentum=0.99, dampening=0, weight_decay=0, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.234, patience=55/3.7)
model.learn(train_set, valid_set, optimiser, scheduler)
opt_model = torch.load('models/N_5_ann_dt_1').eval()
opt_model.work_ratio(data_test, dt)

opt_model.calc_loss(test_set)

sum(p.numel() for p in opt_model.parameters() if p.requires_grad)
