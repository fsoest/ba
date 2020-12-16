import numpy as np
from models import SimpleRNN
from dataset import WorkDataset
from torch.utils.data import DataLoader
import torch
from multiproc.data_preprocessing import import_datasets, rev_angle_embedding
from sklearn.model_selection import train_test_split
from multiproc.pwc_helpers import wrapper
# model = SimpleRNN()

N = 3
batch_size = 30
seed = 42
dt = 5
rho = 'eigen'
N_sobol = 10

data = import_datasets('multi_train_data', N, dt, rho, N_sobol, [0, 1, 2])
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
# %%
train_set = WorkDataset(data_train, N, embed=True)
test_set = WorkDataset(data_test, N, embed=True)

dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
torch.manual_seed(seed)
model = SimpleRNN(4, 4, 10, batch_size)
model = model.double()

learning_rate = 1e-2
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
# %%
# torch.device('cpu')
for epoch in range(20):
    for i, batch in enumerate(dataloader):
        hidden = model.initHidden()
        # Loop over batches
        output = torch.zeros((batch_size, N, 4), dtype=torch.float64)
        x = torch.squeeze(batch['x'])
        y = torch.squeeze(batch['y'])
        optimiser.zero_grad()
        for j in range(N):
            # RNN steps
            output[:, j], hidden = model(x[:, j], hidden)
        loss = criterion(output, y)
        loss.backward()
        optimiser.step()
    print(loss.item())
# %%
with torch.no_grad():
    X_test = test_set.__getitem__(range(len(test_set)))['x']
    y_pred = torch.zeros((len(test_set), N, 4), dtype=torch.float64)
    hidden = model.HiddenTest(len(test_set))
    for j in range(N):
        # RNN steps
        y_pred[:, j], hidden = model(X_test[:, j], hidden)

trans_pred = rev_angle_embedding(y_pred, N)
E_pred = np.zeros(len(y_pred))

for i in range(len(E_pred)):
    E_pred[i] = wrapper(trans_pred[i], data_test[i, 0][:N], data_test[i, 0][N:], dt, data_test[i, 3], N)

100 * np.mean(E_pred / data_test[:, 2])
x.shape
# %%

import matplotlib.pyplot as plt
plt.plot(data_test[:, 2])

plt.plot(E_pred)
np.mean(E_pred)/np.mean(data_test[:, 2])
