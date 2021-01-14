from multiproc.data_preprocessing import import_datasets
from multiproc.pwc_helpers import angles_to_states
import numpy as np
from scipy.linalg import expm


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

N = 5
dt = 5
rho = 'eigen'
N_sobol = 45
runs = range(5)

tau = 0.4
data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
len(data)
n_noise = 3

for n in range(n_noise):
    data = np.concatenate((data, data))

len(data)

taus = np.linspace(0, 1, 8)

N_data_init = 10000
for i in range(N_data_init):
    kets = angles_to_states(data[i, 0])
    for j, tau in enumerate(taus[1:]):
        noisy = np.zeros((N, 2), dtype=np.complex128)
        for n in range(N):
            U = rand_unit(tau)
            noisy[n] = U @ kets[n]
        noisy_angles = get_angles(noisy[np.newaxis])
        data[j * N_data_init + i, 0] = noisy_angles[0]

np.save('multi_train_data/N_5_noisy', data)
