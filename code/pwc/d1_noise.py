import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from multiproc.pwc_helpers import angles_to_states, state_to_angles, wrapper
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding, import_datasets
from sklearn.model_selection import train_test_split
import qutip as qt


s_x = np.array([[0, 1], [1, 0]], dtype=np.complex256)
s_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex256)
s_z = np.array([[1, 0], [0, 1]], dtype=np.complex256)

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

def fidelity(a, b):
    """
    Calculates fidelity of two pure states a, b
    """
    return np.abs(np.dot(a, np.conj(b)))

def get_angles(kets):
    N = kets.shape[1]
    angles = np.zeros((kets.shape[0], 2 * N))
    angles[:, :N] = 2 * np.arctan2(np.abs(kets[:, :, 1]),np.abs(kets[:, :, 0]))
    angles[:, N:] = (np.angle(kets[:, :, 1]) - np.angle(kets[:, :, 0])) % (2 * np.pi)
    return angles


def noisy_drive(angles, runs, tau, model, dt, rho_0, N):
    """
    angles: input
    runs: amount of times to create random noise per angle
    """
    kets = angles_to_states(angles)
    N = len(angles) // 2
    noisy_kets = np.zeros((runs, N, 2), dtype=np.complex128)
    noisy_angles = np.zeros((runs, 2*N))
    fid = np.full(runs, 1.0)
    # Initialise random unitaries
    U = np.zeros((runs, N, 2, 2), dtype=np.complex128)
    for run in range(runs):
        for n in range(N):
            U[run, n] = rand_unit(tau)
            noisy_kets[run, n] = U[run, n] @ kets[n]
            fid[run] *= fidelity(noisy_kets[run, n], kets[n])
    noisy_angles = get_angles(noisy_kets)
    # Calculate work outputs
    X_clean = torch.from_numpy(angle_embedding(angles[np.newaxis], N))
    with torch.no_grad():
        hidden, cell = model.HiddenCellTest(len(X_clean))
        y_pred, internals = model.forward(X_clean, hidden, cell)
    trans_pred = rev_angle_embedding(y_pred.detach().numpy(), N)
    work = wrapper(trans_pred[0], angles[:N], angles[N:], dt, rho_0, N)
    noisy_work = np.zeros(runs)
    for run in range(runs):
        noisy_work[run] = wrapper(trans_pred[0], noisy_angles[run, :N], noisy_angles[run, N:], dt, rho_0, N)
    return work, noisy_work, fid


def noisy_trans(angles, runs, tau, model, dt, rho_0, N):
    # Predict transducer
    X_clean = torch.from_numpy(angle_embedding(angles[np.newaxis], N))
    with torch.no_grad():
        hidden, cell = model.HiddenCellTest(len(X_clean))
        y_pred, internals = model.forward(X_clean, hidden, cell)
    trans_pred = rev_angle_embedding(y_pred.detach().numpy(), N)
    work = wrapper(trans_pred[0], angles[:N], angles[N:], dt, rho_0, N)

    noisy_work = np.zeros(runs)

    kets = angles_to_states(trans_pred[0])
    N = len(angles) // 2
    noisy_kets = np.zeros((runs, N, 2), dtype=np.complex128)
    noisy_angles = np.zeros((runs, 2*N))
    fid = np.full(runs, 1.0)
    # Initialise random unitaries
    U = np.zeros((runs, N, 2, 2), dtype=np.complex128)
    for run in range(runs):
        for n in range(N):
            U[run, n] = rand_unit(tau)
            noisy_kets[run, n] = U[run, n] @ kets[n]
            fid[run] *= fidelity(noisy_kets[run, n], kets[n])
    noisy_angles = get_angles(noisy_kets)
    for run in range(runs):
        noisy_work[run] = wrapper(noisy_angles[run], angles[:N], angles[N:], dt, rho_0, N)
    return work, noisy_work, fid


def fidelities(angles, runs, tau):
    kets = angles_to_states(angles)
    N = len(angles) // 2
    noisy_kets = np.zeros((runs, N, 2), dtype=np.complex128)
    noisy_angles = np.zeros((runs, 2*N))
    fidelities = np.full(runs, 1.0)
    # Initialise random unitaries
    U = np.zeros((runs, N, 2, 2), dtype=np.complex128)
    for run in range(runs):
        for n in range(N):
            U[run, n] = rand_unit(tau)
            noisy_kets[run, n] = U[run, n] @ kets[n]
            fidelities[run] *= fidelity(noisy_kets[run, n], kets[n])
    return fidelities

# %%
N = 1000

x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
c = np.zeros(N)
dh = np.zeros(N)

for i in range(N):
    herm = rand_herm()
    x[i] = np.trace(s_x @ herm)
    y[i] = np.trace(s_y @ herm)
    z[i] = np.trace(s_z @ herm)
    c[i] = np.trace((s_x + s_y) @ herm / np.sqrt(2))
    w, v = np.linalg.eigh(herm)
    dh[i] = w[1] - w[0]

plt.hist(x, label='x', bins=30)
plt.hist(y, label='y', bins=30)
plt.hist(z, label='z', bins=30)
plt.hist(c, label='c', bins=30)
plt.legend()
# %%
seed = 42
np.random.seed(seed)
# %%
taus = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for tau in taus[1:]:
    fidel = fidelities(data[500, 0], 1000, tau)
    plt.hist(fidel, label=tau)
plt.legend()
# plt.plot(std, avg_fidel)
# %%
import torch
model = torch.load('models/dt_1_bi').eval()
model
seed = 42
N = 5
dt = 1
N_sobol = 45
rho = 'eigen'
runs = range(21)
net = 'lstm'

data = import_datasets('multi_train_data', N, dt, rho, N_sobol, runs)
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
data_train, data_valid = train_test_split(data_train, test_size=0.1, random_state=seed)

# %%
data_points = len(data_test)
noisy_work_drive = np.zeros((len(taus), data_points))
work_drive = np.zeros((len(taus), data_points))
avg_fidel_drive = np.zeros((len(taus), data_points))
for i, tau in enumerate(taus):
    for j in range(data_points):
        work, noisy_work, fid = noisy_drive(data_test[j, 0], 100, tau, model, dt, data_test[j, 3], 5)
        noisy_work_drive[i, j] = np.mean(noisy_work)
        work_drive[i, j] = work
        avg_fidel_drive[i, j] = np.mean(fid)


# %%
e_opt = np.zeros(len(noisy_work_drive[2]))
for i, d in enumerate(data_test):
    e_opt = d[2]

plt.boxplot(-1*noisy_work_drive[2] + e_opt)
# %%
data_points = len(data_test)
noisy_work_trans = np.zeros((len(taus), data_points))
work_trans = np.zeros((len(taus), data_points))
avg_fidel_trans = np.zeros((len(taus), data_points))
for i, tau in enumerate(taus):
    for j in range(data_points):
        work, noisy_work, fid = noisy_trans(data_test[j, 0], 100, tau, model, dt, data_test[j, 3], 5)
        noisy_work_trans[i, j] = np.mean(noisy_work)
        work_trans[i, j] = work
        avg_fidel_trans[i, j] = np.mean(fid)

# %%
np.save('noise/noisy_work_drive_dt_1', noisy_work_drive)
np.save('noise/work_drive_dt_1', work_drive)
np.save('noise/avg_fidel_drive_dt_1', avg_fidel_drive)

np.save('noise/noisy_work_trans_dt_1', noisy_work_trans)
np.save('noise/work_trans_dt_1', work_trans)
np.save('noise/avg_fidel_trans_dt_1', avg_fidel_trans)
# %%
plt.scatter(avg_fidel[i], ratio_avg[i], label=tau, alpha=0.1)
plt.legend(title='$\\tau$')
plt.xlabel('$\overline{F}_T$')
plt.ylabel('$\epsilon$')
# plt.savefig('/home/fsoest/ba/phystex/img/noisy_trans_bi_true.png', dpi=300)

# %%
noisy_work_drive = np.load('noise/noisy_work_drive_dt_1.npy')
work_drive = np.load('noise/work_drive_dt_1.npy')
avg_fidel_drive = np.load('noise/avg_fidel_trans_dt_1.npy')
noisy_work_trans = np.load('noise/noisy_work_trans_dt_1.npy')
work_trans = np.load('noise/work_trans_dt_1.npy')
avg_fidel_trans = np.load('noise/avg_fidel_trans_dt_1.npy')
# %%
for i, tau in enumerate(taus):
    trans_delta = (-1 * noisy_work_trans[i] + work_trans[i])#/(np.abs(noisy_work_trans)[i] + np.abs(work_trans)[i])
    plt.scatter(avg_fidel_trans[i], trans_delta, label=tau, alpha=0.005)
    plt.scatter(np.mean(avg_fidel_trans[i]), np.mean(trans_delta), c='k', marker='.')
leg = plt.legend(title='$\\tau$')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel('$\overline{F}_T$')
plt.xlim(0.64, 1.01)
plt.ylabel('$\Delta W$')
plt.savefig('/home/fsoest/ba/phystex/img/noisy_trans_dt_1.png', dpi=300)

# %%
for i, tau in enumerate(taus):
    drive_delta = (-1 * noisy_work_drive[i] + work_drive[i])#/(np.abs(noisy_work_drive)[i] + np.abs(work_drive)[i])
    plt.scatter(avg_fidel_drive[i], drive_delta, label=tau, alpha=0.005)
    plt.scatter(np.mean(avg_fidel_drive[i]), np.mean(drive_delta), c='k', marker='.')
leg = plt.legend(title='$\\tau$')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel('$\overline{F}_D$')
plt.xlim(0.64, 1.01)
plt.ylabel('$\Delta W$')
plt.savefig('/home/fsoest/ba/phystex/img/noisy_drive_dt_1.png', dpi=300)
