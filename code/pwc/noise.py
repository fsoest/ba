import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from multiproc.pwc_helpers import angles_to_states, state_to_angles, wrapper

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
    return np.abs(np.dot(a, b))**2

# %%
N = 1000

x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
dh = np.zeros(N)

for i in range(N):
    herm = rand_herm()
    x[i] = np.trace(s_x @ herm)
    y[i] = np.trace(s_y @ herm)
    z[i] = np.trace(s_z @ herm)
    w, v = np.linalg.eigh(herm)
    dh[i] = w[1] - w[0]

plt.hist(x, label='x', bins=30)
plt.hist(y, label='y', bins=30)
plt.hist(z, label='z', bins=30)
plt.legend()
# %%


def noisy_work(angles, runs, std, model, dt, rho_0):
    kets = angles_to_states(angles)
    N = len(angles) // 2
    noisy_kets = np.zeros((runs, N, 2), dtype=np.complex128)
    noisy_angles = np.zeros((runs, 2*N))
    fidelities = np.full(runs, 1.0)
    # Initialise random unitaries
    U = np.zeros((runs, N, 2, 2), dtype=np.complex128)
    taus = np.random.normal(0, scale=std, size=(runs, N))
    for run in range(runs):
        for n in range(N):
            U[run, n] = rand_unit(taus[run, n])
            noisy_kets[run, n] = U[run, n] @ kets[n]
            fidelities[run] *= fidelity(noisy_kets[run, n], kets[n])
        noisy_angles[run] = state_to_angles(noisy_kets[run])

    # Calculate work outputs
    trans_pred = model(angles)
    work = wrapper(trans_pred, angles[:N], angles[N:], dt, rho_0, N)
    noisy_work = np.zeros(runs)
    for run in range(runs):
        noisy_work[run] = wrapper(trans_pred, noisy_angles[run, :N], noisy_angles[run, N:], dt, rho_0, N)
