import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import tensorflow as tf
from sklearn.model_selection import train_test_split
# %%

def int_operator(theta, phi):
  """
  Returns partial system Hamiltonian, A * sigma_p
  """
  sigma_p = np.matrix([[0, 0], [1, 0]], dtype=np.complex128)
  sigma_m = np.matrix([[0, 1], [0, 0]], dtype=np.complex128)
  return np.sin(theta) * np.exp(1j * phi) * sigma_p / 2 + np.sin(theta) * np.exp(-1j * phi) * sigma_m / 2


def xyz_operator(a, c, d):
    """
    Partial system Hamiltonian, parametrised by a, b = c + id
    """
    sigma_p = np.matrix([[0, 0], [1, 0]], dtype=np.complex128)
    sigma_m = np.matrix([[0, 1], [0, 0]], dtype=np.complex128)
    b = c + 1j * d
    return a * (b * sigma_p + np.conj(b) * sigma_m) / (a**2 + c**2 + d**2)


def H_s(theta_d, phi_d, a, c, d):
    """
    """
    drive_op = int_operator(theta_d, phi_d)
    trans_op = xyz_operator(a, c, d)
    return drive_op + trans_op


def energy(theta_d, phi_d, a, c, d, dt, rho_0, N):
    E = np.zeros(N, dtype=np.complex128)
    rho = rho_0
    for i in range(N - 1):
        U = expm(-1j * dt * H_s(theta_d[i], phi_d[i], a[i], c[i], d[i]))
        # Time evolution of state density matrix
        rho = U @ rho @ np.matrix(U).H
        # Change in Hamiltonian due to Transducer, Drive constant
        dH = xyz_operator(a[i + 1], c[i + 1], d[i + 1]) - xyz_operator(a[i], c[i], d[i])
        E[i] = np.trace(rho @ dH)

    return np.cumsum(E.real)[-1]


def wrapper(transducer, theta_d, phi_d, dt, rho_0, N):
    theta_t = transducer[:int(len(transducer)/2)]
    phi_t = transducer[int(len(transducer)/2):]
    a = transducer[:N]
    c = transducer[N:2 * N]
    d = transducer[2 * N:]
    return energy(theta_d, phi_d, a, c, d, dt, rho_0, N)

# %%
N = 5
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state

theta_d = np.linspace(0, 2*np.pi, 25)
phi_d = np.cos(np.linspace(0, 5, 25)) * np.pi * 2

# %%
%%timeit
res = minimize(wrapper, np.full(3 * N, 0.5), args=(theta_d, phi_d, dt, rho_0, N))
# %%
wrapper(res.x, theta_d, phi_d, dt, rho_0, N)

    # %%
def custom_loss(y_actual, y_pred):
    custom_loss = tf.keras.backend.square(y_actual - tf.keras.backend.l2_normalize(y_pred))
    return custom_loss

# %%
N_tot = 16800
X = np.zeros((N_tot, 2 * N))
y = np.zeros((N_tot, 3 * N))
for i in range(N_tot):
    theta_d = np.random.uniform(0, np.pi, N)
    phi_d = np.random.uniform(0, 2 * np.pi, N)
    res = minimize(wrapper, np.full(3 * N, 0.5), args=(theta_d, phi_d, dt, rho_0, N))
    X[i][:N] = theta_d
    X[i][N:] = phi_d
    y[i] = res.x
    if i % 1000 == 0:
        print(i)


np.save('train_data/x_N_5', X)
np.save('train_data/y_N_5', y)


# %%
X = np.load('train_data/x_N_5_xyz.npy')
y = np.load('train_data/y_N_5_xyz.npy')


# %%
for val in y:
    own_normalise(val, N)
X.shape
# %%
input = np.zeros((16800, N, 4))

for i, x in enumerate(X):
    a = np.reshape(x, (2, N)).T
    input[i,:, :2] = np.sin(a)
    input[i,:, 2:] = np.cos(a)

X_train, X_test, y_train, y_test = train_test_split(input, y, test_size=0.15, random_state=42)

input[0]
np.reshape(X[0], (2, N)).T
np.arctan2(input[0, :, 1], input[0, :, 3])


# %%
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3 * N))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mse')

# %%
model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.1)

# %%



E_pred = np.zeros(len(X_test))
E_actual = np.zeros(len(X_test))
y_pred = model.predict(X_test)
rand = np.random.uniform(-1, 1, (2520, 3 * N))
E_rand = np.zeros(len(X_test))
# %%
for i, x in enumerate(X_test):
    theta_d = np.arctan2(x[:, 0], x[:, 2])
    phi_d = np.arctan2(x[:, 1], x[:, 3])
    E_pred[i] = wrapper(y_pred[i], theta_d, phi_d, dt, rho_0, N)
    E_actual[i] = wrapper(y_test[i], theta_d, phi_d, dt, rho_0, N)
    E_rand[i] = wrapper(rand[i], theta_d, phi_d, dt, rho_0, N)

# %%
plt.plot(range(len(E_pred)), E_pred, label='pred')
plt.plot(range(len(E_pred)), E_actual, label='actual')
plt.plot(range(len(delta)), delta, label='delta')
plt.legend()
# %%
np.mean(E_pred)/np.mean(E_actual)
np.mean(E_actual)
# %%
delta_rand = E_rand - E_actual

len(E_pred[E_pred < 0]) / len(E_pred)
