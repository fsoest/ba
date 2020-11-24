import numpy as np
import tensorflow as tf
from data_preprocessing import angle_embedding
from sklearn.model_selection import train_test_split
from pwc_helpers import model_evaluation
import matplotlib.pyplot as plt
from scipy.linalg import expm
from models import CustomSchedule
# %%
N = 5
X = np.load('train_data/x_N_5_xyz.npy')
y = np.load('train_data/y_N_5_ang.npy')
y_cart = np.zeros((len(y), 3 * N))
y_cart[:, :N] = np.cos(y[:, :N] / 2)
y_cart[:, N: 2 * N] = np.sin(y[:, :N] / 2) * np.cos(y[:, N:])
y_cart[:, 2 * N: 3 * N] = np.sin(y[:, :N] / 2) * np.sin(y[:, N:])

input = angle_embedding(X, N)

X_train, X_test, y_train, y_test = train_test_split(input, y_cart, test_size=0.15, random_state=42)

# %%
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(3 * N))
learning_rate = CustomSchedule(5, 1000)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')

# %%
model.fit(X_train, y_train, epochs=2000, verbose=1, validation_split=0.1)

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


def model_evaluation(model, X_test, y_test, dt, rho_0, N):
    """
    Evaluates a given model, gives E_pred, E_actual as output
    """
    E_pred = np.zeros(len(X_test))
    E_actual = np.zeros(len(X_test))
    y_pred = model.predict(X_test)
    # rand = np.random.uniform(-1, 1, (2520, 3 * N))
    # E_rand = np.zeros(len(X_test))
    for i, x in enumerate(X_test):
        theta_d = np.arctan2(x[:, 0], x[:, 2])
        phi_d = np.arctan2(x[:, 1], x[:, 3])
        E_pred[i] = wrapper(y_pred[i], theta_d, phi_d, dt, rho_0, N)
        E_actual[i] = wrapper(y_test[i], theta_d, phi_d, dt, rho_0, N)
        # E_rand[i] = wrapper(rand[i], theta_d, phi_d, dt, rho_0, N)
    return E_pred, E_actual

# %%
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state
E_pred, E_actual = model_evaluation(model, X_test, y_test, dt, rho_0, N)
np.mean(E_pred)/np.mean(E_actual)
np.mean(E_actual)
len(E_pred[E_pred < 0]) / len(E_pred)

# %%
plt.plot(range(len(E_pred)), E_pred, label='pred')
plt.plot(range(len(E_pred)), E_actual, label='actual')
plt.legend()
