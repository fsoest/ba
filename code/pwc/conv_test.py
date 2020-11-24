import numpy as np
import tensorflow as tf
from data_preprocessing import angle_embedding, equivalent_vectors
from sklearn.model_selection import train_test_split
from pwc_helpers import model_evaluation as model_eval
import matplotlib.pyplot as plt
from models import CustomSchedule
# %%
N = 5
X = np.load('train_data/x_N_5_xyz.npy')
y = np.load('train_data/y_N_5_ang.npy')
for i, val in enumerate(y):
    y[i] = equivalent_vectors(val, N)
# %%
input = angle_embedding(X, N)
output = angle_embedding(y, N)
new_o = np.zeros((len(output), 20))
input_exp = np.expand_dims(input, axis=3)
for i, val in enumerate(output):
    new_o[i] = np.reshape(output[i], (N * 4,))
X_train, X_test, y_train, y_test = train_test_split(input_exp, new_o, test_size=0.15, random_state=42)
# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(20, (3, 4), activation='relu', input_shape=(N, 4, 1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4 * N))
learning_rate = CustomSchedule(5, 100)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
# %%
model.fit(X_train, y_train, epochs=10000, verbose=1, validation_split=0.1)

# %%
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state

E_pred, E_actual = model_eval(model, X_test, y_test, dt, rho_0, N)
np.mean(E_pred)/np.mean(E_actual)
# %%
plt.plot(range(len(E_pred)), E_pred, label='pred')
plt.plot(range(len(E_pred)), E_actual, label='actual')
plt.legend()

# %%
from scipy.linalg import expm
def int_operator(theta, phi):
    """
    Returns partial system Hamiltonian, A * sigma_p
    """
    sigma_p = np.matrix([[0, 0], [1, 0]], dtype=np.complex128)
    sigma_m = np.matrix([[0, 1], [0, 0]], dtype=np.complex128)
    return np.sin(theta) * np.exp(1j * phi) * sigma_p / 2 + np.sin(theta) * np.exp(-1j * phi) * sigma_m / 2


def H_s(theta_d, phi_d, a, b):
    """
    """
    drive_op = int_operator(theta_d, phi_d)
    trans_op = int_operator(a, b)
    return drive_op + trans_op


def energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N):
    E = np.zeros(N, dtype=np.complex128)
    rho = rho_0
    for i in range(N - 1):
        U = expm(-1j * dt * H_s(theta_d[i], phi_d[i], theta_t[i], phi_t[i]))
        # Time evolution of state density matrix
        rho = U @ rho @ np.matrix(U).H
        # Change in Hamiltonian due to Transducer, Drive constant
        delta_trans = np.sin(theta_t[i + 1]) * np.exp(1j * phi_t[i + 1]) - np.sin(theta_t[i]) * np.exp(1j * phi_t[i])
        dH = int_operator(theta_t[i+1], phi_t[i+1]) - int_operator(theta_t[i], phi_t[i])
        E[i] = np.trace(rho @ dH)

    return np.cumsum(E.real)[-1]


def wrapper(transducer, theta_d, phi_d, dt, rho_0, N):
    theta_t = transducer[:int(len(transducer)/2)]
    phi_t = transducer[int(len(transducer)/2):]
    return energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N)


def model_eval(model, X_test, y_test, dt, rho_0, N):
    """
    Evaluates a given model, gives E_pred, E_actual as output
    """
    E_pred = np.zeros(len(X_test))
    E_actual = np.zeros(len(X_test))
    y_pred = model.predict(X_test)
    pred_reshape = np.zeros((len(X_test), N, 4))
    actual_reshape = np.zeros((len(X_test), N, 4))
    y_trans_test = np.zeros((len(X_test), 2 * N))
    y_trans_pred = np.zeros((len(X_test), 2 * N))
    for i, x in enumerate(X_test[:, :, :, 0]):
        theta_d = np.arctan2(x[:, 0], x[:, 2])
        phi_d = np.arctan2(x[:, 1], x[:, 3])
        pred_reshape[i] = np.reshape(y_pred[i], (N, 4))
        actual_reshape[i] = np.reshape(y_test[i], (N, 4))
        y_trans_test[i, :N] = np.arctan2(actual_reshape[i, :, 0], actual_reshape[i, :, 2])
        y_trans_test[i, N:] = np.arctan2(actual_reshape[i, :, 1], actual_reshape[i, :, 3])
        y_trans_pred[i, :N] = np.arctan2(pred_reshape[i, :, 0], pred_reshape[i, :, 2])
        y_trans_pred[i, N:] = np.arctan2(pred_reshape[i, :, 1], pred_reshape[i, :, 3])
        E_pred[i] = wrapper(y_trans_pred[i], theta_d, phi_d, dt, rho_0, N)
        E_actual[i] = wrapper(y_trans_test[i], theta_d, phi_d, dt, rho_0, N)
    return E_pred, E_actual
