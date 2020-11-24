import numpy as np
import tensorflow as tf
from data_preprocessing import angle_embedding, equivalent_vectors
from sklearn.model_selection import train_test_split
from pwc_helpers import model_evaluation_embedded, model_evaluation_convolution
import matplotlib.pyplot as plt
from models import CustomSchedule
# %%
X_train = np.load('train_data/X_N_20_10000_dt_1.npy')
y_train = np.load('train_data/y_N_20_10000_dt_1.npy')
X_test = np.load('train_data/X_N_20_2000_dt_1.npy')
y_test = np.load('train_data/y_N_20_2000_dt_1.npy')

N = 20
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state

input = angle_embedding(X_train, N)
output = angle_embedding(y_train, N)

# Expan dimension for Convolutions
# input_exp = np.expand_dims(input, axis=3)

new_o = np.zeros((len(output), N * 4))
for i, val in enumerate(output):
    new_o[i] = np.reshape(output[i], (N * 4,))
input.shape
# %%
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4 * N))
learning_rate = CustomSchedule(5, 3000)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
model.summary()
# %%
model.fit(input, new_o, epochs=20000, verbose=1, validation_split=0.1)

# %%
X_test_embed = angle_embedding(X_test, N)
y_test_embed = angle_embedding(y_test, N)
E_pred, E_actual = model_evaluation_embedded(model, X_test_embed, y_test_embed, dt, rho_0, N)
np.mean(E_pred)/np.mean(E_actual)

# %%
plt.plot(range(len(E_pred)), E_pred, label='pred')
plt.plot(range(len(E_pred)), E_actual, label='actual')
plt.legend()
