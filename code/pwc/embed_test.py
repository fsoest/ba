import numpy as np
import tensorflow as tf
from data_preprocessing import angle_embedding, equivalent_vectors
from sklearn.model_selection import train_test_split
from pwc_helpers import model_evaluation_embedded
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
for i, val in enumerate(output):
    new_o[i] = np.reshape(output[i], (N * 4,))
X_train, X_test, y_train, y_test = train_test_split(input, new_o, test_size=0.15, random_state=42)

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
model.add(tf.keras.layers.Dense(4 * N))
learning_rate = CustomSchedule(5, 1000)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
# %%
model.fit(X_train, y_train, epochs=2000, verbose=1, validation_split=0.1)

# %%
dt = 1
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state


E_pred, E_actual = model_evaluation_embedded(model, X_test, y_test, dt, rho_0, N)

# %%
plt.plot(range(len(E_pred)), E_pred, label='pred')
plt.plot(range(len(E_pred)), E_actual, label='actual')
plt.legend()
np.mean(E_pred)/np.mean(E_actual)
np.mean(E_actual)
len(E_pred[E_pred < 0]) / len(E_pred)
