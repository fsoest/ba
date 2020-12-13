from multiproc.pwc_helpers import wrapper, model_evaluation_embedded
import numpy as np
from sklearn.model_selection import train_test_split
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
import tensorflow as tf
import matplotlib.pyplot as plt
from models import CustomSchedule
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping


# %%
N = 3
dt = 5
# Load Data
seed = 42

d1 = np.load('multi_train_data/N_3/dt_5_eigen_sobol_10_run_0.npy', allow_pickle=True)
d2 = np.load('multi_train_data/N_3/dt_5_eigen_sobol_10_run_1.npy', allow_pickle=True)
d3 = np.load('multi_train_data/N_3/dt_5_eigen_sobol_10_run_2.npy', allow_pickle=True)

data = np.concatenate((d1, d2, d3))

# Train test split
data_train, data_test = train_test_split(data, test_size=0.18, random_state=seed)
# We remove last time step, as it does not alter dynamics
X_train = angle_embedding(data_train[:, 0], N)[:, :N - 1]
y_train = angle_embedding(data_train[:, 1], N, reshape=True)
X_test = angle_embedding(data_test[:, 0], N)[:, :N - 1]

# %%

# Model initialisation
tf.random.set_seed(seed)
callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N - 1, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(4 * N))

# learning_rate = CustomSchedule(4, 500)
learning_rate = ExponentialDecay(1e-2, 10000, 0.96)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')

# %%
model.summary()
history = model.fit(X_train, y_train, epochs=10000, verbose=1, validation_split=0.1, callbacks=[callback])
# %%
# Model prediciton output
trans_pred = rev_angle_embedding(model.predict(X_test), N, reshape=True) % (2*np.pi)
E_pred = np.zeros(len(X_test))

for i in range(len(E_pred)):
    E_pred[i] = wrapper(trans_pred[i], data_test[i, 0][:N], data_test[i, 0][N:], dt, data_test[i, 3], N)

np.mean(E_pred / data_test[:, 2])


# %%
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'])

# %%
trans_pred = rev_angle_embedding(model.predict(X_train), N, reshape=True) % (2*np.pi)
E_pred = np.zeros(len(X_train))

for i in range(len(E_pred)):
    E_pred[i] = wrapper(trans_pred[i], data_train[i, 0][:N], data_train[i, 0][N:], dt, data_train[i, 3], N)

np.mean(E_pred / data_train[:, 2])
# %%
plt.boxplot(trans_pred)
