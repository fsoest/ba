from pwc_helpers import wrapper, model_evaluation_embedded
import numpy as np
from N2_analyt import E
from scipy.optimize import minimize
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_preprocessing import angle_embedding, rev_angle_embedding
from models import CustomSchedule
import matplotlib.pyplot as plt
# %%
N = 2
X = np.load('train_data/X_N_2_50000_dt_3.npy')
y = np.load('train_data/y_N_2_50000_dt_3.npy')
X_embed = angle_embedding(X, N)
y_embed = angle_embedding(y, N, reshape=True)

X_train, X_test, y_train, y_test = train_test_split(X_embed, y_embed, test_size=0.18, random_state=42)
# %%
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4 * N))

learning_rate = CustomSchedule(4, 200)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
# %%
model.fit(X_train, y_train, epochs=1000, verbose=1, validation_split=0.1)

# %%
N = 2
dt = 3
zero_state = np.matrix([1, 0], dtype=np.complex128)
rho_0 = zero_state.H @ zero_state
# %%
E_pred, E_actual = model_evaluation_embedded(model, X_test, y_test, dt, rho_0, N)

np.mean(E_pred[E_actual!=0]/E_actual[E_actual!=0])
# %%
plt.scatter(range(len(E_pred)), -E_pred, label='Predicted', marker='.')
plt.scatter(range(len(E_actual)), -E_actual, label='Test', marker='.')
plt.legend()
# %%
import qutip as qt
from plot_helpers import tp_to_uvw
# %%
X_low = rev_angle_embedding(X_test[E_pred>0], 2)
y_low = rev_angle_embedding(y_test[E_pred>0], 2, reshape=True)

theta_low = X_low[:3, 0]
phi_low = X_low[:3, 2]

drives = tp_to_uvw(theta_low, phi_low)
trans = tp_to_uvw(y_low[:3, 0], y_low[:3, 2])
trans_prime = tp_to_uvw(y_low[:3, 1], y_low[:3, 3])

# %%
b = qt.Bloch()
b.add_points(drives)
b.add_points(trans)
b.add_points(trans_prime)
b.show()
 # %%
X.shape
y.shape
# %%
plt.plot(range(50), X[:50, 0], label='Theta Drive')
plt.plot(range(50), X[:50, 2], label='Phi Drive')
plt.plot(range(50), y[:50, 0], label='Theta trans')
plt.plot(range(50), y[:50, 2], label='Phi trans')
plt.plot(range(50), y[:50, 1], label='Theta trans')
plt.plot(range(50), y[:50, 3], label='Phi trans')

# %%
trans = y[:50, 0], y[:50, 1], y[:50, 2], y[:50, 3]
eng1 = E(trans, X[:50, 0], X[:50, 2], 3)
plt.plot(range(len(eng)), -eng, label='opt')
plt.plot(range(len(eng1)), -eng1, label='1')
plt.legend()

# %%
from data_preprocessing import equivalent_vectors
for i, data in enumerate(y):
    y[i] = equivalent_vectors(data, 2)

plt.boxplot(y)
''
