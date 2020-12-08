from multiproc.pwc_helpers import wrapper, model_evaluation_embedded
import numpy as np
from N2_analyt import E
from scipy.optimize import minimize
import tensorflow as tf
from sklearn.model_selection import train_test_split
from multiproc.data_preprocessing import angle_embedding, rev_angle_embedding
from models import CustomSchedule
import matplotlib.pyplot as plt
# %%
N = 2
E = np.load('multiproc/train_data/N_2/dt_5/eigen/E_run_0.npy')
y = np.load('train_data/n2dt5/N_2_dt_5_eigen_y_tot.npy')

X_embed = angle_embedding(X, N)
y_embed = angle_embedding(y, N, reshape=True)

X_train, X_test, y_train, y_test = train_test_split(X_embed, y_embed, test_size=0.18, random_state=42)

# %%
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(N, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4 * N))

learning_rate = CustomSchedule(4, 500)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss='mse')
# %%
model.fit(X_train, y_train, epochs=2000, verbose=1, validation_split=0.1)
# %%
N = 2
dt = 5

rho_0.shape

# %%
X_now = X_test
y_now = y_test

E_pred, E_actual = model_evaluation_embedded(model, X_now, y_now, dt, rho_0, N)

np.mean(E_pred[E_actual!=0]/E_actual[E_actual!=0])
# %%
plt.scatter(range(len(E_pred)), -E_pred, label='Predicted', marker='.')
plt.scatter(range(len(E_actual)), -E_actual, label='Test', marker='.')
plt.legend()
plt.show()
# %%
import qutip as qt
from plot_helpers import tp_to_uvw
# %%
X_low = rev_angle_embedding(X_test[E_pred>0], 2)
y_low = rev_angle_embedding(y_test[E_pred>0], 2, reshape=True)

theta_low = X_low[:, 0]
phi_low = X_low[:, 2]

drives = tp_to_uvw(theta_low, phi_low)
trans = tp_to_uvw(y_low[:3, 0], y_low[:3, 2])
trans_prime = tp_to_uvw(y_low[:3, 1], y_low[:3, 3])

# %%
b = qt.Bloch()
b.add_points(drives)
# b.add_points(trans)
# b.add_points(trans_prime)
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


# %%
y_rev = rev_angle_embedding(model.predict(X_test), 2, reshape=True)

plt.boxplot(y)
np.mean(y[:, 1])
# %%
loel = np.load('train_data/X_N')
loel.shape

eng = np.load('train_data/N2dt5/N_2_dt_5_0_E_tot.npy')
rhos[14]
E[14]
X[14]
E = np.load('multiproc/train_data/N_2/dt_5/eigen/E_run_0.npy')
X = np.load('multiproc/train_data/N_2/dt_5/eigen/X_run_0.npy')
y = np.load('multiproc/train_data/N_2/dt_5/eigen/y_run_0.npy')
rhos = np.load('multiproc/train_data/N_2/dt_5/eigen/rho_run_0.npy')


plt.plot(range(len(E)), E)


data = np.load('multiproc/train_data/N_2/dt_5_eigen_sobol_10_run_0.npy', allow_pickle=True)
data[0, 3]
data[0, 2]

plt.scatter(range(len(data)), data[:, 2])
plt.boxplot(data[:, 2])

    data[1]
wrapper(data[1, 1], data[1, 0][:2], data[1, 0][2:], 5, data[1, 3], 2)



wrapper(np.array([X[14, 0], np.pi - X[14, 0], X[14, 2], X[14, 2]+np.pi]), X[14, :2], X[14, 2:], 5, rhos[14], 2)

X[14, 0]

# %%
%matplotlib inline

def w(x):
    return wrapper(np.array([thet_d[0], np.pi/2, np.pi, x]), X[13, :2], np.array([np.pi/6, 6]), 5, rhos[13], 2)
x = np.linspace(0, 2*np.pi, 100)
Y = x

plt.plot(x, [w(c) for c in x])
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, Y, z, cmap=cm.coolwarm)
fig.colorbar(surf)
np.unravel_index(z.argmin(), z.shape)
#%%
wrapper(data[0, 1], data[0, 0][:2], data[0, 0][2:], 5, data[0, 3], 2)

trans_now = data[0, 1]
thet_d = data[0, 0][:2]
phi_d = data[0, 0][2:]
wrapper(trans_now, thet_d, phi_d, 5, data[0, 3], 2)

zero_rho = np.matrix([1, 0], dtype=np.complex128).H @ np.matrix([1, 0], dtype=np.complex128)
zero_rho
# %%

def factor(theta, phi):
    return np.sin(theta) * np.exp(1j*phi)/2

alpha = factor(thet_d[0], phi_d[0]) + factor(trans_now[0], trans_now[2])


def power(alpha, dtau, dt, rho):
    part0 = (np.abs(rho[0, 0])**2 - np.abs(rho[1, 1])**2) * np.sin(2 * np.abs(alpha) * dt) * np.imag(dtau * np.conj(alpha)) / np.abs(alpha)
    part1 = np.real(dtau * rho[0, 1]) * np.cos(np.abs(alpha) * dt)**2
    part2 = np.real(dtau * rho[1, 0] * np.conj(alpha)/alpha)* np.sin(np.abs(alpha) * dt)**2
    return part0 - 2 * (part1 + part2)

dtau = factor(trans_now[1], trans_now[3]) - factor(trans_now[0], trans_now[2])

power(alpha, dtau, 5, data[0, 3])

# %%
def pplot(t):
    return power(alpha, dtau, t, rhos[14])
t = np.linspace(0, 50, 100)
plt.plot(t, [pplot(c) for c in t])

# %%
d = y - X
d
for i in range(len(d[E < -0.999])):
    plt.scatter(range(4), d[E < -0.999][i,])


plt.boxplot(y)
