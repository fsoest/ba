import numpy as np
import qutip as qt
from data_preprocessing import equivalent_vectors


b = qt.Bloch()

X = np.load('train_data/X_N_2_50000_dt_3.npy')
X_eq = np.zeros(X.shape)
for i, x in enumerate(X):
    X_eq[i, :] = equivalent_vectors(x, 2)

X_cart = np.zeros((X.shape[0], 3))
X_cart[:, 0] = np.sin(X_eq[:, 0]) * np.cos(X_eq[:, 3])
X_cart[:, 1] = np.sin(X_eq[:, 0]) * np.sin(X_eq[:, 3])
X_cart[:, 2] = np.cos(X_eq[:, 0])


# %%
b.add_points(X_cart.T)
b.clear()
b.show()
