import numpy as np

X = np.load('train_data/X_N_10_tot_dt_1.npy')
y = np.load('train_data/y_N_10_tot_dt_1.npy')

X_1 = np.load('train_data/X_N_10_1990_dt_1.npy')
X_2 = np.load('train_data/X_N_10_1998_dt_1.npy')
X_3 = np.load('train_data/X_N_10_1999_dt_1.npy')
X_4 = np.load('train_data/X_N_10_2000_dt_1.npy')
X_5 = np.load('train_data/X_N_10_2001_dt_1.npy')

y_1 = np.load('train_data/y_N_10_1990_dt_1.npy')
y_2 = np.load('train_data/y_N_10_1998_dt_1.npy')
y_3 = np.load('train_data/y_N_10_1999_dt_1.npy')
y_4 = np.load('train_data/y_N_10_2000_dt_1.npy')
y_5 = np.load('train_data/y_N_10_2001_dt_1.npy')

X.shape

X_tot = np.concatenate((X, X_1, X_2, X_3, X_4, X_5))
X_tot.shape

y_tot = np.concatenate((y, y_1, y_2, y_3, y_4, y_5))
y_tot.shape

np.save('train_data/X_N_10_tot_dt_1.npy', X_tot)
np.save('train_data/y_N_10_tot_dt_1.npy', y_tot)
