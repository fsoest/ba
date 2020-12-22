import numpy as np
import sys


if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) == 5:
        N_dim = int(args[1])
        dt = int(args[2])
        rho = args[3]
        run_max = int(args[4])
    else:
        print('Parameters: N_dim, dt, rho: [haar, 0, eigen], run_max')
        sys.exit()

    E = np.load('train_data/N_{0}/dt_{1}/{2}/E_run_0.npy'.format(N_dim, dt, rho))
    X = np.load('train_data/N_{0}/dt_{1}/{2}/X_run_0.npy'.format(N_dim, dt, rho))
    y = np.load('train_data/N_{0}/dt_{1}/{2}/y_run_0.npy'.format(N_dim, dt, rho))


    if rho == 'haar':
        psis = np.load('train_data/N_{0}/dt_{1}/{2}/psis_run_0.npy'.format(N_dim, dt, rho))
        for i in range(run_max - 1):
            e_i = np.load('train_data/N_{0}/dt_{1}/{2}/E_run_{3}.npy'.format(N_dim, dt, rho, i+ 1))
            E = np.concatenate([E, e_i])

            X_i = np.load('train_data/N_{0}/dt_{1}/{2}/X_run_{3}.npy'.format(N_dim, dt, rho, i + 1))
            X = np.concatenate([X, X_i])

            y_i = np.load('train_data/N_{0}/dt_{1}/{2}/y_run_{3}.npy'.format(N_dim, dt, rho, i + 1))
            y = np.concatenate([y, y_i])

            psis_i = np.load('train_data/N_{0}/dt_{1}/{2}/psis_run_{3}.npy'.format(N_dim, dt, rho, i + 1))
            psis = np.concatenate([psis, psis_i])

        np.save('train_data/tot/N_{0}_dt_{1}_{2}_psis_tot.npy'.format(N_dim, dt, rho), psis)

    else:
        for i in range(run_max - 1):
            e_i = np.load('train_data/N_{0}/dt_{1}/{2}/E_run_{3}.npy'.format(N_dim, dt, rho, i + 1))
            E = np.concatenate([E, e_i])

            X_i = np.load('train_data/N_{0}/dt_{1}/{2}/X_run_{3}.npy'.format(N_dim, dt, rho, i + 1))
            X = np.concatenate([X, X_i])

            y_i = np.load('train_data/N_{0}/dt_{1}/{2}/y_run_{3}.npy'.format(N_dim, dt, rho, i + 1))
            y = np.concatenate([y, y_i])


    np.save('train_data/tot/N_{0}_dt_{1}_{2}_E_tot.npy'.format(N_dim, dt, rho), E)
    np.save('train_data/tot/N_{0}_dt_{1}_{2}_X_tot.npy'.format(N_dim, dt, rho), X)
    np.save('train_data/tot/N_{0}_dt_{1}_{2}_y_tot.npy'.format(N_dim, dt, rho), y)
