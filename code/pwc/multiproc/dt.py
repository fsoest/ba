from multiprocessing import Pool
from create_train_data import create_data
import sys
import numpy as np
import sobol_seq
from functools import partial


if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) == 10:
        N_workers = int(args[1])
        N_dim = int(args[2])
        N_data = int(args[3])
        dt_start = int(args[4])
        dt_stop = int(args[5])
        dt_num = int(args[6])
        rho = args[7]
        run = int(args[8])
        N_sobol = int(args[9])
    else:
        print('Parameters: N_workers, N_dim, N_data, dt_start, dt_stop, dt_num, rho: [haar, 0, eigen], run, N_sobol')
        sys.exit()

    np.random.seed(42)

    # Create random seeds for the workers
    seeds = np.random.randint(0, 2000000000, size=(N_data), dtype=np.int32)


    T = np.linspace(dt_start, dt_stop, dt_num)
    E = np.zeros((dt_num, N_data))
    for i, t in enumerate(T):
        f = partial(create_data, N_dim, t, rho, N_sobol)
        with Pool(processes=N_workers) as pool:
            result = pool.map(f, seeds + run)
            E[i] = np.array(result)[:, 2]


    try:
        np.save('train_data/vardt_N_{0}_rho_{1}/dt_{2}_{3}_E_sobol_{4}_run_{5}'.format(N_dim, rho, int(dt_start), int(dt_stop), N_sobol, run), E)
        np.save('train_data/vardt_N_{0}_rho_{1}/dt_{2}_{3}_times'.format(N_dim, rho, int(dt_start), int(dt_stop)), T)
    except:
        np.save('train_data/vardt_N_{0}_rho_{1}_dt_{2}_{3}_E_sobol_{4}_run_{5}'.format(N_dim, rho, int(dt_start), int(dt_stop), N_sobol, run), E)
        np.save('train_data/vardt_N_{0}_rho_{1}_dt_{2}_{3}_times'.format(N_dim, rho, int(dt_start), int(dt_stop)), T)
