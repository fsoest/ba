from multiprocessing import Pool
from create_train_data import create_data
import sys
import numpy as np
import sobol_seq
from functools import partial


if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) == 8:
        N_workers = int(args[1])
        N_dim = int(args[2])
        N_data = int(args[3])
        dt = float(args[4])
        rho = args[5]
        run = int(args[6])
        N_sobol = int(args[7])
    else:
        print('Parameters: N_workers, N_dim, N_data, dt, rho: [haar, 0, eigen], run, N_sobol')
        sys.exit()

    np.random.seed(42)

    # Create random seeds for the workers
    seeds = np.random.randint(0, 2000000000, size=(N_data), dtype=np.int32)

    f = partial(create_data, N_dim, dt, rho, N_sobol)

    with Pool(processes=N_workers) as pool:
        result = pool.map(f, seeds + run)

        try:
            np.save('train_data/N_{0}/dt_{1}_{2}_sobol_{3}_run_{4}'.format(N_dim, int(dt), rho, N_sobol, run), result)
        except:
            # np.save('/scratch/ws/1/s2205896-qwork/N_{0}_dt_{1}_{2}_sobol_{3}_run_{4}'.format(N_dim, int(dt), rho, N_sobol, run), result)
            np.save('train_data/N_{0}_dt_{1}_{2}_sobol_{3}_run_{4}'.format(N_dim, int(dt), rho, N_sobol, run), result)
