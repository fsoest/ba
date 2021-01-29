from multiprocessing import Pool
import sys
import numpy as np
from functools import partial
from lower_bound import lower_bound
from qutip import rand_ket_haar as rkh
from multiproc.pwc_helpers import state_to_angles


def calc_lower(N_dim, dt, seed):
    # Seed
    np.random.seed(seed)
    # Create uniform drives
    kets = np.zeros((N_dim, 2, 1), dtype=np.complex128)
    for j in range(N_dim):
        kets[j] = rkh(2).full()
    # Angles from state vectors
    theta_d, phi_d = state_to_angles(kets)
    e = lower_bound(np.concatenate((theta_d, phi_d)), N_dim, dt)[0]
    e = np.cumsum(e)[-1]
    return e

if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) == 8:
        N_workers = int(args[1])
        N_dim = int(args[2])
        N_data = int(args[3])
        dt_start = int(args[4])
        dt_stop = int(args[5])
        dt_num = int(args[6])
        run = int(args[7])
    else:
        print('Parameters: N_workers, N_dim, N_data, dt_start, dt_stop, dt_num, run')
        sys.exit()

    np.random.seed(42)

    # Create random seeds for the workers
    seeds = np.random.randint(0, 2000000000, size=(N_data), dtype=np.int32)


    T = np.linspace(dt_start, dt_stop, dt_num)
    E = np.zeros((dt_num, N_data))
    for i, t in enumerate(T):
        f = partial(calc_lower, N_dim, t)
        with Pool(processes=N_workers) as pool:
            result = pool.map(f, seeds + run)
            E[i] = np.array(result)


        np.save('lower_dt/vardt_N_{0}_rho_{1}_dt_{2}_{3}_run_{4}_E'.format(N_dim, 'eigen', int(dt_start), int(dt_stop), run), E)
        np.save('lower_dt/vardt_N_{0}_rho_{1}_dt_{2}_{3}_times'.format(N_dim, 'eigen', int(dt_start), int(dt_stop)), T)
