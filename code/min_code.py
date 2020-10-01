def func_to_min(x, t_arr, drive_t, drive_p, H_dst, rhs, t_span):
    trans_t = x[:N]
    trans_p = x[N:]
    t_arr, drive_t, drive_p, H_dst, rhs, t_span = min_args
    solver_args = (t_arr, phi, phi, trans_t, trans_p, H_dst)
    return solve_ivp(rhs, t_span, np.array([0, 1/np.sqrt(2) + 0j, 1/np.sqrt(2) + 0j]), args=solver_args).y[0][-1].real

drive_t = theta
drive_p = phi
min_args = (t_arr, drive_t, drive_p, H_dst, rhs, t_span)
x_0 = np.zeros(2 * N)
x_0[:N] = theta
x_0[N:] = phi
minimize(func_to_min, x_0, args=min_args)
