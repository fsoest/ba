import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm


def int_operator(theta, phi):
    """
    Returns partial system Hamiltonian, A * sigma_p
    """
    sigma_p = np.matrix([[0, 0], [1, 0]], dtype=np.complex128)
    sigma_m = np.matrix([[0, 1], [0, 0]], dtype=np.complex128)
    return np.sin(theta) * np.exp(1j * phi) * sigma_p / 2 + np.sin(theta) * np.exp(-1j * phi) * sigma_m / 2


def H_s(theta_d, phi_d, a, b):
    """
    Total system Hamiltonian
    """
    drive_op = int_operator(theta_d, phi_d)
    trans_op = int_operator(a, b)
    return drive_op + trans_op


def energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N):
    E = np.zeros(N, dtype=np.complex128)
    rho = rho_0
    for i in range(N - 1):
        U = expm(-1j * dt * H_s(theta_d[i], phi_d[i], theta_t[i], phi_t[i]))
        # Time evolution of state density matrix
        rho = U @ rho @ np.matrix(U).H
        # Change in Hamiltonian due to Transducer, Drive constant
        # delta_trans = np.sin(theta_t[i + 1]) * np.exp(1j * phi_t[i + 1]) - np.sin(theta_t[i]) * np.exp(1j * phi_t[i])
        dH = int_operator(theta_t[i+1], phi_t[i+1]) - int_operator(theta_t[i], phi_t[i])
        E[i] = np.trace(rho @ dH)

    return np.cumsum(E.real)[-1]


def rho_path(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N, res_steps):
    E = np.zeros(N, dtype=np.complex128)
    rhos = np.zeros(((N - 1) * res_steps + 1 , 2, 2), dtype=np.complex128)
    rho_step = np.zeros((N, 2, 2), dtype = np.complex128)
    rho = rho_0
    rhos[0] = rho_0
    rho_step[0] = rho_0
    for i in range(N - 1):
        for k in range(res_steps):
            U = expm(-1j * dt * H_s(theta_d[i], phi_d[i], theta_t[i], phi_t[i]) / res_steps)
            # Time evolution of state density matrix
            rho = U @ rho @ np.matrix(U).H
            rhos[res_steps * i + k + 1] = rho
            # Change in Hamiltonian due to Transducer, Drive constant
        dH = int_operator(theta_t[i+1], phi_t[i+1]) - int_operator(theta_t[i], phi_t[i])
        E[i] = np.trace(rho @ dH)
        rho_step[i + 1] = rho

    return rhos, rho_step, E


def wrapper(transducer, theta_d, phi_d, dt, rho_0, N):
    theta_t = transducer[:int(len(transducer)/2)]
    phi_t = transducer[int(len(transducer)/2):]
    return energy(theta_d, phi_d, theta_t, phi_t, dt, rho_0, N)


def model_evaluation(model, X_test, y_test, dt, rho_0, N):
    """
    Evaluates a given model, gives E_pred, E_actual as output
    """
    E_pred = np.zeros(len(X_test))
    E_actual = np.zeros(len(X_test))
    y_pred = model.predict(X_test)
    # rand = np.random.uniform(-1, 1, (2520, 3 * N))
    # E_rand = np.zeros(len(X_test))
    for i, x in enumerate(X_test):
        theta_d = np.arctan2(x[:, 0], x[:, 2])
        phi_d = np.arctan2(x[:, 1], x[:, 3])
        E_pred[i] = wrapper(y_pred[i], theta_d, phi_d, dt, rho_0, N)
        E_actual[i] = wrapper(y_test[i], theta_d, phi_d, dt, rho_0, N)
        # E_rand[i] = wrapper(rand[i], theta_d, phi_d, dt, rho_0, N)
    return E_pred, E_actual


def model_evaluation_embedded(model, X_test, y_test, dt, rho_0, N):
    """
    Evaluates a given model, gives E_pred, E_actual as output
    """
    E_pred = np.zeros(len(X_test))
    E_actual = np.zeros(len(X_test))
    y_pred = model.predict(X_test)
    pred_reshape = np.zeros((len(X_test), N, 4))
    actual_reshape = np.zeros((len(X_test), N, 4))
    y_trans_test = np.zeros((len(X_test), 2 * N))
    y_trans_pred = np.zeros((len(X_test), 2 * N))
    for i, x in enumerate(X_test):
        theta_d = np.arctan2(x[:, 0], x[:, 2])
        phi_d = np.arctan2(x[:, 1], x[:, 3])
        pred_reshape[i] = np.reshape(y_pred[i], (N, 4))
        actual_reshape[i] = np.reshape(y_test[i], (N, 4))
        y_trans_test[i, :N] = np.arctan2(actual_reshape[i, :, 0], actual_reshape[i, :, 2])
        y_trans_test[i, N:] = np.arctan2(actual_reshape[i, :, 1], actual_reshape[i, :, 3])
        y_trans_pred[i, :N] = np.arctan2(pred_reshape[i, :, 0], pred_reshape[i, :, 2])
        y_trans_pred[i, N:] = np.arctan2(pred_reshape[i, :, 1], pred_reshape[i, :, 3])
        if rho_0.shape == (2, 2):
            E_pred[i] = wrapper(y_trans_pred[i], theta_d, phi_d, dt, rho_0, N)
            E_actual[i] = wrapper(y_trans_test[i], theta_d, phi_d, dt, rho_0, N)
        else:
            E_pred[i] = wrapper(y_trans_pred[i], theta_d, phi_d, dt, rho_0[i], N)
            E_actual[i] = wrapper(y_trans_test[i], theta_d, phi_d, dt, rho_0[i], N)
    return E_pred, E_actual


def model_evaluation_convolution(model, X_test, y_test, dt, rho_0, N):
    """
    Evaluates a given model, gives E_pred, E_actual as output
    """
    E_pred = np.zeros(len(X_test))
    E_actual = np.zeros(len(X_test))
    y_pred = model.predict(X_test)
    pred_reshape = np.zeros((len(X_test), N, 4))
    actual_reshape = np.zeros((len(X_test), N, 4))
    y_trans_test = np.zeros((len(X_test), 2 * N))
    y_trans_pred = np.zeros((len(X_test), 2 * N))
    for i, x in enumerate(X_test[:, :, :, 0]):
        theta_d = np.arctan2(x[:, 0], x[:, 2])
        phi_d = np.arctan2(x[:, 1], x[:, 3])
        pred_reshape[i] = np.reshape(y_pred[i], (N, 4))
        actual_reshape[i] = np.reshape(y_test[i], (N, 4))
        y_trans_test[i, :N] = np.arctan2(actual_reshape[i, :, 0], actual_reshape[i, :, 2])
        y_trans_test[i, N:] = np.arctan2(actual_reshape[i, :, 1], actual_reshape[i, :, 3])
        y_trans_pred[i, :N] = np.arctan2(pred_reshape[i, :, 0], pred_reshape[i, :, 2])
        y_trans_pred[i, N:] = np.arctan2(pred_reshape[i, :, 1], pred_reshape[i, :, 3])
        E_pred[i] = wrapper(y_trans_pred[i], theta_d, phi_d, dt, rho_0, N)
        E_actual[i] = wrapper(y_trans_test[i], theta_d, phi_d, dt, rho_0, N)
    return E_pred, E_actual


def state_to_angles(kets):
    """
    Input: kets [N, 2]
    """
    theta = 2 * np.arctan2(np.abs(kets[:, 1]),np.abs(kets[:, 0]))
    phi = (np.angle(kets[:, 1]) - np.angle(kets[:, 0])) % (2 * np.pi)
    return theta[:, 0], phi[:, 0]


def get_eigen_rho(theta, phi):
    """
    """
    alp = np.zeros(len(theta), dtype=np.complex128)
    alp = np.sin(theta) * np.exp(1j * phi) / 2
    zero_state = np.zeros((len(theta), 2), dtype=np.complex128)
    zero_state[np.abs(alp) != 0, 0] = np.conj(alp[np.abs(alp) != 0])/np.abs(alp[np.abs(alp) != 0])/np.sqrt(2)
    zero_state[np.abs(alp) != 0, 1] = 1/np.sqrt(2)
    zero_state[np.abs(alp) == 0, 0] = 1/np.sqrt(2)
    rho_0 = np.zeros((len(theta), 2, 2), dtype=np.complex128)
    rho_0 = zero_state[:, :, np.newaxis] * np.conj(zero_state[:, np.newaxis, :])

    return rho_0


def rho_to_embedding(rho):
    u = np.real(rho[1, 0] + rho[0, 1])
    v = np.real(1j * (rho[0, 1] - rho[1, 0]))
    w = np.real(rho[0, 0] - rho[1, 1])
    theta = np.arccos(w)
    phi = np.arctan2(np.real(v), u)
    return np.array([np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)])


def rho_to_angles(rho):
    u = np.real(rho[1, 0] + rho[0, 1])
    v = np.real(1j * (rho[0, 1] - rho[1, 0]))
    w = np.real(rho[0, 0] - rho[1, 1])
    theta = np.arccos(w)
    phi = np.arctan2(np.real(v), u)
    return theta, phi    
