import numpy as np


def angle_embedding(X, N, reshape=False):
    """
    Creates embedding from (theta, phi) data
    If reshape: False: X data, Shape is input shape: (N, 4)
    If reshape True: y data, output shape (4 * N,)
    """
    input = np.zeros((X.shape[0], N, 5))
    shaped = np.zeros((X.shape[0], N * 4))
    for i, x in enumerate(X):
        a = np.reshape(x, (2, N)).T
        input[i,:, :2] = np.sin(a)
        input[i,:, 2:-1] = np.cos(a)
        shaped[i] = np.reshape(input[i, :, :-1], (N * 4))
        input[i, -1, -1] = 1
    if reshape == True:
        return shaped
    return input


def mult_embedding(X, N):
    """
    Creates multiplicative embedding
    """
    input = np.zeros((X.shape[0], N, 7))
    for i, x in enumerate(X):
        thetas = x[:N]
        phis = x[N:]
        input[i, :, 0] = np.sin(thetas) * np.sin(phis)
        input[i, :, 1] = np.sin(thetas) * np.cos(phis)
        input[i, :, 2] = np.sin(thetas)
        input[i, :, 3] = np.cos(thetas)
        input[i, :, 4] = np.sin(phis)
        input[i, :, 5] = np.cos(phis)
        input[i, -1, -1] = 1
    return input


def out_embedding(X, N):
    output = np.zeros((X.shape[0], N, 2))
    for i, x in enumerate(X):
        output[i, :, 0] = x[:N] / np.pi
        output[i, :, 1] = x[N:] / (2 * np.pi)
    return output

def rev_out_embedding(X, N):
    output = np.zeros((X.shape[0], 2 * N))
    for i, x in enumerate(X):
        output[i, :N] = x[:, 0] * np.pi
        output[i, N:] = x[:, 1] * 2 * np.pi
    return output


def rev_mult_embedding(X, N):
    output = np.zeros((X.shape[0], 2 * N))
    for i, x in enumerate(X):
        output[i, :N] = np.arctan2(x[:, 2], x[:, 3])
        output[i, N:] = np.arctan2(x[:, 0], x[:, 1]) % (2*np.pi)
    return output


def rev_angle_embedding(X, N, reshape=False):
    """
    Creates (theta, phi) data from embedded model data
    If reshape == False, input data is X data (N, 4)
    If reshape == True, input data is y data (4 * N)
    """
    output = np.zeros((X.shape[0], 2 * N))
    if reshape == False:
        for i, x in enumerate(X):
            output[i, :N] = np.arctan2(x[:, 0], x[:, 2])
            output[i, N:] = np.arctan2(x[:, 1], x[:, 3]) % (2 * np.pi)
        return output
    else:
        for i, x in enumerate(X):
            shape = np.reshape(x, (N, 4))
            output[i, :N] = np.arctan2(shape[:, 0], shape[:, 2])
            output[i, N:] = np.arctan2(shape[:, 1], shape[:, 3]) % (2 * np.pi)
        return output


def equivalent_vectors(y, N):
    """
    Prevents same vectors on Bloch sphere being represented by different angles, as output of optimiser theta € [0, 2pi]
    Takes output of minimiser and dimension as input
    """
    scale = y % (2 * np.pi)
    selection = scale[:N] > np.pi
    scale[:N][selection] -= 2 * np.pi
    scale[:N][selection] *= -1
    scale[N:][selection] += np.pi
    return scale


def cartesian_normalise(X, N):
    """

    """
    X = np.reshape(X, (3, N))
    X /= np.linalg.norm(X, axis=0)
    return np.reshape(X, (3 * N,))


def import_datasets(root_dir, N, dt, rho, sobol, runs):
    """
    Imports datasets consisting of multiple files
    """
    data = np.load('{0}/N_{1}/dt_{2}_{3}_sobol_{4}_run_{5}.npy'.format(root_dir, N, dt, rho, sobol, runs[0]), allow_pickle=True)
    for run in runs[1:]:
        d = np.load('{0}/N_{1}/dt_{2}_{3}_sobol_{4}_run_{5}.npy'.format(root_dir, N, dt, rho, sobol, run), allow_pickle=True)
        data = np.concatenate((data, d))
    return data


def angles_to_cart(angles):
    """
    Converts angles [[N*theta, N*pi]] to [x, y, z]
    """
    N = angles.shape[1] // 2
    cart = np.zeros((len(angles), 3 * N))
    for i, angle in enumerate(angles):
        x = np.sin(angle[:N]) * np.cos(angle[N:])
        y = np.sin(angle[:N]) * np.sin(angle[N:])
        z = np.cos(angle[:N])
        cart[i] = np.concatenate((x, y, z))
    return cart
