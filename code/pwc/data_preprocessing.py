import numpy as np


def angle_embedding(X, N, reshape=False):
    """
    If reshape: False: X data, Shape is input shape: (N, 4)
    If reshape True: y data, output shape (4 * N,)
    """
    input = np.zeros((X.shape[0], N, 4))
    shaped = np.zeros((X.shape[0], N * 4))
    for i, x in enumerate(X):
        a = np.reshape(x, (2, N)).T
        input[i,:, :2] = np.sin(a)
        input[i,:, 2:] = np.cos(a)
        shaped[i] = np.reshape(input[i], (N * 4))
    if reshape == True:
        return shaped
    return input


def rev_angle_embedding(X, N, reshape=False):
    """
    """
    output = np.zeros((X.shape[0], 2 * N))
    if reshape == False:
        for i, x in enumerate(X):
            output[i, :N] = np.arctan2(x[:, 0], x[:, 2])
            output[i, N:] = np.arctan2(x[:, 1], x[:, 3])
        return output
    else:
        for i, x in enumerate(X):
            shape = np.reshape(x, (N, 4))
            output[i, :N] = np.arctan2(shape[:, 0], shape[:, 2])
            output[i, N:] = np.arctan2(shape[:, 1], shape[:, 3])
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
