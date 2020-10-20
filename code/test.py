import numpy as np
from helpers import ket_to_real

ket = np.array([1 + 0.5j, 0.5 - 2j])

np.array([4, *ket_to_real(ket)])
