H = qt.qeye(2)
z = H - H
def matr_el(H, i, j, k, l, m, n):
    bra = qt.tensor(qt.fock(2, i), qt.fock(2, j), qt.fock(2, k)).dag()
    ket = qt.tensor(qt.fock(2, l), qt.fock(2, m), qt.fock(2, n))
    return bra * H * ket

def scal_dens(rho, a, b):
    return qt.fock(2, a).dag() * rho * qt.fock(2, b)

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        z += matr_el(H_dst, i, j, k, l, m, n) * qt.fock(2, j) * qt.fock(2, m).dag() * scal_dens(rho(psi_d(1)), n, i) * scal_dens(rho(psi_d(np.pi)), l, k)
