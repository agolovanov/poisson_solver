import numpy as _np
import math as _math
import scipy.sparse as _sparse

class PoissonCylindric:
    def __init__(self, zmin : float, zmax : float, dz : float, rmax : float, dr : float, gamma0 : float = 1.0):
        self.gamma0 = gamma0
        self.v0 = _math.sqrt(1 - 1 / gamma0 ** 2)

        self.dr = dr
        self.dz = dz

        if zmin > 0.0:
            raise ValueError(f'zmin should be negative. Value given: {zmin}')
        if zmax < 0.0:
            raise ValueError(f'zmax should be positive. Value given: {zmax}')

        self.Nz = int((zmax - zmin) / dz + 1)
        self.Nr = int(rmax / dr + 1)
        self.N = self.Nz * self.Nr
        self.zmin = float(zmin)
        self.zmax = self.zmin + (self.Nz - 1) * dz
        self.rmax = (self.Nr - 1) * dr
        self.z1d = _np.linspace(self.zmin, self.zmax, self.Nz)
        self.r1d = _np.linspace(0.0, self.rmax, self.Nr)
        self.z, self.r = _np.meshgrid(self.z1d, self.r1d)

        self._A = _sparse.dok_matrix((self.N, self.N))

        Nr = self.Nr
        Nz = self.Nz
        A = self._A
        for i in range(Nr):
            for j in range(Nz):
                if (i == 0):
                    # derivative is equal to zero at r = 0
                    A[Nz * i + j, Nz * i + j] = 1.
                    A[Nz * i + j, Nz * (i + 1) + j] = -1.
                elif (j == 0):
                    # 1/R scaling at zmin
                    A[Nz * i + j, Nz * i + j] = \
                        1. + dz * gamma0 ** 2 * abs(zmin) / (gamma0 ** 2 * zmin ** 2 + i ** 2 * dr ** 2)
                    A[Nz * i + j, Nz * i + j + 1] = -1.
                elif (i == Nr - 1):
                    # 1/R scaling at rmax
                    A[Nz * i + j, Nz * i + j] = \
                        1. + dr * rmax / (rmax ** 2 + gamma0 ** 2 * (j - 0.5 * Nz) ** 2 * dz ** 2)
                    A[Nz * i + j, Nz * (i - 1) + j] = -1.
                elif (j == Nz - 1):
                    # 1/R scaling at zmax
                    A[Nz * i + j, Nz * i + j] = 1. + dz * gamma0 ** 2 * zmax / (gamma0 ** 2 * zmax ** 2 + i ** 2 * dr ** 2)
                    A[Nz * i + j, Nz * i + j - 1] = -1.
                else:
                    A[Nz * i + j, Nz * i + j] = - 2. / dr ** 2 - 2. / dz ** 2 / gamma0 ** 2
                    A[Nz * i + j, Nz * (i + 1) + j] = 1 / dr ** 2 + 0.5 / dr ** 2 / i
                    A[Nz * i + j, Nz * (i - 1) + j] = 1 / dr ** 2 - 0.5 / dr ** 2 / i
                    A[Nz * i + j, Nz * i + j + 1] = 1 / dz ** 2 / gamma0 ** 2
                    A[Nz * i + j, Nz * i + j - 1] = 1 / dz ** 2 / gamma0 ** 2

        self._A = _sparse.csc_matrix(self._A)

    def solve(self, source_func):
        S = source_func(self.z, self.r)
        S[0] = 0
        S = S.reshape((self.N,))
        u = _sparse.linalg.spsolve(self._A, S)
        return self.z, self.r, u.reshape(self.Nr, self.Nz)