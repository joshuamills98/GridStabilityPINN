"""
A numerical solver designed to solve the PDE:
m d^2phi/dt^2 +d dphi/dt +BV1V2sn(phi) +P = 0
Here we do the testing necessary to show that the
numerical solver is indeed slower than our GridStabilityPINN
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import RK45


def func(t, A, P, m, d, B):
    """
    t represents time
    A is a column vector with u and delta
    """

    u = A[0]
    delta = A[1]
    fu = (P-B*np.sin(delta)-d*u)/m
    fdelta = u
    Anew = np.array([[fu, fdelta]]).flatten()
    return Anew


def RK45Solver(P_range=[0.08, 0.18], m=0.1, B=0.2, d=0.05, N_t=100, N_P=100):

    dsol = np.zeros((N_P, N_t))
    ddotsol = np.zeros((N_P, N_t))
    start = time.time()
    for i, P in enumerate(np.linspace(P_range[0], P_range[1], N_P)):
        sol = RK45(
            fun=lambda t, y: func(t, y, P, m, d, B), t0=0,
            y0=np.array([[0.1], [0.1]]).flatten(),
            t_bound=20, first_step=20/N_t, max_step=20/N_t)

        t = []
        for j in range(N_t):
            dsol[i, j] = sol.y[1]
            ddotsol[i, j] = sol.y[0]
            t.append(sol.t)
            sol.step()
    end = time.time()
    total_time = end - start
    return total_time, t, dsol, ddotsol


if __name__ == '__main__':
    P_range = [0.08, 0.15]
    N_P = 100
    N_t = 100
    P_list = np.linspace(P_range[0], P_range[1], N_P)
    total_time, t, dsol, ddotsol = RK45Solver(P_range=P_range, m=0.4,
                                              d=0.1, N_t=N_t, N_P=N_P)
    print('Code completed in {:.2f} seconds'.format(total_time))
    cs1 = plt.contourf(t, P_list, dsol, cmap='plasma', levels=100)
    cbar = plt.colorbar(cs1)
    cbar.set_label('Delta(t,P)')
