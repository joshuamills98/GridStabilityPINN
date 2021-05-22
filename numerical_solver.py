"""
A numerical solver designed to solve the PDE: m d^2phi/dt^2 +d dphi/dt +BV1V2sn(phi) +P = 0
Here we do the testing necessary to show that the numerical solver is indeed slower than our GridStabilityPINN
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
    Anew = np.array([[fu,fdelta]]).flatten()
    return Anew


def RK45Solver(Prange = [0.08,0.18],m=0.1,B=0.2,d=0.05, nT=100, nP=100):
    dsol = np.zeros((nP,nT))
    ddotsol = np.zeros((nP,nT))
    start = time.time()
    for i, P in enumerate(np.linspace(Prange[0],Prange[1],nP)):
        # print('i = ',i, 'P = ',P)
        sol = RK45(fun = lambda t, y: func(t, y, P, m, d, B), t0 = 0 , y0 =  np.array([[0.1],[0.1]]).flatten(),t_bound = 20, first_step=20/nT, max_step=20/nT)

        t= []
        for j in range(nT):
            dsol[i,j] = sol.y[1]
            ddotsol[i,j] = sol.y[0]
            t.append(sol.t)
            sol.step()
    end = time.time()
    total_time = end - start
    return total_time, t, dsol, ddotsol

if __name__ == '__main__':
    Prange = [0.08,0.15]
    nP=100
    nT=100
    P_list = np.linspace(Prange[0],Prange[1],nP)
    total_time, t, dsol,ddotsol = RK45Solver(Prange = Prange, m=0.4, d=0.1,nT=nT, nP = nP)
    print('Code completed in {:.2f} seconds'.format(total_time))
    cs1 = plt.contourf(t, P_list, dsol, cmap = 'plasma' , levels =100)
    cbar = plt.colorbar(cs1)
    cbar.set_label('Delta(t,P)')
    # print(t)
