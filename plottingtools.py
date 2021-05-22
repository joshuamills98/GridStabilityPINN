import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
from numerical_solver import func, RK45Solver
from matplotlib.gridspec import GridSpec
plt.style.use('bmh')
plt.rcParams.update({'font.size': 14})

def find_nearest(array, value):
    """
    Helper function for plotting
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def plot_contour(basemodel, m = 0.1 , d = 0.05,tbounds = (0,20), nt = 100, 
                nP = 100, Pbounds = (0.08,0.18),PSlice = [0.12, 0.15, 0.18], 
                cmap = 'plasma', savename ='.\\Plots\\result.img', 
                nlevels = 80, save=False):
    

    assert m>=0.1 and m<=0.4 and d>=0.05 and d<= 0.15, 'Ensure m and d are within bounds of training (0.1<=m<=0.4, 0.05<d<=0.15)'

    tnum = np.linspace(tbounds[0], tbounds[1], nt)
    Pnum = np.linspace(Pbounds[0], Pbounds[1], nP)  
    t, P = np.meshgrid(tnum, Pnum)
    mvals, dvals = np.repeat(m, nt*nP), np.repeat(d, nt*nP)

    tPmd = np.stack([t.flatten(), P.flatten(), mvals, dvals], axis=-1)

    u = basemodel.predict(tPmd).reshape((nt,nP))
    
    fig = plt.figure(figsize=(12,8))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    cs1 = plt.contourf(t,P,u,cmap = cmap,levels =nlevels)
    plt.xlabel('t')
    plt.ylabel('Power')
    cbar = plt.colorbar(cs1)
    cbar.set_label('Delta(t,P)')
    _,_, delta_numerical_solution, _ = RK45Solver(m = m, d=d, nT= nt, nP=nP)

    idx = []
    Pslices = []
    for P in PSlice:
        index, val = find_nearest(Pnum, P)
        Pslices.append(val)
        idx.append(index)

    for i, P_cs in enumerate(Pslices):
        plt.subplot(gs[1, i])
        tPmdslice = np.stack([tnum, np.full(tnum.shape, P_cs), np.full(tnum.shape, m), np.full(tnum.shape, d)], axis=-1)
        u = basemodel.predict(tPmdslice)
        plt.plot(tnum, u, label = 'PINN Model')
        plt.plot(tnum,delta_numerical_solution[idx[i],:], color = 'red', linestyle='dashed', label = 'Numerical Solution')
        plt.title('P = {:.2f}'.format(P_cs))
        plt.xlabel('Time (seconds)')
        plt.ylabel('delta(P,x)')
        # plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(savename, transparent=True)
    plt.show()


# def ErrorAnalysisPlot(basemodel):
#     """

#     Plot the error between numeric solution and PINN over a specified range of K and V values
#     For the numerical scheme to be stable nt>>nx

#     """
#     """
#     Set up K and V grid over which different solutions will be compared

#     """
#     x_c = xbounds[1]
#     K = np.linspace(KRange[0],KRange[1],n_scenarios)
#     V = np.linspace(VRange[0],VRange[1],n_scenarios)

#     [Kgrid,Vgrid] = np.meshgrid(K,V)
#     KVgrid = np.hstack((Kgrid.flatten()[:,None], Vgrid.flatten()[:,None]))
#     Pe = KVgrid[:,1]*x_c/KVgrid[:,0]

#     tnum = np.linspace(tbounds[0],tbounds[1],nt) #set up domain
#     xnum = np.linspace(xbounds[0],xbounds[1],nx)
#     dx = xnum[1]-xnum[0]
#     dt = tnum[1]-tnum[0]

#     Error = []

#     for i in range(n_scenarios**2):
 
        
#     cs1 = plt.contourf(Kgrid,Vgrid,Error, levels = levels)
#     x1,x2 = np.meshgrid(training_points[0],training_points[1])
#     grid = np.hstack((x1.flatten()[:,None], x2.flatten()[:,None]))
#     plt.plot(grid[:,0],grid[:,1],'rx',alpha=0.5, label = 'Training Points')

#     plt.plot()
#     plt.xlabel('Diffusion Coefficient, K (m^2/s)')
#     plt.ylabel('Velocity of Fluid, V (m/s)')
#     plt.legend()
#     cbar = plt.colorbar(cs1)
#     cbar.set_label('Error')
    
#     return Error


def ErrorPlot(PINNModel, m = 0.1, d = 0.05, Prange = [0.08,0.18], nP=100, nt=100, levels =80):
    Pnum = np.linspace(Prange[0],Prange[1],nP)
    tnum= np.linspace(0,20,nt)    
    mvals, dvals = np.repeat(m, nt*nP), np.repeat(d, nt*nP)
    t, P = np.meshgrid(tnum, Pnum)

    tPmd = np.stack([t.flatten(), P.flatten(), mvals, dvals], axis=-1)
    x = [tPmd, tPmd, tPmd]
    f_out, _, _ = PINNModel(x)
    f_out = f_out.numpy().reshape((nP,nt))
    cs1 = plt.contourf(t, P ,np.abs(f_out), levels = levels)
    cbar = plt.colorbar(cs1)
    cbar.set_label('Error')









    