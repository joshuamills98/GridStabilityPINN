import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
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

def numerical_solver(x, t, K, V):
    """
    Solves the convection diffusion equation numerically using a basic forward scheme.

    Arguments:
        x: spatial domain over which to solve
        t: temporal domain over which to solve 
        V: Convective coefficient
        K: Diffusive coefficient
    
    Outputs:
        Solutiontime: Time to solve
        Tmat: Matrix of solved values
    """
    nx = len(x)
    nt = len(t)
    T = np.zeros((nx,nt))
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    start = time.time()
    T[:,0] = np.sin(np.pi*x)

    for ti in range(1,nt-1):
        for xi in range(1,nx-1): # domain
            T[xi,ti] = (1-2*K*dt/dx**2)*T[xi,ti-1]+(K*dt/dx**2+V*dt/(2*dx))*T[xi-1,ti-1]+(K*dt/dx**2-V*dt/(2*dx))*T[xi+1,ti-1]
    
    end = time.time()
    solution_time = end-start

    return T, solution_time


def plot_contour(basemodel, m = 0.1 , d = 0.05,tbounds = (0,20), nt = 100, nP = 100,
                Pbounds = (0.08,0.18),PSlice = [0.12, 0.15, 0.18], cmap = 'plasma', savename ='.\\Plots\\result.img', nlevels = 80, save=False):
    

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
  
    idx = []
    Pslices = []
    for P in PSlice:
        index, val = find_nearest(Pnum, P)
        Pslices.append(val)
        idx.append(index)

    for i, P_cs in enumerate(Pslices):
        plt.subplot(gs[1, i])
        tPmdslice = np.stack([tnum, np.full(tnum.shape, P_cs), np.full(tnum.shape, m), np.full(tnum.shape, d)], axis=-1)
        start = time.time()
        u = basemodel.predict(tPmdslice)
        end = time.time()
        plt.plot(tnum, u, label = 'PINN Model')
        # plt.plot(xnum,T[:,idx[i]], color = 'red', linestyle='dashed', label = 'Numerical Solution')
        plt.title('P = {:.2f}'.format(P_cs))
        plt.xlabel('Time (seconds)')
        plt.ylabel('delta(P,x)')
        # plt.legend()
    PINNtime = end - start
    plt.tight_layout()
    if save:
        plt.savefig(savename, transparent=True)
    plt.show()
    return PINNtime


def ErrorAnalysisPlot(  
                    basemodel, nt=1000,nx=40, n_scenarios=4, 
                    KRange = [0.001,4], VRange = [0.001, 4], 
                    tbounds = [0,1], xbounds = [0,1], levels=30,
                    training_points = ([0.1,1,2,3], [0.2,0.9,1.8,3.2])):
    """

    Plot the error between numeric solution and PINN over a specified range of K and V values
    For the numerical scheme to be stable nt>>nx

    """
    """
    Set up K and V grid over which different solutions will be compared

    """
    x_c = xbounds[1]
    K = np.linspace(KRange[0],KRange[1],n_scenarios)
    V = np.linspace(VRange[0],VRange[1],n_scenarios)

    [Kgrid,Vgrid] = np.meshgrid(K,V)
    KVgrid = np.hstack((Kgrid.flatten()[:,None], Vgrid.flatten()[:,None]))
    Pe = KVgrid[:,1]*x_c/KVgrid[:,0]

    tnum = np.linspace(tbounds[0],tbounds[1],nt) #set up domain
    xnum = np.linspace(xbounds[0],xbounds[1],nx)
    dx = xnum[1]-xnum[0]
    dt = tnum[1]-tnum[0]

    Error = []

    for i in range(n_scenarios**2):
 
        K = KVgrid[i,0]
        V = KVgrid[i,1]
        Pe = V*x_c/K
        assert(dx < 2*K/V)
        assert(dt < dx**2/(2*K))
        """
        Numerical Solution
        """

        T, solution_time = numerical_solver(xnum, tnum, K,V)

        """
        PINN Solution using Numerical Grid
        """

        if Pe<=1:
            Pe = 1
            t_c = x_c**2/K
        else:
            t_c = x_c/V
        
        Peclet_Input = np.repeat(Pe,nt*nx)

        t, x = np.meshgrid(tnum/t_c, xnum/x_c)

        txPe = np.stack([t.flatten(), x.flatten(), Peclet_Input], axis=-1)

        u = basemodel.predict(txPe, batch_size=nt*nx)

        u = u.reshape(T.shape)

        Error.append(np.log(np.abs((np.subtract(u, T))).mean()))

    Error = np.array(Error)

    Error = Error.reshape((n_scenarios,n_scenarios))
    fig = plt.figure(figsize=(10,8))

    cs1 = plt.contourf(Kgrid,Vgrid,Error, levels = levels)
    x1,x2 = np.meshgrid(training_points[0],training_points[1])
    grid = np.hstack((x1.flatten()[:,None], x2.flatten()[:,None]))
    plt.plot(grid[:,0],grid[:,1],'rx',alpha=0.5, label = 'Training Points')

    plt.plot()
    plt.xlabel('Diffusion Coefficient, K (m^2/s)')
    plt.ylabel('Velocity of Fluid, V (m/s)')
    plt.legend()
    cbar = plt.colorbar(cs1)
    cbar.set_label('Error')
    
    return Error










    