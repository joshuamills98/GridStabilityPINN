import numpy as np
import matplotlib.pyplot as plt
from numerical_solver import RK45Solver
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


def plot_contour(base_model, m=0.4,
                 d=0.15, t_bounds=(0, 20),
                 N_t=100, N_P=100,
                 P_bounds=(0.08, 0.18), P_slice=[0.09, 0.15, 0.17],
                 cmap='magma', save_name='.\\Plots\\result.img',
                 N_levels=150, save=False):

    assert m >= 0.1 and m <= 0.4 and d >= 0.05 and d <= 0.15, (
                                                "Ensure m and d"
                                                "are within"
                                                "bounds of training"
                                                "(0.1<=m<=0.4, 0.05<d<=0.15)")

    tnum = np.linspace(t_bounds[0], t_bounds[1], N_t)
    Pnum = np.linspace(P_bounds[0], P_bounds[1], N_P)
    t, P = np.meshgrid(tnum, Pnum)
    mvals, dvals = np.repeat(m, N_t*N_P), np.repeat(d, N_t*N_P)

    tPmd = np.stack([t.flatten(), P.flatten(), mvals, dvals], axis=-1)
    u = base_model.predict(tPmd).reshape((N_t, N_P))
    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    cs1 = plt.contourf(tnum, Pnum, u, cmap=cmap, levels=N_levels)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power (p.u.)')
    for p in P_slice:
        plt.axhline(y=p, color='w', linestyle='--', alpha=0.7)
    cbar = plt.colorbar(cs1)
    cbar.set_label(r'$\delta(P,t)$')
    total_time, _, delta_numerical_solution, _ = RK45Solver(
                                    m=m, d=d, N_t=N_t, N_P=N_P)
    print('RK45 Duration = {:.2f}'.format(total_time))

    idx = []
    P_slices = []
    for P in P_slice:
        index, val = find_nearest(Pnum, P)
        P_slices.append(val)
        idx.append(index)

    for i, P_cs in enumerate(P_slices):
        plt.subplot(gs[1, i])
        tPmdslice = np.stack([tnum,
                              np.full(tnum.shape, P_cs),
                              np.full(tnum.shape, m),
                              np.full(tnum.shape, d)], axis=-1)
        u = base_model.predict(tPmdslice)
        plt.plot(tnum, u, label='PINN Prediction')
        plt.plot(tnum,
                 delta_numerical_solution[idx[i], :],
                 color='red',
                 linestyle='dashed',
                 label='Numerical Solution')
        if i == 0:
            plt.legend(fontsize='xx-small', loc='lower center')

        plt.title(r'$P = {:.2f}\  p.u.$'.format(P_cs))
        plt.xlabel(r'Time (seconds)')
        plt.ylabel(r'$\delta(P,t)$')
    plt.tight_layout()
    if save:
        plt.savefig(save_name, dpi=300, transparent=True)
    plt.show()


def ErrorPlot(PINNModel, m=0.1,
              d=0.05, Prange=[0.08, 0.18],
              N_P=100, N_t=100,
              levels=80):

    Pnum = np.linspace(Prange[0], Prange[1], N_P)
    tnum = np.linspace(0, 20, N_t)
    mvals, dvals = np.repeat(m, N_t*N_P), np.repeat(d, N_t*N_P)
    t, P = np.meshgrid(tnum, Pnum)

    tPmd = np.stack([t.flatten(), P.flatten(), mvals, dvals], axis=-1)
    x = [tPmd, tPmd, tPmd]
    f_out, _, _ = PINNModel(x)
    f_out = f_out.numpy().reshape((N_P, N_t))
    cs1 = plt.contourf(t, P, np.abs(f_out), levels=levels)
    cbar = plt.colorbar(cs1)
    cbar.set_label('Error')
