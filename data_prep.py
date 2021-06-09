"""
This function will prepare the domain data for use by the PINN

"""
import numpy as np
from smt.sampling_methods import LHS


def data_prep(t_range=(0, 20), P_range=(0.08, 0.18),
              N_col=10000, N_ini=100,
              m_range=(0.1, 0.4), d_range=(0.05, 0.15),
              delta0=0.1, omega0=0.1):

    """
    Arguments:
        - t_range: Tuple indiciating lower and upper bound of time domain
        - P_range: Tuple indicating lower and upper bound of Power domain
        - N_col: Number of collocation points
        - N_ini : Numper of pointsfor initial condition
        - mVals: List of values for inertia constant over which you to train
        network
        - dvals: list of values for damping coefficient over which you to train
        network
        - initialcondition1: Initial angle delta(0,P)=delta_0
        - initialcondition2: Initial frequency omega(0,P) = omega_0

    Outputs:
        x_train list containing:
        - Px_col: Input Px collocation matrix of shape (N_col, 2)
        - Px_init: Input Px initial condition matrix of shape (N_ini,2)

        y_train list containing:
        - c_col: output values for collocation matrix. Shape (N_col,1) (zeros)
        - c_ini_1 : output values for initial condition delta0. Shape (N_ini,1)
        - c_ini_2 : output values for initial condition omega0. Shape (N_ini,1)
    """

    """
    write x - training data

    """
    collocation_limits = np.array([t_range, P_range, m_range, d_range])
    sampling = LHS(xlimits=collocation_limits)
    tP_col = sampling(N_col)

    tP_ini_1 = sampling(N_ini)  # Simulate N_ini random points in Xrange for
    # the initial condition
    tP_ini_1[:, 0] = t_range[0]  # Set T vals to be 0

    tP_ini_2 = sampling(N_ini)
    tP_ini_2[:, 0] = t_range[0]

    """
    Write y- training data
    """

    c_col = np.zeros((N_col, 1))
    c_ini_1 = np.repeat(delta0, N_ini).reshape(-1, 1)
    c_ini_2 = np.repeat(omega0, N_ini).reshape(-1, 1)

    x_train = [tP_col, tP_ini_1, tP_ini_2]
    y_train = [c_col, c_ini_1, c_ini_2]

    return x_train, y_train
