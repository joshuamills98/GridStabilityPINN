"""
This function will prepare the domain data for use by the PINN

"""
import numpy as np
from smt.sampling_methods import LHS

def data_prep(Trange = (0,20), Prange = (0.08, 0.018), N_col= 10000, N_ini =100, 
                mrange = (0.1, 0.4), drange = (0.05, 0.15), delta0=0.1, omega0 = 0.1):
  #test

    """
    Arguments:
        - Trange: Tuple indiciating lower and upper bound of time domain
        - Prange: Tuple indicating lower and upper bound of Power domain
        - N_col: Number of collocation points 
        - N_ini : Numper of pointsfor initial condition
        - mVals: List of values for inertia constant over which you to train network
        - dvals: list of values for damping coefficient over which you to train network
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
    collocation_limits = np.array([Trange, Prange, mrange, drange])
    sampling = LHS(xlimits=collocation_limits)
    tP_col = sampling(N_col)

    tP_ini_1 = sampling(N_ini)  #simulate N_ini random points in Xrange for the initial condition
    tP_ini_1[:,0] = Trange[0]  # set T vals to be 0

    tP_ini_2 = sampling(N_ini)
    tP_ini_2[:,0] = Trange[0]


    """
    write y- training data
    
    """

    c_col = np.zeros((N_col,1))
    c_ini_1 = np.repeat(delta0, N_ini).reshape(-1,1)
    c_ini_2 = np.repeat(omega0, N_ini).reshape(-1,1)


    x_train = [tP_col, tP_ini_1, tP_ini_2]

    y_train = [c_col, c_ini_1, c_ini_2]

    return x_train, y_train


