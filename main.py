#%%
import os
import tensorflow as tf
import numpy as np

from derivativelayer import DerivativeLayer
from basenetwork import Network
from PINN import PINNModel
from optimizer import Optimizer
from pyDOE import lhs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from plottingtools import plot, errorplot
from data_prep import prep_data
"""
We will solve the equation between 0 and Xmax and 0 and Tmax
"""

#Global Parameters for training
N_col = 10000 #Number of collocation points
N_bound = 100 #total number of boundary and initial condition points
N_ini = 100
N_test_points = 1000
V=1
K=1

#Define initial and boundary conditions
def initialcondition(x):
    return np.sin(np.pi*x)

def boundarycondition1(t): 
    return 0

def boundarycondition2(t):
    return 0

x_train, y_train = prep_data()
#model initialization

basemodel = Network.basemodel()
PINNConvDiff = PINNModel(basemodel, V=V,K=K).build()


optimizer = Optimizer(model=PINNConvDiff, x_train=x_train, y_train=y_train,maxiter= 20)
optimizer.fit()
#%%
plot(basemodel,100,1,1)
errorplot(PINNConvDiff,100,1,1)