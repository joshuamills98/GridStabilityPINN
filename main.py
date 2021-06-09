import numpy as np
import argparse
from basenetwork import Network
from PINN import PINNModel
from optimizer import Optimizer
from data_prep import data_prep
from plottingtools import plot_contour
import pickle
import os
import time

"""
We will solve the equation between 0 and Xmax and 0 and Tmax
"""

os.chdir('C:\\Users\\joshb\\Desktop\\Machine_Learning\\GridStabilityPINN\\')
# Global Parameters for training
N_col = 10000  # Number of collocation points
N_bound_ini = 100  # Total number of boundary and initial condition points
N_test_points = 1000

base_model = Network.base_model(hidden_layers=[20]*10)
GridStabilityPINN = PINNModel(base_model).build()
x_train, y_train = data_prep(N_col=100, N_ini=300)

optimizer = Optimizer(model=GridStabilityPINN,
                      x_train=x_train, y_train=y_train,
                      maxiter=20)

parser = argparse.ArgumentParser(description="Retrain selection")
parser.add_argument(
    'retrain', type=str,
    default='No',
    help="Train from scratch (Yes) or use pretrained weights (No)")
args = parser.parse_args()

if __name__ == '__main__':
    if args.retrain == 'Yes':
        optimizer.fit(2500)
        with open('NNWeights10x20-{}.pickle'
                  .format(time.strftime("%H-%M-%S")), 'wb') as handle:
            pickle.dump(base_model.weights, handle)

    if args.retrain == 'No':
        weights = pickle.load(open("NNWeights10x20.pickle", "rb"))
        flattened_weights = np.concatenate(
            [np.array(w).flatten() for w in weights])
        optimizer.set_weights(flattened_weights)
        K_test = 0.2
        V_test = 0.4
        xbounds = [0, 1]
        tbounds = [0, 1]
        plot_contour(base_model, N_P=100, save=False)
