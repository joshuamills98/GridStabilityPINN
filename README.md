## GridStabilityPINN

These files were developed as part of my Master's Thesis on the implementation of Physics-Informed Neural Networks (PINNs) within the engineering design platform FLOW. Here, PINNs are used to solve the Swing equation given by:

![equation](https://latex.codecogs.com/gif.latex?m%5CDdot%7B%5Cdelta%7D%20&plus;d%5CDot%7B%5Cdelta%7D%20&plus;%20B_%7B12%7DV_%7B1%7DV_%7B2%7D%20%5Csin%7B%5Cdelta%7D%20-%20P%20%3D%200)

This equation is relevant in the stability analysis of the electrical grid and describes the response of the *rotor angle* under changing generator loads. Rotor angle stability is important, particularly given the rise of low inertia energy resources. The equation explores the Single Machine Infinite Bus model shown below (image taken from [[1]](#1)):

![image](/plots/SMIB.png "SMIB system")

While Misyris et al. 2020 has shown PINNs to be effective at modelling the output of the function for given generator inertial constant *m* and damping coefficient *d*, here I extend this work
to explore the PINNs ability to generalize so that:

![equation](https://latex.codecogs.com/gif.latex?NN%28t%2CP_%7B1%7D%2C%20m%2C%20d%29%20%5Capprox%20%5Cdelta%28t%2CP_%7B1%7D%2C%20m%2C%20d%29)

The goal was to develop a neural network that could replace pre-existing numerical simulations which are far more taxing computationally. While the PINN showed an *4-5x* speed up over pre-existing numerical solvers, the accuracy of the PINN was varied over the domain, future research should explore larger networks and perhaps incorporation of multiple 
PINNs, each of which is specialized to a particular region of the solution.

Below shows the output of the PINN for *m=0.4* and *d=0.15*. 

![image](/plots/result.jpg "Output of the PINN for *m=0.4* and *d=0.15*")

# Files:

* **basenetwork.py-** Underlying neural network structure 
* **derivativelayer.py-** Residual network operations
* **PINN.py-** The entire construction of the PINN is wrapped up here
* **optimizer.py-** Creation of the optimizer object for training of the PINN
* **data_prep.py** Collocation points and boundary training data are developed and prepared for the training of the model
* **plottingtools.py** Tools for plotting the error and results of the PINN
* **numerical_solver.py-** Runge-Kutta numerical solution for comparison
* **NNWights10x20.pickle-** Pre-trained weights to use for exploration of the solution


## References
<a id="1">[1]</a> 
George S. Misyris and Andreas Venzke and Spyros Chatzivasileiadis (2020). 
Physics-Informed Neural Networks for Power Systems.
