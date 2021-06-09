# GridStabilityPINN

These files were developed as part of my Master's Thesis on Physics-Informed Neural Networks (PINNs) within FLOW. Here, PINNs are used to solve the Swing equation given by:

![equation](https://latex.codecogs.com/gif.latex?m%5CDdot%7B%5Cdelta%7D%20&plus;d%5CDot%7B%5Cdelta%7D%20&plus;%20B_%7B12%7DV_%7B1%7DV_%7B2%7D%20%5Csin%7B%5Cdelta%7D%20-%20P%20%3D%200)

This equation is relevant in the stability analysis of the electrical grid and describes the response of the *rotor angle* under changing generator loads. The equation explores the Single Machine Infinite Bus model shown below:



While Misyris et al. 2020 has shown PINNs to be effective at modelling the output of the function for given generator inertial constant *m* and damping coefficient *d*, here I extend this work
to explore the PINNs ability to generalize so that:

![equation(https://latex.codecogs.com/gif.latex?NN%28t%2CP_%7B1%7D%2C%20m%2C%20d%29%20%5Capprox%20%5Cdelta%28t%2CP_%7B1%7D%2C%20m%2C%20d%29)



To my knowledge, PINNs have not yet been implemented to solve the convection-diffusion equation and no research performed on allowing the PINN to generalize over a range of system parameters (in this case D and V) The goal is for the PINN to learn the solution

equation

@misc{GridPINN,
      title={Physics-Informed Neural Networks for Power Systems}, 
      author={George S. Misyris and Andreas Venzke and Spyros Chatzivasileiadis},
      year={2020},
      eprint={1911.03737},
      archivePrefix={arXiv},
      primaryClass={eess.SY}
