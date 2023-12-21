import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt

# function that returns di/dt 
def model(i,t):
    didt = -i/(R*C) 
    return didt

# network parameters 
V=5
C=0.001 
R=100

# initial condition 
i0=V/R
t0=0

# time points
t = np.linspace(0,9,50) # solve ODE
y = odeint(model,i0,t)

# plot results 
plt.plot(t,y)
plt.grid(color='r', linestyle='-', linewidth=0.5) 
plt.rcParams.update({'font.size': 20}) 
plt.xticks(range(0,10))
plt.xlabel('time (Sec)')
plt.ylabel('current (A)') 
plt.show()