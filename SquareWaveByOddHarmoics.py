import matplotlib.pyplot as plt
import numpy as np
f=1000
w=2*f*np.pi
y=0
t=np.arange(0,1*0.001,0.001/f)
for i in range(1,200,2):
    y=y+np.sin(i*w*t)*4/(i*np.pi)
plt.plot(t,y)
plt.show()