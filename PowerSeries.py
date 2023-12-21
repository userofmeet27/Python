import numpy as np
import matplotlib.pyplot as plt

x=[]
y=[]
for i in np.arange(0,1,0.1):
    ans=1/(1-i)
    y.append(ans)
    x.append(i)
plt.plot(x,y)
x1 = []
y1 = []
x1 = np.arange(0,1,0.1)
y1 = 1+x1+x1**2+x1**3+x1**4
plt.plot(x1,y1)
plt.show()