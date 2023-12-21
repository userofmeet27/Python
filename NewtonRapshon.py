import numpy as np
import matplotlib.pyplot as plt
def fcn_nr(x):
 y=x**3+4*(x)**2-10;
 deriv=3*(x)**2+8*x;
 return [y,deriv]; 
x=100+100j
itermax=100;
iter=0;
errmax=0.000012
error1=1;   
while(error1>errmax and iter<itermax):
    iter=iter+1;
    f=fcn_nr(x); 
    if (f[1]==0):
        break;
    xnew=x-f[0]/f[1]
    error1=np.abs((xnew-x)/xnew)*100;
    x=xnew
print(x) 
