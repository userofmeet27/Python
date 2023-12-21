import matplotlib.pyplot as plt # importing the required module
import numpy as np
fig = plt.figure()
plt1 = fig.add_subplot(311)
plt2 = fig.add_subplot(312)
plt3 = fig.add_subplot(313)
plt.style.use('fivethirtyeight')
#Part 1: Generate array of random numbers
rn=np.random.rand(1,10)
print(np.mean(rn))
#Part 2: Generate random number with Gaussian/Normal Distribution
n1 = np.random.normal(0.0, 1, 1000)
print(np.mean(n1)) #for verification
plt1.hist(n1, bins=np.arange(-5,5,0.1))
#Part 3: Generate a sine wave and add gaussian noise to it
t=np.arange(0,0.1,0.0001)
x=10*np.sin(2*3.14*50*t)
#print(n1)
plt2.plot(t, x, color ='r')
y=x+n1
plt3.plot(t,y,'r')