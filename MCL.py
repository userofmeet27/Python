import numpy as np
m= np.array([10, 7, 14, 3, 8, 5, 12, 9])
mean = np.sum(m) / len(m)
sorted_array = np.sort(m)
print(sorted_array)
if len(sorted_array) % 2 == 0:
    middle1 = sorted_array[len(sorted_array) // 2 - 1]
    middle2 = sorted_array[len(sorted_array) // 2]
    median = (middle1 + middle2) / 2
else:
    median = sorted_array[len(sorted_array) // 2]
min_value = sorted_array[0]
print("Array:",m)
print("Mean:",mean)
print("Median:",median)
print("Minimum:",min_value)

#%%

import numpy as np
A = np.array([[2, 4], [5, -6]])
B = np.array([[9, -3], [3, 6]])
C = A.dot(B)
print(C)
C = A + B
print(C)
#Note:(*)is used for array multiplication (multiplication of corresponding elements of two arrays) not matrix multiplication.
import numpy as np
A = np.array([[2, 4], [5, -6]])
B = np.array([[9, -3], [3, 6]])
C = A + B
D = A*B # element wise multiplicatioon
print(C)
print(D)

#%%

import random
array_size = 10
my_array = [random.randint(1, 100) for i in range(array_size)]
print("Original Array:", my_array)
for i in range(array_size):
    min_index = i
    for j in range(i + 1, array_size):
        if my_array[j] < my_array[min_index]:
            min_index = j
    my_array[i], my_array[min_index] = my_array[min_index], my_array[i]
print("Sorted Array:", my_array)

#%%

def factorial(n):
    if n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
n = 7
fact = factorial(n)
print(f"The factorial of {n} is {fact}")


#%%

# importing the required functions from various modules
from numpy import array,mat,transpose,zeros,identity,eye,diag,shape,ones 
from numpy.linalg import inv,det,eig,matrix_rank
from scipy.linalg import lu

# A and B matrix defined using keyword "array" 
A=array([[2,3,7],[4,1,11],[5,13,9]])
B=array([[5,8,11],[3,7,4],[13,2,1]])

# Y and Z matrix defined using keyword "mat" (Elements same as in A and B) 
Y=mat([[2, 3, 7],[4, 1, 11],[5,13,9]])
Z=mat([[5,8,11],[3,7,4],[13,2,1]])

print("A = \n",A,"\n\nB= \n",B) 
print("\nY= \n",Y,"\n\nZ= \n",Z)

# Size of matrix for both cases (matrix defined by array and mat)
m,n=shape(A); m1,n1=shape(Z); # m and n indicates row and column respectively

print("\nm=",m,"\n\nn=",n) 
print("\nm1=",m1,"\n\nn1=",n1)
 
# Addition & Subtraction of matrix for both cases (matrix defined by array and mat) 
C=A+B; D=A-B;
C1=Y+Z; D1=Y-Z;

print("\nAddition=\n",C,"\n\nAddition1=\n",C1) 
print("\nSubtraction=\n",D,"\n\nSubtraction1=\n",D1)

# Multiplication of matrix for both cases (matrix defined by array and mat) 
E=A*B; F=A.dot(B);
E1=Y*Z; F1=Y.dot(Z);

print("\nMul1=\n",E,"\n\nMul11=\n",E1) 
print("\nMul2=\n",F,"\n\nMul21=\n",F1)

# Inverse, transpose and determinant of matrix for both cases (matrix defined by array and mat)
G=inv(A);G1=inv(Y);G2=Y**(-1);
H=transpose(A);H1=transpose(Y); H2=round(det(A)); H3=round(det(Y))

print("\nInverse=\n",G,"\n\nInverse1=\n",G1,"\n\nInverse2=\n",G2) 
print("\nTranspose=\n",H,"\n\nTranspose1=\n",H1)
print("\nDeterminant=\n",H2,"\n\nDeterminant1=\n",H3)

# General keywords for getting all zero or one or identity matrix or diagonal matrix 
I=identity(3);I1=eye(3); # identity matrix of size 3*3
J=ones([2,3])	# matrix of size 2*3 having all elements as 1 
K=zeros([3,3])	# matrix of size 3*3 having all elements as 0 
P=diag([4,3,2])	# diagonal matrix

print("\nI=\n",I,"\n\nI1=\n",I1) 
print("\nJ=\n",J,"\n\nK=\n",K) 
print("\nP=\n",P)

# Rank,Eigen value & Eigen Vectors of matrix for both cases (matrix defined by array and # mat)
R=matrix_rank(A); R1=matrix_rank(Y) 
Val,Vect=eig(A); Val1,Vect1=eig(Y)

print("\nRank=\n",R,"\n\nRank1=\n",R1)
print("\nEigen value=\n",Val,"\n\nEigen Vectors=\n",Vect) 
print("\nEigen value1=\n",Val1,"\n\nEigen Vectors1=\n",Vect1)

# Lower & Upper triangle of the matrix 
L=lu(A)[1]; 
L1=lu(Y)[1]
U=lu(A)[2]; U1=lu(Y)[2]

print("\nL=\n",L,"\n\nL1=\n",L1)
print("\nU=\n",U,"\n\nU1=\n",U1)
 
#%%

a = [[2, 5, -8], [5, -7, 3], [-9, 4, 11]]
b = [-3, 7, 13]
n = len(b)
for i in range(n):
    pivot = a[i][i]
    for j in range(i + 1, n):
        factor = a[j][i] / pivot
        b[j] -= factor * b[i]
        for k in range(i, n):
            a[j][k] -= factor * a[i][k]
x = [0] * n
for i in range(n - 1, -1, -1):
    x[i] = b[i]
    for j in range(i + 1, n):
        x[i] -= a[i][j] * x[j]
    x[i] /= a[i][i]
print("Solution:")
for i in range(n):
    print(f"x{i+1} = {x[i]}")

#%%

import numpy as np
matrix = np.zeros((5, 10))
print(matrix)

#%%

import numpy as np
matrix = np.full((4, 5), 5)
print(matrix)

#%%

tuple = ( 'abcd', 786 , 2.23, 'john', 70.2 )
list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
# tuple[2] = 1000	# Invalid syntax with tuple
list[2] = 1000
print(list)

#%%
import numpy as np
from numpy.linalg import inv
b = np.matrix([[2, 5, -8], [5, -7, 3], [-9, 4, 11]])
a = np.matrix([[-3], [7], [13]])
x=inv(b)
y=x.dot(a)
print(y)
print("I1 = ",y[0],"I2 = ",y[1],"I3 = ",y[2])

#%%

import numpy as np
from numpy.linalg import inv
b = np.matrix([[0.25, -1/6], [-1/6, 1/3]])
a = np.matrix([[1], [-4]])
x=b**(-1)
y=x.dot(a)
print(y)

#%%

import numpy as np
from numpy.linalg import inv
b = np.matrix([[12, -6], [-6, 9]])
a = np.matrix([[12], [3]])
x=inv(b)
y=x.dot(a)
print(y)

#%%

import numpy as np
from numpy.linalg import inv
b = np.matrix([[0.75, -0.25], [-0.25, 1/4+1/6]])
a = np.matrix([[5], [5]])
x=inv(b)
y=x.dot(a)
print(y)
print(y[0])

#%%

import matplotlib.pyplot as plt	# importing the required module
x = [1,2,3]	# x axis values
y = [2,4,1]	# corresponding y axis values
plt.plot(x, y)	# plotting the points
plt.xlabel('x - axis')	# naming the x axis
plt.ylabel('y - axis')	# naming the y axis
plt.title('My first graph!')	# giving a title to my graph
plt.show()

#%%

import matplotlib.pyplot as plt

# line 1 points 
x1 = [1,2,3]
y1 = [2,4,1]
plt.plot(x1, y1, label = "line 1")	# plotting the line 1 points

# line 2 points 
x2 = [1,2,3]
y2 = [4,1,3]
plt.plot(x2, y2, label = "line 2")	# plotting the line 2 points

plt.xlabel('x - axis')	# naming the x axis
plt.ylabel('y – axis')	# naming the y axis 
plt.title('Two lines on same graph!')	# giving a title to my graph plt.legend()	# show a legend on the plot
plt.show()

#%%

import matplotlib.pyplot as plt
x1 = [1,2,3]
y1 = [2,4,1]
plt.plot(x1, y1, label = "line 1")
x2 = [1,2,3]
y2 = [4,1,3]
plt.plot(x2, y2, label = "line 2")
x3 = [0,2,4]
y3 = [0,1,3]
plt.plot(x3, y3, label = "line 3")
plt.xlabel('x - axis')	
plt.ylabel('y – axis')	
plt.title('Three lines on same graph!')	
plt.legend()
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
# function to generate coordinates
def create_plot(ptype):
# setting the x-axis vaues
    x = np.arange(-10, 10, 0.1)

    if ptype == 'linear':
        y = x
    elif ptype == 'quadratic':
        y = x**2        
    elif ptype == 'cubic':
        y = x**3
    elif ptype == 'quartic':
        y = x**4
    return(x, y)
plt.style.use('fivethirtyeight')
fig = plt.figure()
plt1 = fig.add_subplot(221)
plt2 = fig.add_subplot(222)
plt3 = fig.add_subplot(223)
plt4 = fig.add_subplot(224)
x, y = create_plot('linear')
plt1.plot(x, y, color ='r')
plt1.set_title('$y_1 = x$')
x, y = create_plot('quadratic')
plt2.plot(x, y, color ='b')
plt2.set_title('$y_2 = x^2$')
x, y = create_plot('cubic')
plt3.plot(x, y, color ='g')
plt3.set_title('$y_3 = x^3$')
x, y = create_plot('quartic')
plt4.plot(x, y, color ='k')
plt4.set_title('$y_4 = x^4$')
fig.subplots_adjust(hspace=.5,wspace=0.5)
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 4*(np.pi), 0.01) # setting the x - coordinates
y = np.sin(x) # setting the corresponding y -coordinates
plt.plot(x, y) # potting the points
plt.show() 
fruits = ["apple", "banana", "cherry"]
for x in fruits:
 print(x)
 if x == "banana":
     break

#%%

c=0
for i in range(2,20):
    for j in range(2,20):
        if i % j == 0:
            break
    if i==j:
        print(i)

#%%

import matplotlib.pyplot as plt
import numpy as np
fig=plt.figure()
f=1000
w=2*f*np.pi
t=np.arange(0,0.001,0.001/f)
y=np.sin(w*t)*4/(np.pi)
plt1=fig.add_subplot(121)
z=y
plt1.plot(t,z)
plt2=fig.add_subplot(122)
y1=4*np.sin(3*w*t)/(3*np.pi)
z1=y+y1
plt2.plot(t,z1)
plt.show()

#%%
import numpy as np
b = [951, 402, 984, 651, 360, 69, 408, 319, 601, 485, 980, 507, 725,
          544, 615, 83, 165, 575, 219, 390, 984, 592, 236, 105, 942, 941, 386, 462, 47,
          907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345, 399, 162, 758,
          918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217, 815, 67, 104, 58,
          24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 753, 854, 685, 93, 857, 440,
          126, 721, 328, 753, 470, 743, 527]
Numbers=np.sort(b)
for number in Numbers:
    if number > 237:
        break 
    elif number % 2 == 0:
        print(number)
        
#%%

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
for num in range(2, 201):
    if is_prime(num):
        print(num)
        
#%%

A = [10, 9, 8, 3, 1, 0]
for i in range(len(A)):
    for j in range(i + 1, len(A)):
        if A[i] < A[j]:
            A[i], A[j] = A[j], A[i]
print(A)

#%%

A='meet'
print(A[::-1])

#%%

import matplotlib.pyplot as plt
import numpy as np
fig=plt.figure()
f=1000
w=2*f*np.pi
t=np.arange(0,1*0.001,0.001/f)
y=np.sin(w*t)*4/(np.pi)
plt1=fig.add_subplot(241)
z=y
plt1.plot(t,z)
plt2=fig.add_subplot(242)
y1=4*np.sin(3*w*t)/(3*np.pi)
z1=y+y1
plt2.plot(t,z1)
plt3=fig.add_subplot(243)
y2=4*np.sin(5*w*t)/(5*np.pi)
z2=z1+y2
plt3.plot(t,z2)
plt4=fig.add_subplot(244)
y3=4*np.sin(7*w*t)/(7*np.pi)
z3=z2+y3
plt4.plot(t,z3)
plt5=fig.add_subplot(245)
y4=4*np.sin(9*w*t)/(9*np.pi)
z4=z3+y4
plt5.plot(t,z4)
plt6=fig.add_subplot(246)
y5=4*np.sin(11*w*t)/(11*np.pi)
z5=z4+y5
plt6.plot(t,z5)
plt7=fig.add_subplot(247)
y6=4*np.sin(13*w*t)/(13*np.pi)
z6=z5+y6
plt7.plot(t,z6)
plt8=fig.add_subplot(248)
y7=4*np.sin(15*w*t)/(15*np.pi)
z7=z6+y7
plt8.plot(t,z7)
plt.show()


#%%

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

#%%

import math
R = 350 
C = 45*10**-6 
f = 50 
impedance = math.sqrt(R**2 + (1 / (2 * math.pi * f * C))**2)
print(f"Impedance of the series RC circuit: {impedance:.2f} ohms")

#%%

import math
R = 350 
L = 100e-3
impedance = math.sqrt(R**2 + (2 * math.pi * f * L)**2)
print(f"Impedance of the series RL circuit: {impedance:.2f} ohms")

#%%

import math
R = 250  
L1 = 1*10**-3
L2 =  1*10**-3
C1 = 55*10**-6 
C2 = 555*10**-6 
f = 50 
ZL1 = 1 * 2 * math.pi * f * L1
ZC1 = 1 / (1 * 2 * math.pi * f * C1)
ZL2 = 1 * 2 * math.pi * f * L2
ZC2 = 1 / (1 * 2 * math.pi * f * C2)
Zbranch1 = 1 / (1 / ZL1 + 1 / ZC1)
Zbranch2 = 1 / (1 / ZL2 + 1 / ZC2)
Ztotal = R + 1 / (1 / Zbranch1 + 1 / Zbranch2)
print(f"Total impedance of the circuit: {abs(Ztotal):.2f} ohms")

#%%

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
R=200;
C=15*10**-6;
L=230*10**-3;
f=np.arange(50.69,150.69,1)
temp=[];
for i in range(len(f)):
 XL=2*sp.pi*f[i]*L;
 XC=1/(2*sp.pi*f[i]*C);
 Z=np.sqrt(R*R+(XL-XC)*(XL-XC));
 temp.append(Z);
plt.plot(f,temp);
plt.title("RLC Impedance");
plt.xlabel("Frequency");
plt.ylabel("Impedance");
plt.show()
f0=1/(2*sp.pi*np.sqrt(L*C));
print("Resonance Frequency = ",f0); 

#%%

import math
Vpeak = 230 * math.sqrt(2)  
R = 50  
L = 33*10**-3  
f = 50  
w = 2 * math.pi * f
Z = math.sqrt(R**2 + (1 *w* L)**2)
I = Vpeak / Z
P = abs(I)**2 * R
S = I * Vpeak.conjugate()
cos_phi = P / abs(S)
print(f"Current (I): {abs(I):.2f} A")
print(f"Average Power (P): {P:.2f} W")
print(f"Apparent Power (S): {abs(S):.2f} VA")
print(f"Power Factor (PF): {cos_phi:.2f}")

#%%

import math
Vrms = 230 
f = 50  
R = 50  
L = 33*10**-3
omega = 2 * math.pi * f
Z = math.sqrt(R**2 + (omega * L)**2)
I = Vrms / Z
S = Vrms * I
phi = math.atan(omega * L / R)  
P = Vrms * I * math.cos(phi)
PF = P / S
print(f"Current (I): {I:.2f} A")
print(f"Apparent Power (S): {S:.2f} VA")
print(f"Real Power (P): {P:.2f} W")
print(f"Power Factor (PF): {PF:.2f}")

#%%

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


#%%

def equation(x):
    return x**3 + 4*x**2 - 10
def bisection_method(a, b, tol, max_iter):
    if equation(a) * equation(b) >= 0:
        return None

    iter_count = 0
    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if equation(midpoint) == 0:
            return midpoint
        elif equation(a) * equation(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

        iter_count += 1
        if iter_count >= max_iter:
            print("Maximum iterations reached.")
            break

    return (a + b) / 2.0
a = 1.0
b = 2.0
tolerance = 1e-6
max_iterations = 100
root = bisection_method(a, b, tolerance, max_iterations)
if root is not None:
    print(f"Approximate root: {root:.6f}")
    print(f"Value at the root: {equation(root):.6f}")

#%%

def secant_method(f, x0, x1, tol, max_iter):
    for i in range(max_iter):
        x2 = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    raise Exception("Secant method did not converge")
def equation(x):
    return x**3 + 4*x**2 - 10
x0 = 1.0
x1 = 2.0
tolerance = 1e-6
max_iterations = 100
root = secant_method(equation, x0, x1, tolerance, max_iterations)
print(f"Approximate root: {root:.6f}")

#%%

import math
def false_position(func, a, b, tolerance=1e-6, max_iterations=100):
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have different signs at the endpoints.")
    
    iterations = 0
    while iterations < max_iterations:
        c = a - (func(a) * (b - a)) / (func(b) - func(a))
        if abs(func(c)) < tolerance:
            return c
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        iterations += 1

    raise ValueError("False position method did not converge within the specified number of iterations.")
def equation(x):
    return x**3 + 4*x**2 - 10
a = 1
b = 2
root = false_position(equation, a, b)
print(f"Approximate root: {root:.6f}")

#%%

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

#%%
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

#%%

from sympy import Symbol 
import numpy as np
import matplotlib.pyplot as plt 
x=Symbol('x')
y=[]

expr1=(np.sqrt(2+x)-np.sqrt(x))/x
Ts = 1.0/10;	# incremental fraction

#vary value from some 0.1 to just below 2. from just above 2 to higher values do not assign value 2
vx = np.arange(1.01,1.99,Ts) 
for i in range(len(vx)):
    f=expr1.subs(x,vx[i]) 
    y.append(f)
plt.plot(vx,(y))

#%%

from sympy import Symbol, sqrt, limit
x = Symbol('x')
expr2 = (sqrt(2 + x) - sqrt(x)) / x
limit_result = limit(expr2, x, 0)
print(f"Limit result as x tends to 0: {limit_result}")

#%%

from sympy import Symbol,Derivative  
y=Symbol('y')
x=Symbol('x')
function = x**2*y**3+12*y**4 
partialderiv=Derivative(function,x)
partialderiv1=Derivative(partialderiv,x)  
pdfunc1=partialderiv1.doit() 
print(pdfunc1)
partialderiv=Derivative(function,y)
partialderiv1=Derivative(partialderiv,y) 
pdfunc1=partialderiv1.doit() 
print(pdfunc1)

#%%

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


# Creating histogram
#fig, ax = plt.subplots(figsize =(10,6 ))
#ax.hist(n1,bins=np.arange(-5,5,0.1))
# Show plot
#plt.show()
'''plt.bar(x, y) # plotting the points
plt.xlabel('x - axis') # naming the x axis
plt.ylabel('y - axis') # naming the y axis
plt.title('My first graph!') # giving a title to my graph
#plt.show() # function to sh#i=linspace(1,20,10)
#print(i)''' 

#%%

import numpy as np
a = 10
count = 0
rn = np.random.randint(1, 3, size=a)
print("The outcomes of the coin are:", rn)
for outcome in rn:
    if outcome == 1:
        count += 1
print("Probability of head :", count/10)
