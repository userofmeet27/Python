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