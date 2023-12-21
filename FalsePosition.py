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