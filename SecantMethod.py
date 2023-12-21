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