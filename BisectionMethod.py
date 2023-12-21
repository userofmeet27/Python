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