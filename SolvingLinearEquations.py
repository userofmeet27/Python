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