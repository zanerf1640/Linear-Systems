import numpy as np

def gaussian_elimnation(A, b):

    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for k in range(n-1):
        max_index = np.argmax(abs(A[k:n, k])) + k
        if A[max_index, k] ==0:
            raise ValueError("Matrix is singular or nealy singular.")
        
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
        
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:n] = A[i, k:n] - factor * A[k, k:n]
            b[i] = b[i] - factor * b[k]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x

# List of given varaibles
V1 = 1.60   # Changed V1 to 1.60 to get Vc to be exactly 3.00V
V2 = 3.3
R1 = 5100
R2, R3, R6 = 3300, 3300, 3300
R4, R5 = 390, 390
R7, R8 = 10000, 10000

# Constructing the coefficient matrix A and constant vector matrix b
A = np.array([
    [1/R1 + 1/R3,  -1/R1,        0,     -1/R3,       0],
    [-1/R1,         1/R1 + 1/R2, -1/R2,   0,          0],
    [0,            -1/R2,         1/R2 + 1/R4, 0,     -1/R4],
    [-1/R3,         0,            0,      1/R3 + 1/R5 + 1/R7, -1/R5],
    [0,             0,           -1/R4,  -1/R5, 1/R4 + 1/R5 + 1/R6 + 1/R8]
])

b = np.array([V1 / R3, 0, 0, 0, V2 / R6])

# Solving the system of equations using Gaussian elimination
solution = gaussian_elimnation(A, b)

print("Node voltages: [Va, Vb, Vc, Vd, Ve] =", np.round(solution, 2))