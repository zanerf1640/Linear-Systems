import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import time

a, b = 0.0, 1.0 # Spatial domain
alpha = 37 # Ambient temperature
beta = 0.25 # Prechilled temperature

# Define the Function given
def f(x):
    return 5 * np.exp(-5 * x + 2)

m = 10 # Number of interior points
h = (b - a) / (m + 1) 
x = np.linspace(a + h, b - h, m)

# Build the matrix A
diag = -2 * np.ones(m)
off_diag = np.ones(m-1)
A = (1 / h**2) * (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))

b_vec = f(x)
b_vec[0] -= alpha / h**2
b_vec[-1] -= beta / h**2

A_pos = -A
b_pos = -b_vec

# Measure the time for Cholesky Decomp and Solving
start = time.perf_counter()
L = np.linalg.cholesky(A_pos)
y = np.linalg.solve(L, b_pos)
U = np.linalg.solve(L.T, y)
cholesky_time = time.perf_counter() - start


print("Time taken for Cholesky decomposition and solving:", cholesky_time)

# Measure the time for LU Decomp and Solving
start = time.perf_counter()
lu, piv = scipy.linalg.lu_factor(A)
U_lu = scipy.linalg.lu_solve((lu, piv), b_vec)
lu_time = time.perf_counter() - start

print("Time taken for LU decomposition and solving:", lu_time)

print("Maximum difference between solutions:", np.max(np.abs(U - U_lu)))

x_full = np.linspace(a, b, m + 2)
U_full = np.zeros(m + 2)
U_full[0], U_full[-1] = alpha, beta
U_full[1: -1] = U

plt.plot(x_full, U_full, marker='o')
plt.title("Steady-State Heat Fistribution")
plt.xlabel("x")
plt.ylabel("Temperature U(x)")
plt.grid(True)
plt.savefig("Heat_Diffusion.png")
plt.show()