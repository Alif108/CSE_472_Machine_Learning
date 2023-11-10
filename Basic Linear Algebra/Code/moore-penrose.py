import numpy as np

limit = 100         # range of integers to be generated
tolerance = 1e-06   # tolerance of error

# generate a random matrix of dimension m x n of integers
def generate_matrix(m, n):
    matrix = np.random.randint(limit, size = (m, n))
    return matrix

# Moore-Penrose Pseudoinverse
# A+ = V. D+ . U^T 
# where D+ = diag(1/λ1, 1/λ2, ..., 1/λn)
def moore_penrose_pseudoinverse(U, D, V):
    diagonal_D = np.zeros((U.shape[0], V.shape[0]))

    for i in range(len(D)):
        diagonal_D[i][i] = 1/D[i]
    D_plus = diagonal_D.T
 
    A_plus = np.matmul(D_plus, U.T)
    A_plus = np.matmul(V.T, A_plus)
    return A_plus


# ----------- driver function --------------- #
m  = int(input("Enter the number of rows of the matrix : "))
n  = int(input("Enter the number of columns of the matrix : "))

A = generate_matrix(m, n)
U, D, V = np.linalg.svd(A)      # break the matrix into U, D, V
A_pseudo = moore_penrose_pseudoinverse(U, D, V)
A_pseudo_numpy = np.linalg.pinv(A)

print(f'Generated Matrix : \n {A} \n')
print(f'Left Singular Vectors, U : \n {U} \n')
print(f'Singular Values, D : \n {D} \n')
print(f'Right Singular Values, V : \n {V.T} \n')
print(f'Moore-Penrose Pseudoinverse by Equation: \n {A_pseudo} \n')
print(f'Moore-Penrose Pseudoinverse by Numpy: \n {A_pseudo_numpy} \n')
print(f'Is Reconstruction Perfect? : {np.allclose(A_pseudo, A_pseudo_numpy, tolerance)} \n')