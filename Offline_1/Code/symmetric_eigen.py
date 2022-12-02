import numpy as np

limit = 100         # range of integers to be generated
tolerance = 1e-06   # teolrance of error


# Generate symmetric invertible matrix of integers
# computing (A + A.T)/2 for arbitrary square matrix A gives a symmetric matrix
def generate_symmetrical_invertible_matrix(n):
    matrix = np.random.randint(limit, size = (n, n))
    matrix = 2 * matrix                                 # multiplying with 2 so the (A+A.T)/2 remains integer
    matrix = (matrix + matrix.T)/2
    return matrix
    
# Matrix reconstruction is done with the following formula: 
# A = V . diag(λ) . V^−1.
def reconstruct(eigen_values, eigen_vectors):
    lambda_matrix = np.diag(eigen_values)
    inv_eigenvector = np.linalg.inv(eigen_vectors)
    reconstructed_matrix = np.matmul(lambda_matrix, inv_eigenvector)
    reconstructed_matrix = np.matmul(eigen_vectors, reconstructed_matrix)
    return reconstructed_matrix

    

# ----------- driver function --------------- #
dim = int(input("Enter the dimension of the matrix : "))

M = generate_symmetrical_invertible_matrix(dim)
eigen_values, eigen_vectors = np.linalg.eig(M)      # break the matrix into eigen values and eigen vectors
reconstructed_matrix = reconstruct(eigen_values, eigen_vectors)

print(f'Generated Matrix : \n {M} \n')
print(f'Eigen Values : \n {eigen_values} \n')
print(f'Eigen Vectors : \n {eigen_vectors} \n')
print(f'Reconstructed Matrix \n {reconstructed_matrix} \n')
print(f'Is Reconstruction Perfect? : {np.allclose(M, reconstructed_matrix, tolerance)} \n')