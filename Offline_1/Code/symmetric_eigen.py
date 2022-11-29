import numpy as np

limit = 100         # range of integers to be generated
tolerance = 1e-06   # teolrance of error
dim = 5             # dimension of matrix

# Generate symmetric invertible matrix of integers
# computing (A + A.T)/2 for arbitrary square matrix A
def generate_symmetrical_invertible_matrix(n):
    matrix = np.random.randint(limit, size = (n, n))
    matrix = 2 * matrix                                 # multiplying with 2 so the (A+A.T)/2 remains integer
    matrix = (matrix + matrix.T)/2
    return matrix
    
def check_reconstruction(M, eigen_values, eigen_vectors):
    lambda_matrix = np.diag(eigen_values)
    inv_eigenvector = np.linalg.inv(eigen_vectors)
    reconstructed_matrix = np.matmul(lambda_matrix, inv_eigenvector)
    reconstructed_matrix = np.matmul(eigen_vectors, reconstructed_matrix)
    
    print(f'Reconstructed Matrix \n {reconstructed_matrix}')
    print(f'Is Reconstruction Perfect? : {np.allclose(M, reconstructed_matrix, tolerance)}')
 
 
# ----------- driver function --------------- #
M = generate_symmetrical_invertible_matrix(dim)
print(f'Generated Matrix : \n {M}')
eigen_values, eigen_vectors = np.linalg.eig(M)
print(f'Eigen Values : \n {eigen_values}')
print(f'Eigen Vectors : \n {eigen_vectors}')
check_reconstruction(M, eigen_values, eigen_vectors)