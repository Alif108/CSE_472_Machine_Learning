import numpy as np

limit = 100         # range of integers to be generated
tolerance = 1e-06   # teolrance of error

# Generate a Random Invertible Matrix of integers
# In mathematics, a square matrix is said to be diagonally dominant if, 
# for every row of the matrix, the magnitude of the diagonal entry in a row is 
# larger than or equal to the sum of the magnitudes of all the other (non-diagonal) entries in that row. 
# A diagonally dominant matrix is invertible.
# In the following function, the diagonal entry of each row is replaced by the sum of that row. 
# So that entry is always greater than all ither entries of the row
def generate_invertible_matrix(n):
    matrix = np.random.randint(limit, size = (n, n))
    row_sums = np.sum(np.abs(matrix), axis = 1)
    np.fill_diagonal(matrix, row_sums)
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

M = generate_invertible_matrix(dim)
eigen_values, eigen_vectors = np.linalg.eig(M)      # break the matrix into eigen values and eigen vectors
reconstructed_matrix = reconstruct(eigen_values, eigen_vectors)

print(f'Generated Matrix : \n {M}')
print(f'Eigen Values : \n {eigen_values}')
print(f'Eigen Vectors : \n {eigen_vectors}')
print(f'Reconstructed Matrix : \n {reconstructed_matrix}')
print(f'Is Reconstruction Perfect? : {np.allclose(M, reconstructed_matrix, tolerance)}')