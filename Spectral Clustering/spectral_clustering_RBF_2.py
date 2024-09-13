import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

# Generate sample data (e.g., two moons)
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

# Step 1: Compute the Affinity Matrix using Gaussian Kernel
# Choose a value for gamma (1 / (2 * sigma^2))
gamma = 1.0
affinity_matrix = rbf_kernel(X, gamma=gamma)

# Step 2: Construct the Degree Matrix
degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

# Step 3: Construct the Unnormalized Laplacian Matrix
laplacian_matrix = degree_matrix - affinity_matrix

# Step 4: Compute Eigenvectors and Eigenvalues
eigvals, eigvecs = np.linalg.eigh(laplacian_matrix)

# Step 5: Select the first k Eigenvectors
# For two clusters, we choose the first two non-zero eigenvectors
k = 2
eigvecs_k = eigvecs[:, :k]

# Step 6: Perform K-means Clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(eigvecs_k)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title("Spectral Clustering with Gaussian Kernel")
plt.show()
