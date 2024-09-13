import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

def gaussian_kernel(X, sigma):
    """Compute the Gaussian kernel similarity matrix."""
    # Compute pairwise squared Euclidean distances
    sq_dists = pairwise_distances(X, metric='euclidean', squared=True)
    
    # Compute the Gaussian (RBF) kernel matrix
    W = np.exp(-sq_dists / (2 * sigma ** 2))
    
    return W

def spectral_clustering(X, k, sigma):
    """Perform spectral clustering using the Gaussian kernel."""
    # Step 1: Compute the similarity matrix using the Gaussian kernel
    W = gaussian_kernel(X, sigma)

    # Step 2: Compute the Degree matrix
    D = np.diag(W.sum(axis=1))

    # Step 3: Compute the unnormalized graph Laplacian
    L = D - W

    # Step 4: Compute the eigenvalues and eigenvectors of the Laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Step 5: Use the first k eigenvectors for clustering
    X_spectral = eigenvectors[:, :k]

    # Step 6: Normalize the rows of the spectral features matrix
    X_spectral = X_spectral / np.linalg.norm(X_spectral, axis=1, keepdims=True)

    # Step 7: Apply K-means clustering
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X_spectral)

    return labels

# Example Usage
if __name__ == "__main__":
    # Generate sample data (2D points)
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [-2, -2]])

    # Perform spectral clustering
    sigma = 1.0  # Gaussian kernel width
    k = 2  # Number of clusters
    labels = spectral_clustering(X, k, sigma)

    print("Cluster labels:", labels)
