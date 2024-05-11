import numpy as np
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data):
        num_images, num_pixels = data.shape
        self.mean = np.mean(data, axis=0)  # mean
        centered_data = data - self.mean  # z
        cov_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.components = self.eigenvectors[:, :self.n_components]

    def compress(self, data):
        centered_data = data - self.mean
        compressed_data = np.dot(centered_data, self.components)
        return compressed_data

    def decompress(self, compressed_data):
        decompressed_data = np.dot(compressed_data, self.components.T) + self.mean
        return decompressed_data