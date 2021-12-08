from typing import List
import numpy as np


class KMeansClustering:
    def __init__(self, documents: List, numClusters: int, numIterations: int, numTries: int = 1) -> None:
        self.numIterations = numIterations
        self.numTries = numTries
        self.documents = documents

        self.clusters = [[] for i in range(numClusters)]

        self._init_centroids()
        print(self.documents)
        print(self.centroids)

    def cluster(self) -> None:
        similarity = self._compute_similarity_to_centroids()
        similarity = self._convert_similarity_matrix_to_cluster_num(similarity)
        self._recalculate_centroids(similarity)

    # TODO: randomly select clusters from documents
    def _init_centroids(self) -> None:
        self.centroids = np.array([self.documents[0, :], self.documents[1, :]])

    def _compute_similarity_to_centroids(self) -> np.ndarray:
        return self._compute_class_cosine_similarity()

    def _compute_class_cosine_similarity(self) -> np.ndarray:
        return self.documents @ self.centroids.T

    def _convert_similarity_matrix_to_cluster_num(self, similarity) -> np.ndarray:
        return np.argmax(similarity, axis=1)

    def _recalculate_centroids(self, similarity) -> None:
        pass


if __name__ == '__main__':
    d1 = np.array([0.0, 0.9, 0.4])
    d2 = np.array([0.8, 0.3, 0.5])
    d3 = np.array([1.0, 0.0, 0.0])
    d4 = np.array([0.0, 1.0, 0.0])
    d5 = np.array([0.7, 0.4, 0.6])
    documents = np.array([d1, d2, d3, d4, d5])

    k_means = KMeansClustering(documents, 2, 3, 1)
    k_means.cluster()
