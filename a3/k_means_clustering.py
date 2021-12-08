from typing import List
import numpy as np
import random
import json
import time


class KMeansClustering:
    def __init__(self, documents: List, num_clusters: int, iterations: int, tries: int = 10) -> None:
        self.iterations = iterations
        self.tries = tries
        self.documents = documents
        self.num_clusters = num_clusters

        self.centroids_per_try = []
        self.tightness_per_try = []

        self._init_centroids()

    def cluster(self) -> None:
        for try_number in range(self.tries):
            print('Try number: ', try_number)
            self._init_centroids()
            start_time = time.time()
            self._single_pass()
            print('Try took', time.time()-start_time, 'seconds')
            self._compute_tightness_per_cluster()
            print('Tightness in this try:', self.tightness_per_try[try_number])
        print('Done. \nRecomputing best result')
        self._compute_best_result()
        print('Done')

    def get_cluster_matrix(self) -> np.ndarray:
        return self.clusters

    def _compute_best_result(self):
        best_tightness_index = np.argmin(self.tightness_per_try)
        self.centroids = self.centroids_per_try[best_tightness_index]
        self._single_pass()

    def _compute_tightness_per_cluster(self):
        tightness = []
        for centroid_num in range(self.num_clusters):
            documents_in_cluster = self.documents[self.clusters == centroid_num]
            distances = np.linalg.norm(
                self.centroids[centroid_num, :] - documents_in_cluster, axis=1)
            tightness.append(np.sum(distances))
        self.tightness_per_try.append(sum(tightness))

    def _single_pass(self) -> None:
        for iteration_num in range(self.iterations):
            if (self.prev_centroids is not None and np.array_equal(self.prev_centroids, self.centroids)):
                print('\tCentroids did not change, aborting')
                break
            print('\tIteration ', iteration_num)
            self._iteration()

    def _iteration(self) -> None:
        similarity = self._compute_similarity_to_centroids()
        similarity = self._get_clusters_from_similarity(similarity)
        self.clusters = similarity
        self._recalculate_centroids()

    # TODO: randomly select clusters from documents
    def _init_centroids(self) -> None:
        self._init_centroids_randomly()
        self.centroids_per_try.append(self.centroids)

    def _init_centroids_randomly(self) -> None:
        chosen_centroid_nums = []
        chosen_centroids = []

        while len(chosen_centroids) < self.num_clusters:
            centroid_num = random.randint(0, self.documents.shape[0]-1)
            if centroid_num not in chosen_centroid_nums:
                chosen_centroids.append(self.documents[centroid_num, :])
                chosen_centroid_nums.append(centroid_num)
        self.centroids = np.array(chosen_centroids)
        self.prev_centroids = None

    def _init_centroids_choose_first_n(self) -> None:
        chosen_centroids = []

        for centroid_num in range(self.num_clusters):
            centroid_num = centroid_num % self.documents.shape[0]
            chosen_centroids.append(self.documents[centroid_num, :])
        self.centroids = np.array(chosen_centroids)
        self.prev_centroids = None

    def _compute_similarity_to_centroids(self) -> np.ndarray:
        return self._compute_class_cosine_similarity()

    def _compute_class_cosine_similarity(self) -> np.ndarray:
        return self.documents @ self.centroids.T

    def _get_clusters_from_similarity(self, similarity: np.ndarray) -> np.ndarray:
        return np.argmax(similarity, axis=1)

    def _recalculate_centroids(self) -> None:
        new_centroids = []
        for centroid_num in range(self.num_clusters):
            documents_in_cluster = self.documents[self.clusters == centroid_num]
            centroid = np.sum(documents_in_cluster, axis=0) / \
                documents_in_cluster.shape[0]
            new_centroids.append(centroid)
        self.prev_centroids = self.centroids
        self.centroids = np.array(new_centroids)


def get_tfidf_vectors(source='doc_tfidf_vectors.json'):
    source_file = open(source, 'r')
    source_dict = json.load(source_file)
    source_file.close()
    dict_values = list(source_dict.values())
    dict_ids = list(source_dict.keys())
    return (dict_ids, np.array(dict_values))


def match_id_to_document(ids, similarities):
    clusters = {}
    for d_id, d_sim in zip(ids, similarities):
        clusters[d_id] = d_sim
    return clusters


if __name__ == '__main__':
    print('Loading values')
    start_time = time.time()
    (dict_ids, documents) = get_tfidf_vectors()
    end_time = time.time() - start_time
    print('Done loading values. Operation took', end_time, 'seconds')
    print('Computing clusters')
    start_time = time.time()

    k_means = KMeansClustering(
        documents, num_clusters=5, iterations=50, tries=20)
    k_means.cluster()

    end_time = time.time() - start_time
    print('Done computing clusters. Operation took', end_time, 'seconds')

    clusters = match_id_to_document(
        dict_ids, k_means.get_cluster_matrix().tolist())

    for key in list(clusters.keys())[:20:]:
        print('Document', key, ' cluster: ', clusters[key])
