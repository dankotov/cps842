from typing import List
import numpy as np
import random
import json
import time
import os
import matplotlib.pyplot as plt

from numpy.core.fromnumeric import put


class KMeansClustering:
    def __init__(self, documents: List, num_clusters: int, iterations: int, tries: int = 10) -> None:
        self.iterations = iterations
        self.tries = tries
        self.documents = documents
        self.num_clusters = num_clusters

        self.centroids_per_try = []
        self.tightness_per_try = []
        self.tightness_per_try_per_cluster = []

        self._init_centroids()

    def cluster(self) -> None:
        for try_number in range(self.tries):
            print('Try number: ', try_number)
            self._init_centroids()
            start_time = time.time()
            self._single_pass()
            print('Try took', time.time()-start_time, 'seconds')
            self._compute_tightness_per_cluster()
            print('Tightness in this try:',
                  self.tightness_per_try_per_cluster[try_number])
        print('Done. \nRecomputing best result')
        self._compute_best_result()
        print('Done')

    def get_number_of_documents(self):
        return len(self.documents)

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
        self.tightness_per_try_per_cluster.append(tightness)

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


# def match_id_to_document(ids, similarities):
#     clusters = {}
#     for d_id, d_sim in zip(ids, similarities):
#         clusters[d_id] = d_sim
#     return clusters

def match_id_to_document(ids, similarities, documents):
    id_to_title_json = open('./parse_results/documents_structured.json', 'r')
    id_to_title = json.load(id_to_title_json)
    id_to_title_json.close()

    id_to_class = {
        "bus": "business",
        "ent": "entertainment",
        "pol": "politics",
        "spo": "sport",
        "tec": "tech"
    }

    clusters = {}

    cluster_combined_tfidf = {}

    for d_id, d_sim, document_tfidf in zip(ids, similarities, documents):
        if not d_sim in clusters:
            clusters[d_sim] = {}

        if not d_sim in cluster_combined_tfidf:
            cluster_combined_tfidf[d_sim] = document_tfidf
        else:
            cluster_combined_tfidf[d_sim] = cluster_combined_tfidf[d_sim] + document_tfidf

        if not "docs" in clusters[d_sim]:
            clusters[d_sim]["docs"] = {}

        if not "class_summary" in clusters[d_sim]:
            clusters[d_sim]["class_summary"] = {}
            for id in id_to_class:
                clusters[d_sim]["class_summary"][id_to_class[id]] = 0

        clusters[d_sim]["docs"][d_id] = {}
        doc_class = id_to_class[d_id[0: 3]]
        clusters[d_sim]["docs"][d_id]['class'] = doc_class
        clusters[d_sim]["docs"][d_id]['title'] = id_to_title[d_id]

        clusters[d_sim]["class_summary"][doc_class] = clusters[d_sim]["class_summary"][doc_class] + 1
    

    dictionary_json = open('./parse_results/dictionary.json', 'r')
    Dictionary = json.load(dictionary_json)
    dictionary_json.close()

    for cluster in clusters:
        tfidf_vector = cluster_combined_tfidf[cluster]
        tmp = np.argpartition(-tfidf_vector, 5)
        highest_tfidfs_pos = tmp[:5]
        clusters[cluster]["highest_idfs"] = []
        for tfidf_pos in highest_tfidfs_pos:
            clusters[cluster]["highest_idfs"].append(list(Dictionary.keys())[tfidf_pos])


    return clusters


def evaluate_purity(clusters, n_of_docs):
    sum_majorities = 0
    for cluster in clusters:
        class_summary = clusters[cluster]["class_summary"]
        classes = list(class_summary.keys())
        members_n = list(class_summary.values())
        majority_class = max(class_summary, key=class_summary.get)
        majority_members_number = class_summary[majority_class]
        print(
            f"\tMajority class for cluster {cluster} is '{majority_class}' with {majority_members_number} members")
        sum_majorities += majority_members_number

        fig = plt.figure(figsize=(10, 5))
        plt.bar(classes, members_n)
        plt.xlabel("Document Classes")
        plt.ylabel("No. of members")
        plt.title(f"Document distribution by class for cluster{cluster}")
        os.makedirs(os.path.dirname(
            f"./clustering_results/visual/"), exist_ok=True)
        plt.savefig(f'./clustering_results/visual/cluster{cluster}.png')

    purity = (1 / n_of_docs) * sum_majorities

    return purity


if __name__ == '__main__':
    print('Loading values')
    start_time = time.time()
    (dict_ids, documents) = get_tfidf_vectors()
    end_time = time.time() - start_time
    print('Done loading values. Operation took', end_time, 'seconds')
    user_k_val = input('Please provide the k value: ')
    while(not user_k_val.isdigit()):
        print('Please provide a valid (integer) k value')
        user_k_val = input('Please provide the k value: ')

    print('Computing', user_k_val, 'clusters')
    start_time = time.time()

    k_means = KMeansClustering(
        documents, num_clusters=int(user_k_val), iterations=50, tries=20)
    k_means.cluster()

    end_time = time.time() - start_time
    print('Done computing clusters. Operation took', end_time, 'seconds')

    clusters = match_id_to_document(
        dict_ids, k_means.get_cluster_matrix().tolist(), documents)

    # print(clusters)
    os.makedirs(os.path.dirname(
        "./clustering_results/clusters.json"), exist_ok=True)
    with open("./clustering_results/clusters.json", "w") as clusters_file:
        json.dump(clusters, clusters_file)

    print("Wrote clustered results to ./clustering_results/clusters.json")

    print("\nEvaluating purity")
    start_time = time.time()

    purity = evaluate_purity(clusters, k_means.get_number_of_documents())

    end_time = time.time() - start_time
    print('Done evaluating purity. Operation took', end_time, 'seconds')
    print(f'Purity is {purity}')

    print('Wrote visual representation of clustering results to ./clustering_results/visual/')

    # for key in list(clusters.keys())[:20:]:
    #     print('Document', key, ' cluster: ', clusters[key])
