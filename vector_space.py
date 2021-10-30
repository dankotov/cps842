import numpy as np


class VectorSpaceModel:
    def __init__(self, documents, query_frequency, N=None, raw_term_frequency=None, document_numbering=None):
        self.documents = np.array(documents)
        self.query_frequency = query_frequency
        self._set_raw_term_frequency(raw_term_frequency)
        self._set_n(N)
        self._set_document_numbering(document_numbering)

        self._find_similarity()

    def get_similarity_list(self):
        self.similarity_list = np.stack((
            self.document_numbering, self.similarity), axis=1)
        return self.similarity_list.tolist()

    def get_sorted_similarity_list(self, sort_by=1):
        unsorted_similarity = self.get_similarity_list()
        return sorted(unsorted_similarity, key=lambda l: l[sort_by], reverse=True)

    def get_similarity_dictionary(self):
        lst = self.get_similarity_list()
        dict = {}
        for value_pair in lst:
            dict[value_pair[0]] = value_pair[1]
        return dict

    def get_sorted_similarity_dictionary(self, sort_by=1):
        lst = self.get_sorted_similarity_list(sort_by=sort_by)
        dict = {}
        for value_pair in lst:
            dict[value_pair[0]] = value_pair[1]
        return dict

    def _set_document_numbering(self, document_numbering):
        documents_number = np.shape(self.documents)[0]
        if document_numbering is None:
            self.document_numbering = np.arange(documents_number)
        else:
            assert(np.shape(document_numbering)[0] == documents_number)
            self.document_numbering = document_numbering

    def _set_raw_term_frequency(self, raw_term_frequency=None):
        if raw_term_frequency is None:
            self.raw_term_frequency = np.sum(self.documents, axis=0)
        else:
            self.raw_term_frequency = raw_term_frequency

    def _set_n(self, N):
        self.n = np.shape(self.documents)[0] if N is None else N

    def _find_similarity(self):
        self._compute_idf()
        self._compute_document_tf()
        self._compute_document_w()
        self._compute_document_magnitude()

        self._compute_query_tf()
        self._compute_query_w()
        self._compute_query_magnitude()

        self._compute_similarity()

    def _compute_idf(self):
        self.idf = np.log10(self.n / self.raw_term_frequency)

    def _compute_document_tf(self):
        self.d_tf = np.where((self.documents > 0),
                             1 + np.log10(self.documents), self.documents)

    def _compute_document_w(self):
        self.d_w = self.d_tf * self.idf

    def _compute_document_magnitude(self):
        self.d_magnitude = np.linalg.norm(self.d_w, axis=1)

    def _compute_query_tf(self):
        self.q_tf = self.query_frequency

    def _compute_query_w(self):
        self.q_w = self.q_tf * self.idf

    def _compute_query_magnitude(self):
        self.q_magnitude = np.linalg.norm(self.q_w)

    def _compute_similarity(self):
        self.similarity = (self.d_w @ self.q_w) / \
            (self.d_magnitude * self.q_magnitude)


if __name__ == "__main__":
    document3f = np.array([4, 2, 1])
    document5f = np.array([3, 4, 2])
    document6f = np.array([1, 2, 3])

    documents = [document3f, document5f, document6f]
    qf = np.array([1, 0, 1])
    N = 10000
    # document_numbering will use a custom numbering that you provide for the document. For example, lets say you have document1, document3, document5
    # if you do not provide numbering, the final answer will look like [0: sim1, 1:sim3, 2:sim5]. If you provide numbering np.array([3,5,6])
    # then the final result will look like [3: sim1, 5:sim3, 6:sim5]. Provide numbering in order of the documents.
    vsm = VectorSpaceModel(
        documents, qf, N=N, raw_term_frequency=np.array([5000, 1000, 500]), document_numbering=np.array([3, 5, 6]))
    print("Unsorted similarity: ", vsm.get_similarity_list())
    print("Unsorted dictionary: ", vsm.get_similarity_dictionary())
    print("Sorted similarity: ", vsm.get_sorted_similarity_list())
    print("Sorted dictionary: ", vsm.get_sorted_similarity_dictionary(sort_by=1))
