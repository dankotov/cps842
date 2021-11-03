import json
import collections
from nltk.stem import PorterStemmer
from collections import OrderedDict
import auxiliary.collection_fields as collection_fields
from auxiliary.field_parsing import *
from auxiliary.term_parsing import *
from auxiliary.helper_fns import *
from search import search


class Eval:
    def __init__(self, queries_file='./cacm/query.text', query_results_file='./cacm/qrels.text', enable_stemming=True, enable_stop_word_removal=True):
        # Queries
        self.queries = {}
        # Relevant documents
        self.query_results = collections.defaultdict(list)
        # Results from search
        self.search_results = {}
        self.enable_stemming = enable_stemming
        self.enable_stop_word_removal = enable_stop_word_removal

        self._read_queries(queries_file)
        self._read_query_results(query_results_file)

        self._get_query_search_results()

        self._calculate_mean_average_precision()
        self._calculate_mean_r_precision()

    def get_map(self):
        return self.mean_average_precision

    def get_mrprecision(self):
        return self.mean_r_precision

    def _read_queries(self, queries_file):
        queries_collection = open(queries_file, 'r')
        self.__read_sections_from_document(queries_collection)

    def __read_sections_from_document(self, queries_collection):
        collection_line = queries_collection.readline()
        while(collection_line):
            collection_line = collection_line.rstrip('\n')
            if is_a_doc_id_identifier(collection_line):
                query_id = int(collection_line.split(' ')[1])
                self.queries[query_id] = {}
                collection_line = queries_collection.readline()
            elif collection_line == collection_fields.ABSTRACT:
                abstract, collection_line = parse_field(queries_collection)
                self.queries[query_id] = abstract
            else:
                collection_line = queries_collection.readline()
        if 0 in self.queries.keys():
            del self.queries[0]

    # depreciated
    def __process_queries(self):
        for key in self.queries:
            self.queries[key] = extract_terms(
                self.queries[key], self.enable_stemming)
            if self.enable_stop_word_removal:
                self.queries[key] = remove_stopwords(self.queries[key])

    def _read_query_results(self, query_results_file):
        query_results_collection = open(query_results_file, 'r')
        collection_line = query_results_collection.readline()
        while(collection_line):
            split_line = (collection_line.rstrip('\n')).split()
            self.query_results[int(split_line[0])].append(int(split_line[1]))
            collection_line = query_results_collection.readline()

    def _get_query_search_results(self):
        for key in self.queries.keys():
            results = search(
                self.queries[key], self.enable_stemming, self.enable_stop_word_removal)
            results = list(results.keys())
            self.search_results[key] = [int(i) for i in results]

    def _calculate_average_precision(self, retrieved_documents, relevant_documents):
        running_sum = 0
        elements = 0
        for i in range(1, len(retrieved_documents)+1):
            if retrieved_documents[i-1] in relevant_documents:
                elements += 1
                running_sum += elements/i
        if len(relevant_documents) > 0:
            return running_sum/(len(relevant_documents))
        else:
            return None

    # this function complutes MAP across all queries from the files
    def _calculate_mean_average_precision(self):
        running_precision_sum = 0
        total_queries = len(self.queries.keys())

        for key in self.queries.keys():
            precision = self._calculate_average_precision(
                self.search_results[key], self.query_results[key])
            if precision is not None:
                running_precision_sum += precision
            else:
                total_queries-1

        if (total_queries > 0):
            self.mean_average_precision = running_precision_sum / total_queries
        else:
            self.mean_average_precision = None

    def _calculate_r_precision(self, retrieved_documents, relevant_documents):
        relevant_count = 0
        r_d_set = set(relevant_documents)
        iter_length = min(len(relevant_documents)+1,
                          len(retrieved_documents)+1)

        for i in range(1, iter_length):
            if retrieved_documents[i-1] in r_d_set:
                relevant_count += 1

        if len(relevant_documents) > 0:
            return relevant_count / len(relevant_documents)
        else:
            return None

    # this function complutes MAP across all queries from the files
    def _calculate_mean_r_precision(self):
        running_r_precision_sum = 0
        total_queries = len(self.queries.keys())

        for key in self.queries.keys():
            rprecision = self._calculate_r_precision(
                self.search_results[key], self.query_results[key])
            if rprecision is not None:
                running_r_precision_sum += rprecision
            else:
                total_queries - 1

        if (total_queries > 0):
            self.mean_r_precision = running_r_precision_sum / total_queries
        else:
            self.mean_r_precision = None


if __name__ == "__main__":
    e = Eval()
    print("MAP: ", e.get_map())
    print("R-Precision: ", e.get_mrprecision())
