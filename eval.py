import json
import collections
from nltk.stem import PorterStemmer
from collections import OrderedDict
import auxiliary.collection_fields as collection_fields
from auxiliary.field_parsing import *
from auxiliary.term_parsing import *
from auxiliary.helper_fns import *


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

        # TODO: MAKE THIS FUNCTION WORK
        # this function sets self.search_results as a dictionary in a form:
        # {
        # querynumber1 : [documentid1, documentid2, documentid3],
        # querynumber2 : [documentid3, documentid2, documentid9, documentid12],
        # }
        # querynumbers and documentid's are both integers
        self._get_query_search_results()

        # Example (numbers from the slides)
        self.query_results = {
            1: [123, 9, 56, 25, 3, 1, 1, 1, 1, 1],
            2: [56, 129, 3]}
        self.search_results = {
            1: [123, 84, 56, 6, 8, 9, 511, 129, 187, 25, 38, 48, 250, 113, 3],
            2: [425, 87, 56, 32, 124, 615, 512, 129, 4, 130, 193, 715, 810, 5, 3]
        }
        self.queries = {
            1: ['hello'],
            2: ['world'],
        }

        self._calculate_mean_average_precision()
        print(self.mean_average_precision)
        self._calculate_mean_r_precision()
        print(self.mean_r_precision)

    def _read_queries(self, queries_file):
        queries_collection = open(queries_file, 'r')
        self.__read_sections_from_document(queries_collection)
        self.__process_queries()

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
        pass

    def _calculate_average_precision(self, retrieved_documents, relevant_documents):
        running_sum = 0
        elements = 0
        for i in range(1, len(retrieved_documents)+1):
            if retrieved_documents[i-1] in relevant_documents:
                elements += 1
                running_sum += elements/i
        return running_sum/(len(relevant_documents))

    # this function complutes MAP across all queries from the files
    def _calculate_mean_average_precision(self):
        running_precision_sum = 0
        total_queries = len(self.queries.keys())

        for key in self.queries.keys():
            running_precision_sum += self._calculate_average_precision(
                self.search_results[key], self.query_results[key])

        if (total_queries > 0):
            self.mean_average_precision = running_precision_sum / total_queries
        else:
            self.mean_average_precision = None

    def _calculate_r_precision(self, retrieved_documents, relevant_documents):
        relevant_count = 0
        r_d_set = set(relevant_documents)
        for i in range(1, len(relevant_documents)+1):
            if retrieved_documents[i-1] in r_d_set:
                relevant_count += 1

        if len(relevant_documents) > 0:
            return relevant_count / len(relevant_documents)

    # this function complutes MAP across all queries from the files
    def _calculate_mean_r_precision(self):
        running_r_precision_sum = 0
        total_queries = len(self.queries.keys())

        for key in self.queries.keys():
            running_r_precision_sum += self._calculate_r_precision(
                self.search_results[key], self.query_results[key])

        if (total_queries > 0):
            self.mean_r_precision = running_r_precision_sum / total_queries
        else:
            self.mean_r_precision = None


if __name__ == "__main__":
    e = Eval()
