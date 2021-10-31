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
        self.queries = {}
        self.query_results = collections.defaultdict(list)
        self.enable_stemming = enable_stemming
        self.enable_stop_word_removal = enable_stop_word_removal

        self._read_queries(queries_file)
        self._read_query_results(query_results_file)

        for i in range(1, 10):
            print(i, '  :  ', self.queries[i], self.query_results[i])

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


if __name__ == "__main__":
    e = Eval()
