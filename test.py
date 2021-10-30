import json
import time

import aux.collection_fields as collection_fields
from aux.field_parsing import *
from aux.term_parsing import *
from aux.helper_fns import *
from aux.search_context import *

from nltk.stem import PorterStemmer


# read all files (dictionary, postings, structured documents)
dictionary_json = open('./dictionary.json', 'r')
Dictionary = json.load(dictionary_json)
dictionary_json.close()

postings_json = open('./postings.json', 'r')
Postings = json.load(postings_json)
postings_json.close()

documents_structured_json = open('./cacm_structured.json', 'r')
documents_structured = json.load(documents_structured_json)
documents_structured_json.close()

# query user for stopword removal and stemming 
stopwords_removal_enabled = user_boolean_selection("Was stopwords removal enabled when indexing? (y/n) ", "Stopwords removal was enabled", "Stopwords removal was disabled")
stemming_enabled = user_boolean_selection("Was stemming enabled when indexing? (y/n) ", "Stemming was enabled", "Stemming was disabled")

# search times list
search_times = []
# query for the term to search for
user_query = input("Please input a term to search for: ")
# re-query for a term to search for until ZZEND is not the input
while(user_query != 'ZZEND'):
    # save the start time of processing query
    start_time = time.time()
    # get the stemmed term from the query
    term = extract_terms(user_query, stemming_enabled)[0]
    # check if term is in dictionary
    if term in Dictionary:
        print("----------------------")
        print("Extracted term:", term)
        print("Document Frequency:", Dictionary[term])
        # iterate over each document the term is posted in
        for doc_id in Postings[term]:
            print("*")
            print("Doc ID:", doc_id)

            if 'title' in documents_structured[doc_id]:
                print("Doc Title:", documents_structured[doc_id]['title'])
            else:
                print("Doc Title: no title for this document")

            print("Term Frequency:", Postings[term][doc_id]['tf'])

            print("Term Position(s):", Postings[term][doc_id]['pos'])

            # construct the combined summary source from title and abstract
            title = documents_structured[doc_id]['title'] if 'title' in documents_structured[doc_id] else ''
            abstract = documents_structured[doc_id]['abstract'] if 'abstract' in documents_structured[doc_id] else ''
            summary_source = title + '; ' + abstract

            # retrieve summary
            print("Summary: ..." + search_context(summary_source, Postings[term][doc_id]['pos'][0], stopwords_removal_enabled, stemming_enabled) + "...")
    else: 
        print("Term not in dictionary")    
    print("----------------------")
    search_time = time.time()-start_time
    search_times.append(search_time)
    print("Search time was: %s seconds" % (search_time))
    print("----------------------")
    user_query = input("Please input a term to search for: ")

acc = 0
for search_time in search_times:
    acc += search_time
avg_search_time = acc / len(search_times)
print("Average search time was:", str(avg_search_time))



