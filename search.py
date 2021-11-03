
import json
import math

from numpy.ma.core import count

import auxiliary.collection_fields as collection_fields
from auxiliary.field_parsing import *
from auxiliary.term_parsing import *
from auxiliary.helper_fns import *
from auxiliary.search_context import *
from vector_space import VectorSpaceModel

from nltk.stem import PorterStemmer
import numpy as np

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

def search(user_query, stemming_enabled, stopwords_removal_enabled):
  # remove the stopwords from the query and stem terms, if booleans are set
  user_query = extract_terms(user_query, stemming_enabled)
  if stopwords_removal_enabled:
    user_query = remove_stopwords(user_query)

  query_terms = eliminate_index(1, user_query)

  query_terms_dfs = [Dictionary[term] for term in query_terms]
  
  query_tfs = {}
  relevant_docs = set()
  for term in query_terms:
    if term in query_tfs:
      query_tfs[term] = query_tfs[term] + 1
    else:
      query_tfs[term] = 1

    term_idf = compute_idf(N=3203, df=Dictionary[term])
    
    doc_term_weights = {}
    for doc_id in Postings[term]:
      term_weight = compute_weight(Postings[term][doc_id]['tf'], term_idf)
      doc_term_weights[doc_id] = term_weight
    
    weights = list(doc_term_weights.values())
    median_weight = np.median(weights)

    champion_docs = set()

    if len(weights) > 10 and len(set(weights)) >= 2:
      for doc in doc_term_weights:
        if doc_term_weights[doc] > median_weight:
          champion_docs.add(doc)
    else:
      for doc in doc_term_weights:
        champion_docs.add(doc)

    relevant_docs.update(champion_docs)
  
  rel_docs_tfs = {}
  for doc_id in relevant_docs:
    doc_tfs = []
    for term in query_terms:
      if doc_id in Postings[term]:
        doc_tfs.append(Postings[term][doc_id]['tf'])
      else:
        doc_tfs.append(0)
    rel_docs_tfs[doc_id] = doc_tfs

  

  vsm = VectorSpaceModel(
      documents=list(rel_docs_tfs.values()),
      document_numbering=np.array(list(rel_docs_tfs.keys())),
      query_frequency=np.array(list(query_tfs.values())),
      raw_term_frequency=np.array(query_terms_dfs),
      N=3203
    )
  # vsm = VectorSpaceModel(
  #     documents=list(rel_docs_tfs.values()),
  #     document_numbering=np.array(list(rel_docs_tfs.keys())),
  #     query_frequency=np.array(list(query_tfs.values())),
  #   )
  rel = vsm.get_sorted_similarity_dictionary(sort_by=1)
  return rel

def compute_weight(tf, idf):
  return tf * idf

def compute_idf(N, df):
  return np.log10(N / df)

def eliminate_index(idf_threshold, query):
  query_terms = {}
  for term in query:
    if term in Dictionary and compute_idf(N=3203, df=Dictionary[term]) > idf_threshold:
      # print(term, compute_idf(N=3203, df=Dictionary[term]), Dictionary[term])
      if term in query_terms:
        query_terms[term] = query_terms[term] + 1
      else:
        query_terms[term] = 1

  return query_terms


def search_old(user_query, stemming_enabled, stopwords_removal_enabled):
    # remove the stopwords from the query and stem terms, if booleans are set
    user_query = extract_terms(user_query, stemming_enabled)
    if stopwords_removal_enabled:
      user_query = remove_stopwords(user_query)
    # check if term is in dictionary
    search_order = []
    query_tfs = {}
    for term in user_query:
      if term in Dictionary:
        if term in query_tfs:
          query_tfs[term] = query_tfs[term] + 1
        else:
          query_tfs[term] = 1
          search_order.append({'term': term, 'df': Dictionary[term], 'idf': np.log10(3203 / Dictionary[term])})
    relevant_docs = {}
    for term in search_order:
      for doc_id in Postings[term['term']]:
        doc_relevant = True
        terms_not_found_in_doc = 0
        doc_tfs = [Postings[term['term']][doc_id]["tf"]]
        for idx in range(1, len(search_order)):
          if doc_id not in Postings[search_order[idx]['term']]:
            terms_not_found_in_doc += 1
            doc_tfs.append(0)
          else:
            doc_tfs.append(Postings[search_order[idx]['term']][doc_id]["tf"])
          if terms_not_found_in_doc >= math.ceil(len(search_order) / 2) + 1:
            doc_relevant = False
            break
        if doc_relevant and int(doc_id) not in relevant_docs:
          relevant_docs[int(doc_id)] = np.array(doc_tfs)

    print(len(relevant_docs.keys()))
    vsm = VectorSpaceModel(
      documents=list(relevant_docs.values()),
      query_frequency=np.array(list(query_tfs.values())),
      document_numbering=np.array(list(relevant_docs.keys()))
    )
    rel = vsm.get_sorted_similarity_dictionary(sort_by=1)
    return [int(doc_id) for doc_id in rel]
