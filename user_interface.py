
import json
import time
import statistics
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

from search import search

# read structured documents
documents_structured_json = open('./cacm_structured.json', 'r')
documents_structured = json.load(documents_structured_json)
documents_structured_json.close()

# query user for stopword removal and stemming 
stopwords_removal_enabled = user_boolean_selection("Was stopwords removal enabled when indexing? (y/n) ", "Stopwords removal was enabled", "Stopwords removal was disabled")
stemming_enabled = user_boolean_selection("Was stemming enabled when indexing? (y/n) ", "Stemming was enabled", "Stemming was disabled")

# search query
user_query = input("Please input a query to search for: ")
# re-query until ZZEND is not the input
while(user_query != 'ZZEND'):
    rel = search(user_query, stopwords_removal_enabled, stemming_enabled)
    counter = 1
    for doc_id in rel:
      print(str(counter) + ".")
      if "title" in documents_structured[str(doc_id)]:
        print("Document Title: ", documents_structured[str(doc_id)]["title"])
      else:
        print("Document Title: ", "no info")
      
      if "authors" in documents_structured[str(doc_id)]:
        print("Document Author(s): ", documents_structured[str(doc_id)]["authors"])
      else:
        print("Document Authors: ", "no info")

      counter += 1
    user_query = input("Please input a query to search for: ")

# The role of information retrieval in knowledge based systems (i.e., expert systems).