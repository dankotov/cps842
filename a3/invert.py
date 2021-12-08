import json
import os
from nltk.stem import PorterStemmer
from collections import OrderedDict
import auxiliary.collection_fields as collection_fields
from auxiliary.field_parsing import *
from auxiliary.term_parsing import *
from auxiliary.helper_fns import *

# the dictionary - a python dictionary (hash map in fact)
Dictionary = {}
# the postings - a python dictionary (hash map in fact)
Postings = {}
# a structured document containing info of interest about each doc (unprocessed): title, abstract, date of pub, authors 
documents_structured = {}
# a flag dictionary that will help identify if a term was already accounted for from this document
term_doc_id_flags = {}

# query for stopwords and stemming selection
stopwords_removal_enabled = enable_stop_word_removal("Do you want to enable the stopwords removal? (y/n) ", "Stopwords removal enabled", "Stopwords removal disabled")
stemming_enabled = enable_porter_stemmer("Do you want to enable stemming? (y/n) ", "Stemming enabled", "Stemming disabled")

# get current working directory
cwd = os.getcwd()
# set path for the soucre collection
source_collection_path = os.path.join(cwd, "bbc")

# iterate over all files in the source collection
for filename in os.listdir(source_collection_path):
    f = os.path.join(source_collection_path, filename)
    #  if the current file is a directory
    if os.path.isdir(f):
        # loop over all files in that directory
        for docname in os.listdir(f):
            d = os.path.join(f, docname)
            #  if current file is a text file
            if os.path.isfile(d) and d.endswith(".txt"):
                # create doc id from first 3 chars of class name and the number id of the doc
                doc_id = filename[0 : 3] + docname.split(".txt")[0]
                doc_text = ""
                # read doc lines
                curr_doc = open(d, 'r')
                curr_doc_lines = curr_doc.readlines()
                curr_doc.close()
                # create a document entry (doc_id:title) in the docs structured file
                documents_structured[doc_id] = curr_doc_lines[0].rstrip("\n")
                # concatenate all lines into one
                for line in curr_doc_lines:
                    doc_text += line.replace("\n", " ")

                # process terms
                process_terms(text=doc_text,
                              doc_id=doc_id,
                              target_dict=Dictionary,
                              target_post=Postings,
                              term_doc_id_flags=term_doc_id_flags,
                              stopwords_option=stopwords_removal_enabled,
                              stemming_option=stemming_enabled)
                
# write the dictionary, postings and structured documents into memory 
dictionary_file = open('./parse_results/dictionary.json', 'w')
json.dump(OrderedDict(sorted(Dictionary.items())), dictionary_file)
dictionary_file.close()

postings_file = open('./parse_results/postings.json', 'w')
json.dump(OrderedDict(sorted(Postings.items())), postings_file)
postings_file.close() 

documents_structured_file = open('./parse_results/documents_structured.json', 'w')
json.dump(OrderedDict(sorted(documents_structured.items())), documents_structured_file)
documents_structured_file.close() 