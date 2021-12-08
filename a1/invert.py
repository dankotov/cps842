import json
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
stopwords_removal_enabled = user_boolean_selection("Do you want to enable the stopwords removal? (y/n) ", "Stopwords removal enabled", "Stopwords removal disabled")
stemming_enabled = user_boolean_selection("Do you want to enable stemming? (y/n) ", "Stemming enabled", "Stemming disabled")

# open collection file for reading
source_collection = open('./cacm.all', 'r')
# read first collection line
collection_line = source_collection.readline()
    
    
# consecutively read and parse collection lines
while(collection_line):
    # strip line of the newline ending
    collection_line = collection_line.rstrip('\n')
    
    # if this line is an identifier for the DOCID in the collection
    if is_a_doc_id_identifier(collection_line):
        # get the doc id
        doc_id = int(collection_line.split(' ')[1])
        # create a corresponding entry for this doc in the document_structured dictionary
        documents_structured[doc_id] = {}

        collection_line = source_collection.readline()

    # if the next line is an identifier for a title field 
    elif collection_line == collection_fields.TITLE:
        # parse the field to get the title and next line reference
        title, collection_line = parse_field(source_collection)

        # if title is not empty
        if title != '':
            # process title for terms
            process_terms(title, doc_id, Dictionary, Postings, term_doc_id_flags, stopwords_removal_enabled, stemming_enabled)
            # insert title attribute to the appropriate document id in the documents structutred dictionary
            documents_structured[doc_id]['title'] = title
    
    # same, as previous elif, but for abstract 
    elif collection_line == collection_fields.ABSTRACT:
        abstract, collection_line = parse_field(source_collection)
        if abstract != '':
            title_len = len(documents_structured[doc_id]['title'].split(' ')) if 'title' in documents_structured[doc_id] else 0
            process_terms(abstract, doc_id, Dictionary, Postings, term_doc_id_flags, stopwords_removal_enabled, stemming_enabled, title_len)
        documents_structured[doc_id]['abstract'] = abstract
    
    # same, as previous elif, but for publication date
    elif collection_line == collection_fields.PUBLICATION_DATE:
        publication_date, collection_line = parse_field(source_collection)
        documents_structured[doc_id]['pub_date'] = publication_date

    # same, as previous elif, but for authors
    elif collection_line == collection_fields.AUTHORS:
        authors, collection_line = parse_field(source_collection)
        documents_structured[doc_id]['authors'] = authors

    # if the line is not a field of interest, move on to next line
    else:
        collection_line = source_collection.readline()

source_collection.close()

# write the dictionary, postings and structured documents into memory 
dictionary_file = open('dictionary.json', 'w')
json.dump(OrderedDict(sorted(Dictionary.items())), dictionary_file)
dictionary_file.close()

postings_file = open('postings.json', 'w')
json.dump(OrderedDict(sorted(Postings.items())), postings_file)
postings_file.close()

documents_structured_file = open('cacm_structured.json', 'w')
json.dump(documents_structured, documents_structured_file)
documents_structured_file.close()