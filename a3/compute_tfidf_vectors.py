import json
import numpy

# read all files (dictionary, postings, structured documents)
dictionary_json = open('./parse_results/dictionary.json', 'r')
Dictionary = json.load(dictionary_json)
dictionary_json.close()

postings_json = open('./parse_results/postings.json', 'r')
Postings = json.load(postings_json)
postings_json.close()

documents_structured_json = open('./parse_results/documents_structured.json', 'r')
documents_structured = json.load(documents_structured_json)
documents_structured_json.close()

doc_tfidf_vectors = {}
number_of_docs = len(documents_structured)

for doc_id in documents_structured:
  doc_tfidf_vector = []
  for term in Dictionary:
    if not doc_id in Postings[term]:
      doc_tfidf_vector.append(0)
      continue
    
    term_idf = numpy.log10(number_of_docs / Dictionary[term])

    term_weight = (1 + numpy.log10(Postings[term][doc_id]["tf"])) * term_idf

    doc_tfidf_vector.append(term_weight)

  doc_tfidf_vectors[doc_id] = doc_tfidf_vector

doc_tfidf_vectors_file = open('./doc_tfidf_vectors.json', 'w')
json.dump(doc_tfidf_vectors, doc_tfidf_vectors_file)
doc_tfidf_vectors_file.close()


