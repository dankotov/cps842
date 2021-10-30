from nltk.stem import PorterStemmer

# get term list from the source text, stemmed if necessary
def extract_terms(text, stemming):
    # flatten text
    text = text.lower()

    chars_to_remove = ":;.,!?\|/][}{)(=%*@`~'\""

    for char in chars_to_remove:
        text = text.replace(char, '')

    # precaution to avoid double spaces present in CACM
    terms = text.replace('  ', ' ').split(' ')

    porter = PorterStemmer()

    # stem if necessary
    if stemming:
        terms_extracted = [porter.stem(term) for term in terms if term != '']
    else:
        terms_extracted = [term for term in terms if term != '']

    return terms_extracted

# remove stopwords in a list of terms
def remove_stopwords(terms):
    stopwords_file = open('stopwords.txt', 'r')
    stopwords = dict.fromkeys(stopwords_file.read().splitlines())
    stopwords_file.close()

    terms_filtered = [term for term in terms if term not in stopwords]

    return terms_filtered

# build the dictionary and postings 
def process_terms(text, doc_id, target_dict, target_post, term_doc_id_flags, stopwords_option, stemming_option, title_len=0):
    stopwords_file = open('stopwords.txt', 'r')
    stopwords = dict.fromkeys(stopwords_file.read().splitlines())
    stopwords_file.close()

    terms = extract_terms(text, stemming_option)
     
    pos = 0

    # iterate over all terms extracted from the source text
    for term in terms:
        # check if a word is a stopword
        if (stopwords_option) and (term in stopwords):
            # if it is, do not add it to neither postings nor dictionary, but increase position for next word
            pos +=1 
            continue
        
        # if a term is already in postings
        if term in target_post:
            # if it is, check if it was already accounted for from this document 
            if doc_id in target_post[term]:
                # if it was, increase term frequency
                target_post[term][doc_id]['tf'] += 1
                # caclulate the term position accounting for the title length
                term_pos = title_len + pos
                # append the new position to the according postings entry
                target_post[term][doc_id]['pos'].append(term_pos)
            # if this is a first occurence of this term in this document
            else:
                # create an according entry in postings for this term-document pair
                target_post[term][doc_id] = {}
                # set term frequency to 1
                target_post[term][doc_id]['tf'] = 1
                # calculate the term position accounting for the title length
                term_pos = title_len + pos
                # set the first occurence of this term in this document
                target_post[term][doc_id]['pos'] = [term_pos]
        # if a term is not in postings yet
        else:
            # create an entry for this term in postings
            target_post[term] = {}
            # create an entry for the document this term is mentioned in
            target_post[term][doc_id] = {}
            # set the term frequency for this term in this doc to 1
            target_post[term][doc_id]['tf'] = 1
            # calculate the term position accounting for the title length
            term_pos = title_len + pos
            # set the first occurence of this term in this document 
            target_post[term][doc_id]['pos'] = [term_pos]

        # if term is already in dictionary
        if term in target_dict:
            # if we the flag that this term has been accounted for from this document is not set
            # (we did not yet process this term in this document)
            if doc_id not in term_doc_id_flags[term]:
                # increase document frequency by 1
                target_dict[term] += 1
                # set the flag that this term has been accounted for from this document
                term_doc_id_flags[term].append(doc_id)
        # if term is not in dictionary yet
        else:
            # set document frequency for this term to 1
            target_dict[term] = 1
            # set the flag that this term has been accounted for from this document
            term_doc_id_flags[term] = [doc_id]

        pos += 1

# stem and normalize a single term
def normalize_term(term, stemming):
    term = term.lower()

    chars_to_remove = ":;.,!?\|/][}{)(=%*@`~'\""

    for char in chars_to_remove:
        text = text.replace(char, '')

    porter = PorterStemmer()

    if stemming:
        return porter.stem(term)
    else:
        return term