from aux.term_parsing import *

def find_closest_occurence_left(main_search_index, search_term, terms, words):
    i = main_search_index
    # loop over all words on the left side of the term that is searched for
    # look for the occurence of the closest left occurence of a term that completes the needed amount of terms on the left side
    while(words[i] != search_term):
        i -= 1

    return i

# same as previous, but on the right
def find_closest_occurence_right(main_search_index, search_term, terms, words):
    i = main_search_index
    while(words[i] != search_term):
        i += 1

    return i

def search_context(source_text, search_index, stopwords_option, stemming_option):
    # getting a list of all the words from the source_text (not stemmed, with stopwords)
    source_text_split = source_text.strip().split(' ')  
    # getting a list of all the words from the source_text (stemmed, with stopwords)
    words = extract_terms(source_text, stemming_option) 

    # if stopwords enabled, remove stopwords from the word list to get terms 
    if stopwords_option:
        terms = remove_stopwords(words)
    # else just terms are same as words
    else:
        terms = words

    # find the corresponding index of the search word in the term list
    search_index_terms = terms.index(words[search_index])

    terms_available_left = 0
    terms_available_right = 0


    # if there are more than or 5 terms on the left side
    if search_index_terms >= 5:
        # get at least 5 terms from the left side
        terms_available_left = 5
        # if there are 5 terms available from the right side
        if search_index_terms + 5 < len(terms):
            # get 5 terms from the right side
            terms_available_right = 5
        # if there are not 5 terms available from the right side
        else:
            # calculate how many terms are available from the right side
            terms_available_right = len(terms) - search_index_terms - 1
            # calclulate how many more terms do we need to get the total of 10 terms
            remaining_terms = 5 - terms_available_right
            # try to get as many terms from the left side to compensate for the lack of terms on the right side
            if search_index_terms - terms_available_left >= remaining_terms:
                terms_available_left += remaining_terms
            # if not possible to get enough terms from the left side to get 10 in total, get as many as possible
            else:
                terms_available_left = search_index_terms

    # if there are less than 5 terms available on the left side
    else:
        # get however many terms are available on the left side
        terms_available_left = search_index_terms
        # calculate how many more terms are needed for the total of 10 
        remaining_terms = 10 - terms_available_left
        # try to get enough terms on the right side to compensate for the lack of terms on the left side
        if search_index_terms + remaining_terms < len(terms):
            terms_available_right = remaining_terms
        # if not possible to get enough terms from the right side to get 10 in total, get as many as possible
        else:
            terms_available_right = len(terms) - search_index_terms - 1 

    context_list = []

    # finds the index of the most left available term that is needed from the left side
    left_index = find_closest_occurence_left(search_index, terms[search_index_terms - terms_available_left], terms, words)

    # finds the index of the most right available term that is needed from the right side
    right_index = find_closest_occurence_right(search_index, terms[search_index_terms + terms_available_right], terms, words)
    for i in range(left_index, right_index + 1):
        context_list.append(source_text_split[i])

    # convert context list of words into a string and return
    return ' '.join(context_list)