

import config
import sys
sys.path.append("..")  # Adds higher directory to python modules path.


def user_boolean_selection(query, confirmation, rejection):
    response = input(query)
    if response.lower() == "y":
        print(confirmation)
        return True
    elif response.lower() == "n":
        print(rejection)
        return False
    return user_boolean_selection(query, confirmation, rejection)


def enable_porter_stemmer(query, confirmation, rejection):
    response = user_boolean_selection(query, confirmation, rejection)
    if response:
        config.change_stemming_option(True)
    elif not response:
        config.change_stemming_option(False)
    return response


def enable_stop_word_removal(query, confirmation, rejection):
    response = user_boolean_selection(query, confirmation, rejection)
    if response:
        config.change_stop_word_removal_option(True)
    elif not response:
        config.change_stop_word_removal_option(False)
    return response
