

from config import Config
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
        Config.ENABLE_PORTER_STEMMER = True
    elif not response:
        Config.ENABLE_PORTER_STEMMER = False
    print(Config.ENABLE_PORTER_STEMMER)


def enable_stop_word_removal(query, confirmation, rejection):
    response = user_boolean_selection(query, confirmation, rejection)
    if response:
        Config.ENABLE_STOP_WORD_REMOVAL = True
    elif not response:
        Config.ENABLE_STOP_WORD_REMOVAL = False
    print(Config.ENABLE_STOP_WORD_REMOVAL)
