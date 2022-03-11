import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from prediction.predictor import Predictor
from utils.data_tools import remove_special_symbols, tokenize_and_lemmatize_and_steam_text


def _clean_text(text:str) -> str:
    """ Does the basic text cleaning

    Args:
        text: The message that should be cleaned
    Returns:
        string: The message cleaned, tokenized and lemmatized
    Raises:
        -
    """
    clean_text = remove_special_symbols(text=text)
    tokens, lemmas = tokenize_and_lemmatize_and_steam_text(clean_text)
    return ' '.join(lemmas)

def predict(text:str) -> str:
    """ Predicts the class for a given text

    Args: 
        text: The message to classify
    Returns:
        string: The class for the message received
    Raises:
        -
    """
    predictor = Predictor()
    clean_text = _clean_text(text)
    result = predictor.predict(clean_text)
    return "spam" if result[0] == 1 else "ham"

def get_classifier_model() -> str:
    """ Gets the classifier algorithm name

    Args:
        -
    Returns:
        string: The classifier algorithm name tha the predict method uses
    Raises:
        -
    """
    return type(Predictor().model).name

def get_version() -> str:
    """ Gets the current version of the library

    Args:
        -
    Returns:
        string: The current version of the library
    Raises:
        -
    """
    return '0.0.1'