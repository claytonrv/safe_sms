import re

import nltk
from typing import List, Tuple
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

language = 'english'
stemmer = SnowballStemmer(language)
lemmatizer = WordNetLemmatizer()
portuguese_stopwords = nltk.corpus.stopwords.words(language)

htmlTagsAndSymbolsRegex = re.compile(r'(<[^>]+>)|(&#[0-9]+;)')
lineBreaksAndTabsRegex = re.compile(r'(\\n|\n|\\t|\t)+')
numbers = re.compile(r'[0-9]')

def remove_special_symbols(text: str) -> str:
    """ Remove all special characters, numbers, stopwords (the, a, and, etc) 
    and words with 1 or 2 characters (u, p, etc) from a given text

    Args:
        text: A text to be clean
    Returns:
        text: The clean text in lowercase
    Raises:
        -
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    text = numbers.sub('', text)
    text = htmlTagsAndSymbolsRegex.sub('', text)
    text = lineBreaksAndTabsRegex.sub(' ', text)
    tokens = tokenizer.tokenize(text)
    non_stop_tokens = [word for word in tokens if (word not in portuguese_stopwords and len(word) > 3)]
    text = ' '.join(non_stop_tokens)
    return text.strip().lower()


def tokenize_and_lemmatize_and_steam_text(text: str) -> Tuple[List[str], List[str]]:
    """ Convert a text into a list of tokens and a list of lemmas
    A token is basically a part of the original text: 
        E.g.: Text = 'That is an example text' tokens = ['That', 'is', 'an', 'example', 'text']
    A lemma is also a part of the original text:
        E.g.: Text = 'That is an example text' lemmas = ['That', 'example', 'text']
    By applying the stem to the lemmas, it removes the context of the word and normalized it:
        E.g.: Text = 'That is an example text' stem_lemmas = ['that', 'exampl', 'text']
    Args:
        text: A text to be normalized
    Returns:
        tuple: A tuple containing the list of tokens and the list of lemmatized and stem tokens
    Raises:
        -
    """
    tokens = nltk.tokenize.word_tokenize(text, language='english')
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in portuguese_stopwords]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    stem_lemmas = [stemmer.stem(lemma) for lemma in lemmas]
    return tokens, stem_lemmas


def is_missing_data(row: dict) -> bool:
    """ Check if a given row is missing data, text or class

    Args:
        row: A dictionay representation of a row with two attributes (text and class)
    Returns:
        bool: A boolean value representing if the row is or not missing data. 
        It return true if the text or the class attrs are None or empty, and false otherwise.
    Raises:
        -
    """
    try:
        valid_class = row['class'] is None or row['class'] == ''
    except KeyError:
        valid_class = False
    valid_text = row['text'] is None or row['text'] == '' or len(row['text']) < 3
    return valid_class and valid_text
