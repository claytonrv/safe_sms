import re

import nltk
from nltk.stem import SnowballStemmer
from string import digits

nltk.download('stopwords')
nltk.download('punkt')

language = 'english'
stemmer = SnowballStemmer(language)
portuguese_stopwords = nltk.corpus.stopwords.words(language)

htmlTagsAndSymbolsRegex = re.compile(r'(<[^>]+>)|(&#[0-9]+;)')
lineBreaksAndTabsRegex = re.compile(r'(\\n|\n|\\t|\t)+')
numbers = re.compile(r'[0-9]')

def remove_special_symbols(text):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    text = numbers.sub('', text)
    text = htmlTagsAndSymbolsRegex.sub('', text)
    text = lineBreaksAndTabsRegex.sub(' ', text)
    tokens = tokenizer.tokenize(text)
    non_stop_tokens = [word for word in tokens if (word not in portuguese_stopwords and len(word) > 3)]
    text = ' '.join(non_stop_tokens)
    return text.strip().lower()


def tokenize_and_lemmatize_text(text):
    tokens = nltk.tokenize.word_tokenize(text, language='english')
    lemmas = [token for token in tokens if len(token) > 2]
    lemmas = [lemma for lemma in lemmas if lemma not in portuguese_stopwords]
    lemmas = [stemmer.stem(lemma) for lemma in lemmas]
    return tokens, lemmas


def is_missing_data(row):
    valid_class = row['class'] is None or row['class'] == ''
    valid_text = row['text'] is None or row['text'] == '' or len(row['text']) < 3
    return valid_class and valid_text
