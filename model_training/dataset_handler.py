import csv
import json
import sys
import nltk
import os

import pandas as pd
import matplotlib.pyplot as plt

from os import path
from pandas import DataFrame
from typing import List, Tuple

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from wordcloud import WordCloud, STOPWORDS
from os.path import dirname, abspath

from sklearn.utils import resample

from utils.data_tools import remove_special_symbols, tokenize_and_lemmatize_and_steam_text, is_missing_data

SPAM_CLASS = 1
HAM_CLASS = 0

TEXT_TO_NUMBER_CLASS_MAP = {
    'spam': 1,
    'ham': 0
}

def _import_csv_dataset(dataset_path: str=None) -> List[dict]:
    """ Open and convert the csv dataset to a dict list
    Args: 
        dataset_path: The full path for a new dataset. By default it will use the spam.csv dataset if it is not defined.
    Returns:
        list[json]: Returns list of dictionaries containing two attributes: text and class
    """
    dataset_folder = 'dataset'
    directory = os.getcwd()
    dataset_filename = 'spam.csv'
    full_path = os.path.join(directory, dataset_folder, dataset_filename)
    json_dataset = []
    with open(full_path) as csvfile:
        spamreader = csv.reader(csvfile)
        next(spamreader)
        for row in spamreader:
            json_dataset.append({
                "class": TEXT_TO_NUMBER_CLASS_MAP[row[0]],
                "text": row[1]
            })
    return json_dataset

def clean_original_dataset(original_dataset:List[dict]) -> List[dict]:
    """ Clean the text, remove the rows with missing data, tokenize and lemmatize the text
    
    Args:
        -
    Returns:
        tuple: Retruns a tuple containing two lists of dictionaries, one with the base dataset 
        (just imported and converted to a list of dicts) and other cleanned.
    Raises:
        -
    """
    # original_dataset = _import_csv_dataset()
    clean_dataset = []
    for index, row in original_dataset.iterrows():
        text = row['text']
        clean_text = remove_special_symbols(text=text)
        tokens, lemmas = tokenize_and_lemmatize_and_steam_text(clean_text)
        row['text'] = clean_text
        try:
            text_class = row['class']
        except KeyError:
            text_class = None
        clean_dataset.append({
            'class': text_class,
            'text': clean_text,
            'tokens': tokens,
            'lemmas': lemmas,
            'words_count': len(tokens)
        })
    return original_dataset, clean_dataset

def get_class_values(target_class: int, dataset: List[dict]) -> List[dict]:
    """ Get all the entries of a given class
    
    Args:
        target_class: A integer (0 for ham, 1 for spam) representing the target class to filter the entries
        dataset: A list of dicts, representing the whole dataset.
    Returns:
        A list of all rows (dicts) of the same class
    Raises:
        -
    """
    return [data for data in dataset if data.get('class') == target_class]

def get_data_by_class(dataset: List[dict]) -> Tuple[List[dict], List[dict]]:
    """ Given a dataset, it splits the dataset by two classes (ham and spam)

    Args:
        dataset: A list of dicts, representing the whole dataset.
    Returns:
        tuple: Retruns a tuple containing two lists of dictionaries, one with all the spam entries
        and other with all the ham entries
    Raises:
        -
    """
    spam_list = get_class_values(target_class=SPAM_CLASS, dataset=dataset)
    ham_list = get_class_values(target_class=HAM_CLASS, dataset=dataset)
    return spam_list, ham_list

def create_classes_bar_plot(spam_list: List[dict], ham_list: List[dict]) -> None:
    """ Create a bar chart comparing the spam and the ham numbers

    Args:
        spam_list: A list of dictionaries with all the spam entries of the dataset
        ham_list: A list of dictionaries with all the ham entries of the dataset
    Returns:
        -
    Raises:
        -
    """
    spam_count = len(spam_list)
    ham_count = len(ham_list)
    total = spam_count + ham_count
    plt.bar(['spam', 'ham'], [spam_count, ham_count], tick_label = ['spam', 'ham'], width = 0.8, color = ['red', 'green'])

    plt.text(x=0, y=spam_count, s=f"{round(((spam_count/total)*100),0)}%", fontdict=dict(fontsize=20))
    plt.text(x=1, y=ham_count, s=f"{round(((ham_count/total)*100),0)}%", fontdict=dict(fontsize=20))

    plt.xlabel(f'spam x ham - {total} rows')
    plt.title('Dataset classes count')

def create_spam_word_cloud(spam_list: List[dict]) -> None:
    """ Create a word cloud chart for the spam entries

    Args:
        spam_list: A list of dictionaries with all the spam entries of the dataset
    Returns:
        -
    Raises:
        -
    """
    spam_big_bloob = ' '.join([data.get('text') for data in spam_list])
    word_cloud = WordCloud(stopwords=STOPWORDS, background_color='black').generate(spam_big_bloob)
    plt.figure(figsize=(15,10))
    plt.clf()
    plt.imshow(word_cloud)
    plt.title('Spam word cloud')
    plt.axis('off')

def create_ham_word_cloud(ham_list: List[dict]) -> None:
    """ Create a word cloud chart for the spam entries

    Args:
        ham_list: A list of dictionaries with all the ham entries of the dataset
    Returns:
        -
    Raises:
        -
    """
    ham_big_bloob = ' '.join([data.get('text') for data in ham_list])
    word_cloud = WordCloud(stopwords=STOPWORDS, background_color='black').generate(ham_big_bloob)
    plt.figure(figsize=(15,10))
    plt.clf()
    plt.imshow(word_cloud)
    plt.title('Ham word cloud')
    plt.axis('off')

def get_most_common_words(entries: List[dict]) -> List[Tuple[str, int]]:
    """ Get the list of the 30 most commom words on the list of entries

    Args:
        entries: A list of dicts (a part of the dataset)
    Returns:
        List(tuple): A list of tuples, where each tuple contains the word and its count on the text
    Raises:
        -
    """
    list_of_words = []
    for item in entries:
        list_of_words = list_of_words + item.get('tokens')
    wordfreqdist = nltk.FreqDist(list_of_words)
    return wordfreqdist.most_common(30)

def create_most_commom_words_bar_plot(data: List[dict], type: str) -> None:
    """ Create a bar chart comparing the word count of the 30 most commom on the dataset texts

    Args:
        data: A list of dictionaries with part of the entries of the dataset (spam or ham list)
    Returns:
        -
    Raises:
        -
    """
    most_commom_words = get_most_common_words(data)
    plt.figure(figsize=(15,10))
    plt.barh(range(len(most_commom_words)),[val[1] for val in most_commom_words], align='center')
    plt.yticks(range(len(most_commom_words)), [val[0] for val in most_commom_words])
    plt.title(f'{type} most commom words')

def get_dataframe_from_dict(dict_dataset: List[dict]) -> DataFrame:
    """ Convert the dict dataset to a pandas dataframe

    Args:
        dict_dataset: A list of dictionaries (the dataset)
    Returns:
        dataframe: A pandas dataframe with all the dataset entries
    Raises:
        -
    """
    df = pd.json_normalize(dict_dataset)
    df_majority = df[(df['class']==0)] 
    df_minority = df[(df['class']==1)] 
    # df_minority_upsampled = resample(
    #     df_minority, 
    #     replace=True,    # sample with replacement
    #     n_samples= 4828, # to match majority class
    #     random_state=200
    # )  # reproducible results
    # original_dataset = pd.concat([df_minority_upsampled, df_majority])
    df_majority_undersampled = resample(
        df_majority, 
        replace=False,    # sample with replacement
        n_samples= 724,   # to match minority class
        random_state=200
    )  # reproducible results
    original_dataset = pd.concat([df_majority_undersampled, df_minority])
    original_dataset.dropna(subset=['text'], inplace=True)
    original_dataset.dropna(subset=['class'], inplace=True)
    original_dataset.reset_index()
    return original_dataset

def get_train_and_test_data(dataframe: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """ Split the dataset in two parts (test and train)

    Args:
        dataframe: A pandas dataframe with all the dataset data
    Returns:
        tuple: A tuple with two pandas dataframes, the train dataframe (80% of the dataset) 
        and the test dataframe (20% of the dataset)
    Raises:
        -
    """
    train=dataframe.sample(frac=0.8,random_state=200) #random state is a seed value
    test=dataframe.drop(train.index)
    return train, test
    

def get_train_data() -> Tuple[List[dict], List[dict]]:
    """ Retrieves the dataset from disk, clean it, add train attributes and split it into train/test

    Args:
        -
    Returns:
        Tuple: A tuple containing the dataset split into two parts (train and test)
    Raises:
        -
    """
    json_dataset = _import_csv_dataset()
    dataset = get_dataframe_from_dict(json_dataset)
    original_data, clean_dataset = clean_original_dataset(dataset)
    spam_list, ham_list = get_data_by_class(clean_dataset)
    most_commom_spam_words = get_most_common_words(spam_list)

    for item in clean_dataset:
        tokens = item['tokens']
        top_spam_words = [tuple[0] for tuple in most_commom_spam_words]
        spam_words_count = len(list(set(tokens) & set(top_spam_words)))
        item['spam_words_count'] = spam_words_count

    dataframe = get_dataframe_from_dict(clean_dataset)
    train,test = get_train_and_test_data(dataframe)
    return train,test

if __name__ == "__main__":
    dataset = _import_csv_dataset()
    dataframe = get_dataframe_from_dict(dataset)
    original_data, clean_dataset = clean_original_dataset(dataframe)
    spam_list, ham_list = get_data_by_class(clean_dataset)
    create_classes_bar_plot(spam_list, ham_list)
    create_most_commom_words_bar_plot(data=spam_list, type="spam")
    create_most_commom_words_bar_plot(data=ham_list, type="ham")
    create_spam_word_cloud(spam_list)
    create_ham_word_cloud(ham_list)
    plt.show()
    train, test = get_train_and_test_data(dataframe)
    print(train)
    print(test)
