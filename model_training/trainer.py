import os
import joblib
from dataset_handler import get_train_data
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class ModelTrainer:
    
    @classmethod
    def run(cls, model:object, save:bool) -> None:
        """ Trains the received model

        Args:
            model: A classification algorithm object (E.g.: LogisticRegression)
            save: A boolean value that indicates if the trained model should be save on
                the persistence file or not
        Returns:
            -
        Raises:
            -
        """
        model_name = type(model).__name__
        print(f'Training {model_name}')
        accuracy_results = []
        train_dataframe,test_dataframe = get_train_data()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2),lowercase=True,stop_words='english')
        test_x = vectorizer.fit_transform(test_dataframe['text'])
        test_y = test_dataframe['class']
        # training the model with unigrams and bigrams
        train_x = vectorizer.fit_transform(train_dataframe['text'])

        # training the model with simple tokens count
        # train_x = train_dataframe['words_count'].values.reshape(-1,1)
        
        # training the model with spam words count inside the original text
        # train_x = train_dataframe['spam_words_count'].values.reshape(-1,1)
        train_y = train_dataframe['class']
        for i in range(0, 100):
            x, y = shuffle(train_x,train_y)
            test_x,test_y = shuffle(test_x,test_y)
            model.fit(x, y)
            result = cross_val_predict(model, test_x, test_y, cv=10)
            accuracy = metrics.accuracy_score(test_y, result)
            cm = metrics.confusion_matrix(test_y, result)
            accuracy_results.append(accuracy)
        
        average = sum(accuracy_results) / len(accuracy_results)
        print(f'{model_name}, Cross-validation result: {average}')
        print('Confusion matrix:')
        print(cm)

        if save:
            persistence_folder = 'persistence'
            directory = os.getcwd()
            model_filename = 'model.joblib'
            vectorizer_filename = 'vectorizer.joblib'
            model_path = os.path.join(directory, persistence_folder, model_filename)
            vectorizer_path = os.path.join(directory, persistence_folder, vectorizer_filename)
            joblib.dump(model, model_path)
            joblib.dump(vectorizer, vectorizer_path)