from dataset_handler import get_train_data
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

class ModelTrainer:
    
    @classmethod
    def run(cls, model):
        model_name = type(model).__name__
        print(f'Training {model_name}')
        accuracy_results = []
        train_dataframe = get_train_data()
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x = vectorizer.fit_transform(train_dataframe['text'])
        y = train_dataframe['class']
        for i in range(0, 100):
            x, y = shuffle(x,y)
            model.fit(x, y)
            result = cross_val_predict(model, x, y, cv=10)
            accuracy = metrics.accuracy_score(y, result)
            accuracy_results.append(accuracy)
        
        average = sum(accuracy_results) / len(accuracy_results)
        print(f'{model_name}, Cross-validation result: {average}')