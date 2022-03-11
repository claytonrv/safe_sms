import os
import joblib
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer

class Predictor:

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load()

    def load(self):
        persistence_folder = 'persistence'
        directory = os.getcwd()
        model_filename = 'model.joblib'
        vectorizer_filename = 'vectorizer.joblib'
        model_path = os.path.join(directory, persistence_folder, model_filename)
        vectorizer_path = os.path.join(directory, persistence_folder, vectorizer_filename)
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
    def predict(self, sms_text):
        text = self.vectorizer.transform([sms_text])
        return self.model.predict(text)