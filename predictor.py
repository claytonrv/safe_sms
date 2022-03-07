import joblib
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer

class Predictor:

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load()

    def load(self):
        self.model = joblib.load(f'persistence/model.joblib')
        self.vectorizer = joblib.load(f'persistence/vectorizer.joblib')
        
    def predict(self, sms_text):
        text = self.vectorizer.transform([sms_text])
        return self.model.predict(text)