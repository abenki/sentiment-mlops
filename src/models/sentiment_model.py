from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SentimentModel:
    """Wrapper for sentiment analysis pipeline"""
    
    def __init__(self, max_features=5000, C=1.0, max_iter=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents='unicode',
            lowercase=True
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X, y):
        """Trains the vectorizer and classifier"""
        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)
        return self
    
    def predict(self, X):
        """Predicts the labels"""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)
    
    def predict_proba(self, X):
        """Predicts the probabilities"""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)
