from sklearn.base import BaseEstimator, TransformerMixin

class categorize_pdays(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda val: (
            'never' if val == 999 else
            'recent' if val <= 5 else
            'old'
        ))
