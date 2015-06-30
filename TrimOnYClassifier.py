from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class TrimOnYClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, base_classifier, min_y, max_y=None):    
    '''
    Truncates data along the target variable for training.  This allows
    the classifier not to be thrown off by outliers.  During prediction
    data is not truncated.
    '''
    self.base_classifier = base_classifier
    self.min_y = min_y
    self.max_y = max_y

  def fit(self, X, y):    
    if hasattr(X, 'trim_on_y'):
      X, y = X.trim_on_y(y, self.min_y, self.max_y)
    else:
      X, y = pd.DataFrame(X).trim_on_y(y, self.min_y, self.max_y)

    self.base_classifier = self.base_classifier.fit(X, y)
    return self

  def predict(self, X): return self.base_classifier.predict(X)
  
  def predict_proba(self, X): return self.base_classifier.predict_proba(X)  