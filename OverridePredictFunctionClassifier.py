from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np

class OverridePredictFunctionClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, base_classifier, predict_function):        
    self.base_classifier = base_classifier
    self.predict_function = predict_function    

  def fit(self, X, y):    
    self.base_classifier = self.base_classifier.fit(X, y)
    return self

  def predict(self, X): 
    if self.predict_function == 'predict_proba':
      return self.base_classifier.predict_proba(X).T[1]
    if self.predict_function == 'decision_function':
      return self.base_classifier.decision_function(X)
    else: raise Exception(self.predict_function + ' not supported')

  def predict_proba(self, X):
    if self.predict_function == 'decision_function':
      df = self.base_classifier.decision_function(X)
      return preprocessing.MinMaxScaler().fit_transform(df)
    elif self.predict_function == 'predict':
      return self.base_classifier.predict(X)
    else:
      return self.base_classifier.predict_proba(X)
  
  