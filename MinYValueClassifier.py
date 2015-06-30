from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np
import pandas as pd
import math

class MinYValueClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, base_classifier, min_value=0):    
    self.base_classifier = base_classifier
    self.min_value = min_value

  def fit(self, X, y):    
    self.base_classifier.fit(X, y)
    return self

  def predict(self, X): 
    predictions = self.base_classifier.predict(X)
    predictions[predictions < self.min_value]  = self.min_value
    return predictions
