from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np
import pandas as pd
import math

class TransformYClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, base_classifier, transformation, transform_on_fit=True, anti_transform_on_predict=True):    
    self.base_classifier = base_classifier
    self.transformation = transformation
    self.shift_val = 0
    self.transform_on_fit = transform_on_fit

  def _do_transformation_impl(self, y):
    if hasattr(self.transformation, '__call__'):
      return self.transformation(y)      
    elif self.transformation == 'log': 
      ymin = y.min()
      if ymin < 0:
        self.shift_val = ymin * -1.01
      return np.log(y + self.shift_val)
    elif self.transformation == 'arcsinh':
      return np.arcsinh(y)
    else: raise Exception('Not Supported: ' + `self.transformation`)    

  def _on_fit_transform(self, y):
    if not self.transform_on_fit: return y    
    return self._do_transformation_impl(y)
  
  def _post_predict_transform(self, y):
    if not self.transform_on_fit: return self._do_transformation_impl(y)
    if not anti_transform_on_predict: return y

    elif self.transformation == 'log': return np.power(math.e, y) - self.shift_val
    elif self.transformation == 'arcsinh': return np.sinh(y)
    else: raise Exception('Not Supported: ' + self.transformation)


  def fit(self, X, y):    
    self.base_classifier.fit(X, self._on_fit_transform(y))
    return self

  def predict(self, X): 
    return self._post_predict_transform(self.base_classifier.predict(X))

  def predict_proba(self, X): 
    return self._post_predict_transform(self.base_classifier.predict_proba(X))

def get_sigmoid_transform(beta):
  return lambda y: 0.5 * ((2 * abs(y - 0.5)) ** beta) * np.sign(y - 0.5) + 0.5

def get_logistic_transform(k):
  return lambda y: 1.0 / (1.0 + np.exp(-k * preprocessing.scale(y)))