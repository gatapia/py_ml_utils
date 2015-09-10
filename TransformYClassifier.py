from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np
import pandas as pd
import math

class TransformYClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, base_classifier, 
      transformation, 
      transform_on_fit=True, 
      anti_transform_on_predict=True):    
    self.base_classifier = base_classifier
    self.transformation = transformation
    self.shift_val = 0
    self.transform_on_fit = transform_on_fit
    self.anti_transform_on_predict = anti_transform_on_predict

  def _do_transformation_impl(self, X, y):
    if hasattr(self.transformation, '__call__'):
      return self.transformation(X, y)      
    elif self.transformation.startswith('log'): 
      if '+' in self.transformation:
        self.shift_val = float(self.transformation.split('+')[1])        
      else:
        ymin = y.min()
        if ymin < 0: 
          self.shift_val = ymin * -1.01
      return np.log(y + self.shift_val)
    elif self.transformation == 'arcsinh':
      return np.arcsinh(y)
    elif self.transformation == 'pseudoLog10':
      return np.arcsinh(y/2.)/np.log(10)
    else: raise Exception('Not Supported: ' + `self.transformation`)    

  def _on_fit_transform(self, X, y):
    if not self.transform_on_fit: return y    
    return self._do_transformation_impl(X, y)
  
  def _post_predict_transform(self, X, y):
    if not self.transform_on_fit: return self._do_transformation_impl(y)
    if not self.anti_transform_on_predict: return y
    if hasattr(self.anti_transform_on_predict, '__call__'):
      return self.anti_transform_on_predict(X, y)
    elif self.transformation.startswith('log'): 
      return np.exp(y) - self.shift_val
    elif self.transformation == 'arcsinh': 
      return np.sinh(y)
    elif self.transformation == 'pseudoLog10': 
      return np.sinh(y * 2) * np.log(10)
    else: raise Exception('Not Supported: ' + self.transformation)


  def fit(self, X, y):    
    self.base_classifier.fit(X, self._on_fit_transform(X, y))
    return self

  def predict(self, X): 
    return self._post_predict_transform(X, self.base_classifier.predict(X))

  def predict_proba(self, X): 
    return self._post_predict_transform(X, self.base_classifier.predict_proba(X))
