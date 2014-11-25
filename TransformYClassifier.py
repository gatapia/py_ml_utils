from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
import math

class TransformYClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, base_classifier, transformation):    
    self.base_classifier = base_classifier
    self.transformation = transformation
    self.shift_val = 0

  def _do_transform(self, y):
    if hasattr(self.transformation, '__call__'):
      return self.transformation(y)      
    elif self.transformation == 'log': 
      ymin = y.min()
      if ymin < 0:
        self.shift_val = ymin * -1.01
      return np.log(y + self.shift_val)
    elif self.transformation == 'arcsinh':
      return np.arcsinh(y)
    else: raise Exception('Not Supported: ' + self.transformation)    
  
  def _do_anti_transform(self, y):
    if self.transformation == 'log': return np.power(math.e, y) - self.shift_val
    elif self.transformation == 'arcsinh': return np.sinh(y)
    else: raise Exception('Not Supported: ' + self.transformation)


  def fit(self, X, y):    
    self.base_classifier.fit(X, self._do_transform(y))
    return self

  def predict(self, X): 
    return self._do_anti_transform(self.base_classifier.predict(X))
