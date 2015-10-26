from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np
import pandas as pd
import math

class TransformYClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, 
      base_classifier, 
      transform_on_fit=None, 
      anti_transform_on_predict=None):   
    '''
    This wrapper allows transformation before calling
    fit.  This is the transform_on_fit param.  We also 
    support transformation after prediction this is the
    anti_transform_on_predict.  One or both can be set.
    If anti_transform_on_predict is the same special
    string as transform_on_fit then we take the anti 
    transformation if possible.  Both of these can
    also be callables or a scaling factor.
    '''
    if transform_on_fit is None and anti_transform_on_predict is None:
      raise Exception('Either transform_on_fit or anti_transform_on_predict must be set')
      
    self._shift_val = 0
    self.base_classifier = base_classifier    
    self.transform_on_fit = transform_on_fit
    self.anti_transform_on_predict = anti_transform_on_predict

  def _on_fit_transform(self, X, y):
    if self.transform_on_fit is None: return y
    if hasattr(self.transform_on_fit, '__call__'):
      return self.transform_on_fit(X, y)      
    elif type(self.transform_on_fit) is float:
      return y * self.transform_on_fit
    elif self.transform_on_fit.startswith('log'): 
      if '+' in self.transform_on_fit:
        self._shift_val = float(self.transform_on_fit.split('+')[1])        
      else:
        ymin = y.min()
        if ymin < 0: 
          self._shift_val = ymin * -1.01
      return np.log(y + self._shift_val)
    elif self.transform_on_fit == 'arcsinh':
      return np.arcsinh(y)
    elif self.transform_on_fit == 'pseudoLog10':
      return np.arcsinh(y/2.)/np.log(10)
    else: raise Exception('Not Supported: ' + `self.transform_on_fit`)    

  def _post_predict_transform(self, X, y):
    if not self.anti_transform_on_predict: return y

    if hasattr(self.anti_transform_on_predict, '__call__'):
      return self.anti_transform_on_predict(X, y)

    if self.anti_transform_on_predict != self.transform_on_fit:
      raise Exception('Currently only the same operation is ' +
          'supported _post_predict_transform as we try to ' +
          'automatically apply the anti transformation')

    if type(self.anti_transform_on_predict) is float:
      return y / self.anti_transform_on_predict
    elif self.anti_transform_on_predict.startswith('log'): 
      return np.exp(y) - self._shift_val
    elif self.anti_transform_on_predict == 'arcsinh': 
      return np.sinh(y)
    elif self.anti_transform_on_predict == 'pseudoLog10': 
      return np.sinh(y * 2) * np.log(10)
    else: raise Exception('Not Supported: ' + self.anti_transform_on_predict)


  def fit(self, X, y, fit_params=None):    
    if fit_params is None: fit_params = {}
    self.base_classifier.fit(X, self._on_fit_transform(X, y), **fit_params)
    return self

  def predict(self, X): 
    return self._post_predict_transform(X, self.base_classifier.predict(X))

  def predict_proba(self, X): 
    return self._post_predict_transform(X, self.base_classifier.predict_proba(X))
