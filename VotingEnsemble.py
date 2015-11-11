
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import cross_validation
from scipy import stats
import numpy as np
import itertools
from scipy.stats import sem 

class VotingEnsemble(BaseEstimator, ClassifierMixin):
  def __init__(self, models, voter='majority', use_proba=False, weights=None):    
    self.models = models
    self.voter = voter
    self.use_proba = use_proba    
    self.weights = weights

  def fit(self, X, y):
    """X can either be a dataset or a list of datasets"""
    if not(isinstance(X, list)): 
      X = list(itertools.repeat(X, len(self.models)))
    if not(isinstance(y, list)): 
      y = list(itertools.repeat(y, len(self.models)))
    for i, m in enumerate(self.models): 
      m.fit(X[i], y[i])
    return self

  def predict_proba(self, X):
    classone_probs = self.predict_impl(X)
    classzero_probs = 1.0 - classone_probs
    return np.vstack((classzero_probs, classone_probs)).transpose()

  def predict(self, X):
    predictions = self.predict_impl(X)
    if self.use_proba:
      new_preds = np.zeros(len(predictions))
      new_preds[predictions > .5] = 1
      predictions = new_preds
    return predictions

  def predict_impl(self, X):
    """X can either be a dataset or a list of datasets"""
    if not(isinstance(X, list)): 
      X = list(itertools.repeat(X, len(self.models)))

    all_preds = []
    if self.voter == 'mean' or self.voter == 'median' or \
        self.voter == 'max' or self.voter == 'min':      
        for i, m in enumerate(self.models):
          if self.use_proba:
            all_preds.append(m.predict_proba(X[i]).T[1])
          else: 
            all_preds.append(m.predict(X[i]))
    else:
      all_preds = [m.predict(X[i]) for i, m in enumerate(self.models)]

    predictions = np.empty(X[0].shape[0], dtype=type(all_preds[0][0]));
    for i in range(X[0].shape[0]):
      i_preds = [ps[i] for ps in all_preds]      

      if (self.voter == 'majority'): predictions[i] = stats.mode(i_preds)[0]
      elif (self.voter == 'mean'): 
        if self.weights is not None: predictions[i] = np.average(i_preds, 0, weights)
        else: predictions[i] = np.mean(i_preds, 0)
      elif (self.voter == 'max'): predictions[i] = np.max(i_preds)
      elif (self.voter == 'min'): predictions[i] = np.min(i_preds)
      elif (self.voter == 'median'): predictions[i] = np.median(i_preds)
      else: raise Error(self.voter + ' is not implemented')
    return predictions