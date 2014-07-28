
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import numpy as np
import itertools

class VotingEnsemble(BaseEstimator, ClassifierMixin):
  def __init__(self, models, voter='majority'):    
    self.models = models
    self.voter = voter

  def fit(self, X, y):
    """X can either be a dataset or a list of datasets"""
    if not(isinstance(X, list)): 
      print 'X is a single dataset'
      X = repeat(X, len(self.models))
    for i, m in enumerate(self.models): m.fit(X[i], y)
    return self

  def predict(self, X):
    """X can either be a dataset or a list of datasets"""
    if not(isinstance(X, list)): 
      print 'X is a single dataset'
      X = itertools.repeat(X, len(self.models))
    all_preds = []
    if self.voter == 'mean' or self.voter == 'median':  
      all_preds = [m.predict_proba(X[i]).T[1] for i, m in enumerate(self.models)]
    else:
      all_preds = [m.predict(X[i]) for i, m in enumerate(self.models)]

    predictions = np.empty(X[0].shape[0], dtype=type(all_preds[0][0]));
    for i in range(X[0].shape[0]):
      i_preds = [ps[i] for ps in all_preds]      

      if (self.voter == 'majority'): predictions[i] = stats.mode(i_preds)[0]
      elif (self.voter == 'mean'): predictions[i] = np.mean(i_preds)
      elif (self.voter == 'median'): predictions[i] = stats.median(i_preds)[0]
      else: raise Error(self.voter + ' is not implemented')
    return predictions