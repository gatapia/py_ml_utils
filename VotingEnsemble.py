
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import cross_validation
from scipy import stats
import numpy as np
import itertools
from scipy.stats import sem 

class VotingEnsemble(BaseEstimator, ClassifierMixin):
  def __init__(self, models, voter='majority', use_proba=False):    
    self.models = models
    self.voter = voter
    self.use_proba = use_proba

  def cv(self, X, y, scorer, n_samples=None, n_folds=5):
    if not(isinstance(X, list)): 
      X = list(itertools.repeat(X, len(self.models)))
    
    if not(isinstance(y, list)): 
      y = list(itertools.repeat(y, len(self.models)))
      
    if n_samples is None: n_samples = X[0].shape[0]

    cv = cross_validation.KFold(n_samples, n_folds=n_folds, indices=False)
    scores = []
    for train, test in cv:      
      Xs = []
      ys = []
      X_tests = []
      
      for i, clf in enumerate(self.models):        
        Xs.append(X[i].iloc[train])
        ys.append(y[i].iloc[train])
        X_tests.append(X[i].iloc[test])

      predictions = self.fit(Xs, ys).predict(X_tests)
      scores.append(scorer(y[0].iloc[test], predictions))

    cv = (np.mean(scores), sem(scores))
    if cfg['debug']: print 'cv %.5f (+/-%.5f)' % cv
    return cv


  def fit(self, X, y):
    """X can either be a dataset or a list of datasets"""
    if not(isinstance(X, list)): 
      print 'X is a single dataset'
      X = list(itertools.repeat(X, len(self.models)))
    if not(isinstance(y, list)): 
      y = list(itertools.repeat(y, len(self.models)))
    for i, m in enumerate(self.models): m.fit(X[i], y[i])
    return self

  def predict(self, X):
    """X can either be a dataset or a list of datasets"""
    if not(isinstance(X, list)): 
      print 'X is a single dataset'
      X = list(itertools.repeat(X, len(self.models)))

    all_preds = []
    if self.voter == 'mean' or self.voter == 'median' or \
        self.voter == 'max' or self.voter == 'min':      
      all_preds = [m.predict_proba(X[i]).T[1] if self.use_proba 
        else m.predict(X[i]) for i, m in enumerate(self.models)]
    else:
      all_preds = [m.predict(X[i]) for i, m in enumerate(self.models)]

    predictions = np.empty(X[0].shape[0], dtype=type(all_preds[0][0]));
    for i in range(X[0].shape[0]):
      i_preds = [ps[i] for ps in all_preds]      

      if (self.voter == 'majority'): predictions[i] = stats.mode(i_preds)[0]
      elif (self.voter == 'mean'): predictions[i] = np.mean(i_preds)
      elif (self.voter == 'max'): predictions[i] = np.max(i_preds)
      elif (self.voter == 'min'): predictions[i] = np.min(i_preds)
      elif (self.voter == 'median'): predictions[i] = np.median(i_preds)
      else: raise Error(self.voter + ' is not implemented')
    return predictions