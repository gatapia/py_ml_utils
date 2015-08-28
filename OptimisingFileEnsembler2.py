from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
sys.path.append('utils')
from misc import *
from FileEnsembler import *

class OptimisingFileEnsembler2(FileEnsembler):  
  def __init__(self, scorer, method='Powell', max_bound=1, min_func=None):
    self.scorer = scorer
    self.min_func = min_func
    self.method = method
    self.max_bound = max_bound
    self.weights = []
    self.score = 0

  def get_weight_applier(self, arrays):
    def apply_weights(weights): return np.average(arrays, 0, weights)
    return apply_weights

  def fit(self, train_files, y):    
    arrays = self._get_files(train_files)
              
    def min_func(weights):
      return -self.scorer(y, self.get_weight_applier(arrays)(weights))

    starting_values = [self.max_bound/2.] * len(arrays)
    bounds = [(0,self.max_bound)] * len(arrays)

    actual_min_f = self.min_func if self.min_func is not None else min_func
    res = scipy.optimize.minimize(actual_min_f, starting_values, method=self.method, bounds=bounds)
    self.weights = res['x']/np.sum(res['x'])
    self.score = actual_min_f(self.weights)

    print('Ensamble Score: {best_score}'.format(best_score=self.score))
    print('Best Weights: {weights}'.format(weights=self.weights))

    return self

  def transform(self, test_files):
    arrays = self._get_files(test_files)
    return self.get_weight_applier(arrays)(self.weights)    
