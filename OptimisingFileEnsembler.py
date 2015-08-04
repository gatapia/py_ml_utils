from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
sys.path.append('utils')
from misc import *
from FileEnsembler import *

class OptimisingFileEnsembler(FileEnsembler):  
  def __init__(self, scorer, method='Powell'):
    self.scorer = scorer
    self.method = method
    self.weights = []
    self.score = 0

  def get_weight_applier(self, arrays):
    def apply_weights(weights):
      weights /= np.sum(weights)
      final_prediction = 0
      for weight, prediction in zip(weights, arrays):
        final_prediction += weight * prediction
      return final_prediction
    return apply_weights

  def fit(self, train_files, y):    
    arrays = self._get_files(train_files)
              
    def min_func(weights):
      return -self.scorer(y, self.get_weight_applier(arrays)(weights))

    starting_values = [0.5] * len(arrays)
    bounds = [(0,1)] * len(arrays)

    res = scipy.optimize.minimize(min_func, starting_values, method=self.method, bounds=bounds)
    self.weights = res['x']/np.sum(res['x'])
    self.score = min_func(self.weights)

    print('Ensamble Score: {best_score}'.format(best_score=self.score))
    print('Best Weights: {weights}'.format(weights=self.weights))

    return self

  def transform(self, test_files):
    arrays = self._get_files(test_files)
    return self.get_weight_applier(arrays)(self.weights)    
