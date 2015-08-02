from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
sys.path.append('utils')
from misc import *

class OptimisingFileEnsembler(BaseEstimator, TransformerMixin):  
  def __init__(self, scorer, min_epochs=10):
    self.scorer = scorer
    self.min_epochs = min_epochs
    self.indexes = []

  def _get_files(self, files):
    if hasattr(files, '__call__'): return files()
    first = files[0]
    if type(first) is str: 
      print 'files:', files
      loaded = [load(f) for f in files]
      if loaded[0] is None: raise Exception('could not load files')
      print 'loaded:', len(loaded)
      return loaded
    return files

  def get_weight_applier(arrays):
    def apply_weights(weights):
      weights /= np.sum(weights)
      final_prediction = 0
      for weight, prediction in zip(weights, arrays):
        final_prediction += weight * prediction
      return final_prediction
    return apply_weights

  def fit(self, train_files, y):    
    arrays = self._get_files(train_files)
    scores = [self.scorer(y, arr) for arr in arrays]
    max_score = np.max(scores)
    ensemble = [arrays[np.argmax(scores)]]
    self.indexes = [np.argmax(scores)]
              
    def min_func(weights):
      return -self.scorer(get_weight_applier(arrays)(weights), y)

    starting_values = [0.5] * len(arrays)
    bounds = [(0,1)] * len(arrays)

    res = scipy.optimize.minimize(min_func, starting_values, method='L-BFGS-B', bounds=bounds)
    self.weights = res['x']/np.sum(res['x'])
    score = min_func(self.weights)
    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=self.weights))


    return self

  def transform(self, test_files):
    arrays = self._get_files(test_files)
    return get_weight_applier(arrays)(self.weights)    
