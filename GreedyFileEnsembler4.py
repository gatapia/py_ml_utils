from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys, inspect
sys.path.append('utils')
from misc import *
from FileEnsembler import *

class GreedyFileEnsembler4(FileEnsembler):  
  def __init__(self, scorer, min_epochs=10, max_replacements=0):
    '''
    arg:
    scorer: function that takes:
      y: actual y values
      predictions: predicted y values
      [optional] indexes: indexes currently being tested
    '''
    self.scorer = scorer
    self.min_epochs = min_epochs
    self.max_replacements = max_replacements    
    self.indexes = []    

    self.best_min_epochs = 0
    self.best_score = 0
    self.best_indexes = []

  def _get_files(self, files):
    if hasattr(files, '__call__'): 
      files = files()
    else:
      first = files[0]      
      if type(first) is str: 
        loaded = [load(f) for f in files]
        if loaded[0] is None: raise Exception('could not load files')
        files = loaded
    
    newarr = []
    for arr in files:
      newarr.append(arr.values if hasattr(arr, 'values') else arr)
    return newarr

  def fit(self, train_files, y):        
    self.y = y
    self.arrays = self._get_files(train_files)
    self.max_score = -999
    self.ensemble = []
    self.indexes = []
    self.current_epoch = -1    
    return self if self.min_epochs == 0 else self.continue_epochs(self.min_epochs)

  def continue_epochs(self, min_epochs):    
    def _score(preds, indexes):
      # inspect.getargspec
      if 'indexes' in inspect.getargspec(self.scorer).args:
        return self.scorer(self.y, preds, indexes)
      else: return self.scorer(self.y, preds)

    self.min_epochs = min_epochs
    for epoch in range(self.current_epoch + 1, self.min_epochs):
      self.current_epoch = epoch
      epoch_scores = []
      for idx, arr in enumerate(self.arrays):
        if self.indexes.count(idx) > self.max_replacements: 
          epoch_scores.append(-1)
          continue

        curr_ensemble = self.ensemble[:] + [arr]        
        preds = np.mean(curr_ensemble, 0)
        score = _score(preds, self.indexes + [idx])
        epoch_scores.append(score)

      epoch_index = np.argmax(epoch_scores)
      epoch_score = epoch_scores[epoch_index] if epoch_index >= 0 else -999
      if epoch_score > self.max_score:
        self.indexes.append(epoch_index)
        self.ensemble.append(self.arrays[epoch_index])        
        self.max_score = epoch_score
        print 'epoch:', epoch, 'found improved score:', self.max_score, 'index:', idx, 'ensemble size:', len(self.ensemble)
        
        if self.max_score > self.best_score:
          self.best_score = self.max_score
          self.best_min_epochs = epoch + 1
          self.best_indexes = self.indexes[:]
          print 'new total best score found:', self.best_score
      else:
        if epoch_index >= 0 and epoch_score != -1:  
          self.ensemble.append(self.arrays[epoch_index])
          self.indexes.append(epoch_index)
          self.max_score = _score(np.mean(self.ensemble, 0), self.indexes)
          print 'no improvement found after ', epoch, 'min_epochs, picking best:', self.max_score, 'index:', epoch_index     
        else:
          print 'no possible improvement found, exiting early'
          break
    
    print 'fit done indexes selected: ', self.indexes, 'self.max_score:', self.max_score
    return self

  def transform(self, test_files):
    test_arrays = self._get_files(test_files)
    test_ensemble = [test_arrays[i] for i in self.indexes]
    return np.mean(test_ensemble, 0)


  def transform_best(self, test_files):
    test_arrays = self._get_files(test_files)
    test_ensemble = [test_arrays[i] for i in self.best_indexes]
    return np.mean(test_ensemble, 0)