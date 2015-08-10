from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
sys.path.append('utils')
from misc import *
from FileEnsembler import *

class GreedyFileEnsembler2(FileEnsembler):  
  def __init__(self, scorer, min_epochs=10, replacement=True, max_replacements=0):
    self.scorer = scorer
    self.min_epochs = min_epochs
    self.replacement = replacement
    self.max_replacements = max_replacements    
    self.indexes = []

    self.best_min_epochs = 0
    self.best_score = 0

  def _get_files(self, files):
    if hasattr(files, '__call__'): return files()
    first = files[0]
    if type(first) is str: 
      loaded = [load(f) for f in files]
      if loaded[0] is None: raise Exception('could not load files')
      return loaded
    return files

  def fit(self, train_files, y):        
    self.y = y
    self.arrays = self._get_files(train_files)
    scores = [self.scorer(self.y, arr) for arr in self.arrays]
    self.max_score = np.max(scores)
    self.ensemble = [self.arrays[np.argmax(scores)]]
    self.indexes = [np.argmax(scores)]
    print 'starting score:', self.max_score, 'index:', self.indexes[0]
    self.current_epoch = -1
    return self.continue_epochs(self.min_epochs)

  def continue_epochs(self, min_epochs):    
    self.min_epochs = min_epochs
    for epoch in range(self.current_epoch + 1, self.min_epochs):
      self.current_epoch = epoch

      epoch_score = -999
      epoch_index = -1
      for idx, arr in enumerate(self.arrays):
        if not self.replacement and idx in self.indexes: continue
        if self.replacement and self.max_replacements > 0 and self.indexes.count(idx) >= self.max_replacements: continue

        idx_ensemble = self.ensemble[:] + [arr]        
        preds = np.mean(idx_ensemble, 0)
        score = self.scorer(self.y, preds)
        if score > epoch_score:
          epoch_score = score
          epoch_index = idx

      if epoch_score > self.max_score:
        self.indexes.append(epoch_index)
        self.ensemble.append(self.arrays[epoch_index])        
        self.max_score = epoch_score
        print 'epoch:', epoch, 'found improved score:', self.max_score, 'index:', idx, 'ensemble size:', len(self.ensemble)
        
        if epoch >= 10 and score > self.best_score:
          self.best_score = self.max_score
          self.best_min_epochs = epoch + 1
          print 'new total best score found'
      else:
        if epoch_index >= 0:                
          self.ensemble.append(self.arrays[epoch_index])
          self.indexes.append(epoch_index)
          self.max_score = self.scorer(self.y, np.mean(self.ensemble, 0))
        print 'no improvement found after ', epoch, 'min_epochs, picking best:', self.max_score, 'index:', epoch_index     
    
    print 'fit done indexes selected: ', self.indexes, 'self.max_score:', self.max_score
    return self

  def transform(self, test_files):
    test_arrays = self._get_files(test_files)
    test_ensemble = [test_arrays[i] for i in self.indexes]
    return np.mean(test_ensemble, 0)
