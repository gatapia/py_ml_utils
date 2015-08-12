from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
sys.path.append('utils')
from misc import *
from FileEnsembler import *

class GreedyFileEnsembler3(FileEnsembler):  
  def __init__(self, scorer, min_epochs=10, replacement=True, 
      max_replacements=0, best_chooser=None):
    self.scorer = scorer
    self.min_epochs = min_epochs
    self.replacement = replacement
    self.max_replacements = max_replacements    
    self.indexes = []
    self.best_chooser = best_chooser

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
    self.max_score = -999
    self.ensemble = []
    self.indexes = []
    self.current_epoch = -1
    return self.continue_epochs(self.min_epochs)

  def continue_epochs(self, min_epochs):    
    self.min_epochs = min_epochs
    for epoch in range(self.current_epoch + 1, self.min_epochs):
      self.current_epoch = epoch
      epoch_scores = []
      for idx, arr in enumerate(self.arrays):
        if not self.replacement and idx in self.indexes: 
          epoch_scores.append(-1)
          continue
        if self.replacement and self.max_replacements > 0 and \
            self.indexes.count(idx) >= self.max_replacements: 
          epoch_scores.append(-1)
          continue

        idx_ensemble = self.ensemble[:] + [arr]        
        preds = np.mean(idx_ensemble, 0)
        score = self.scorer(self.y, preds)
        epoch_scores.append(score)

      epoch_index = self.best_chooser(epoch_scores, self.max_score, self.indexes[:]) if self.best_chooser is not None else np.argmax(epoch_scores)
      epoch_score = epoch_scores[epoch_index] if epoch_index >= 0 else -999
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
        else:
          print 'no possible improvement found, exiting early'
          break
    
    print 'fit done indexes selected: ', self.indexes, 'self.max_score:', self.max_score
    return self

  def transform(self, test_files):
    test_arrays = self._get_files(test_files)
    test_ensemble = [test_arrays[i] for i in self.indexes]
    return np.mean(test_ensemble, 0)
