from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class GreedyEnsembler(BaseEstimator, TransformerMixin):  
  def __init__(self, scorer, min_epochs=10):
    self.scorer = scorer
    self.min_epochs = min_epochs
    self.indexes = []

  def _get_files(self, files):
    if hasattr(files, '__call__'): return files()
    first = files[0]
    if type(first) is str:      
      return [load(f) for f in files]
    return files

  def fit(self, train_files, y):    
    arrays = self._get_files(train_files)
    scores = [self.scorer(y, arr) for arr in arrays]
    max_score = np.max(scores)
    ensemble = [arrays[np.argmax(scores)]]
    self.indexes = [np.argmax(scores)]
    print 'starting score:', max_score

    for epoch in range(self.min_epochs):      
      epoch_improved = False
      epoch_score = -999
      epoch_index = -1
      for idx, arr in enumerate(arrays):
        idx_ensemble = ensemble[:] + [arr]
        score = self.scorer(y, np.mean(ensemble, 0))
        if score >= epoch_score:
          epoch_score = score
          epoch_index = idx

        if score >= max_score:
          epoch_improved = True
          ensemble = idx_ensemble
          self.indexes.append(idx)
          max_score = score
          print 'epoch:', epoch, 'found improved score:', max_score, 'index:', idx, 'ensemble size:', len(ensemble)
      
      if not epoch_improved:        
        idx_ensemble = ensemble[:] + [arrays[epoch_index]]
        score = self.scorer(y, np.mean(ensemble, 0))
        ensemble = idx_ensemble
        self.indexes.append(idx)
        max_score = score   
        print 'no improvement found after ', epoch, 'min_epochs, picking best:', max_score     
    
    print 'fit done indexes selected: ', self.indexes
    return self

  def transform(self, test_files):
    arrays = self._get_files(test_files)
    ensemble = [arrays[i] for i in self.indexes]
    return np.mean(ensemble, 0)
