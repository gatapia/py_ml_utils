from sklearn import cross_validation
import numpy as np

class KLabelFolds():
  def __init__(self, labels, n_folds=3):
    self.labels = labels
    self.n_folds = n_folds
    
  def __iter__(self):
    unique_labels = self.labels.unique()
    cv = cross_validation.KFold(len(unique_labels), self.n_folds)
    for train, test in cv:
      test_labels = unique_labels[test]
      test_mask = self.labels.isin(test_labels)
      train_mask = np.logical_not(test_mask)
      yield (np.where(train_mask)[0], np.where(test_mask)[0])
