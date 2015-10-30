import sklearn, pandas as pd, numpy as np

class SplitClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
  def __init__(self, classifiers, get_split_index, fallback_classifier=None):    
    '''
    classifiers: A classifier or list of classifiers for each split.  If only
      one classifier is given then the same classifier is used to train/predict
      all splits
    get_split_index: a callable that takes a list or data frame and returns
      a list of split ids.  The split ids returned should be integers from 
      0 to the number of splits.  This id is actually used to index the 
      classifier for training and testing.
    fallback_classifier: if any entry in the test set is not indexed 
      (-1) from get_split_index then this fallback classifier is used
      to predict these values.  Note this classifier must already by 
      trained as the fit method will not be called.
    '''
    self.classifiers = classifiers
    self.get_split_index = get_split_index
    self.fallback_classifier = fallback_classifier

  def fit(self, X, y):
    split_indexes = np.array(self.get_split_index(X))
    unique_indexes = np.unique(split_indexes)
    unique_indexes = unique_indexes[unique_indexes>=0]
    if not isinstance(self.classifiers, (list, tuple)):
      self.classifiers = [self.classifiers] * len(unique_indexes)
    elif len(unique_indexes) != len(self.classifiers):
      raise Exception('got ' + str(len(unique_indexes)) + ' splits but ' + str(len(self.classifiers)) + ' classifiers')
    elif np.any(unique_indexes != np.arange(len(unique_indexes))):
      raise Exception('get_split_index should return zero indexed ')
    
    self.classifiers = [sklearn.base.clone(clf) for clf in self.classifiers]    
    for idx in unique_indexes:
      mask = split_indexes == idx
      X_split, y_split = X[mask], y[mask]
      if hasattr(X_split, 'values'): X_split = X_split.values
      if hasattr(y_split, 'values'): y_split = y_split.values
      self.classifiers[idx].fit(X_split, y_split)

    return self

  def transform(self, X): return self.predict_impl(X, 'transform')
  def predict(self, X): return self.predict_impl(X, 'predict')
  def predict_proba(self, X): return self.predict_impl(X, 'predict_proba')

  def predict_impl(self, X, method):
    split_indexes = np.array(self.get_split_index(X))
    unique_indexes = np.unique(split_indexes)
    unique_indexes = unique_indexes[unique_indexes>=0]
    results = None

    if len(unique_indexes) != len(self.classifiers):
      raise Exception('got ' + str(len(unique_indexes)) + ' splits but ' + str(len(self.classifiers)) + ' classifiers')

    remaining = np.ones(X.shape[0])
    for idx in unique_indexes:
      mask = split_indexes == idx
      remaining[mask] = 0
      X_split = X[mask]
      if hasattr(X_split, 'values'): X_split = X_split.values
      predictions = getattr(self.classifiers[idx], method)(X_split)
      if results is None:
        predictions_width = 1 if len(predictions.shape) == 1 else predictions.shape[1]
        results = np.ndarray(shape=X.shape[0] if predictions_width == 1 else (X.shape[0], predictions_width))
      results[mask] = predictions
    
    if np.any(remaining==1):
      if self.fallback_classifier is None: raise Exception('Had test entries that were not indexed but no fallback_classifier was provided')
      results[remaining==1] = getattr(self.fallback_classifier, method)(X[remaining==1])
    return results
