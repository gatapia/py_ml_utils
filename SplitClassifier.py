from sklearn import *
import pandas as pd

class SplitClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
  def __init__(self, clf, splitter):    
    self.clf1 = sklearn.base.clone(clf)
    self.clf2 = sklearn.base.clone(clf)
    self.splitter = splitter

  def fit(self, X, y):
    mask = X.apply(self.splitter, axis=1)
    X1, y1 = X[mask], y[mask]
    X2, y2 = X[~mask], y[~mask]
    self.clf1.fit(X1, y1)
    self.clf2.fit(X2, y2)


  def predict_proba(self, X): 
    def predicter(row):
      is1 = self.splitter(row)
      clf = self.clf1 if is1 else clf2
      return clf.predict_proba(row)

    return X.apply(predicter, axis=1)

