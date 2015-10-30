import unittest, sklearn, sklearn.dummy, numpy as np
from SplitClassifier import SplitClassifier

class T(unittest.TestCase):
  def test_split_classifier_with_single_classifier(self):
    c = sklearn.dummy.DummyClassifier('constant', constant=0)    
    sc = SplitClassifier(c, lambda X: np.arange(len(X)) % 2)
    X = np.ones(shape=(100, 3))
    y = np.zeros(100)
    X_test = np.ones(shape=(100, 3))
    
    sc.fit(X, y)
    sc.classifiers[1].constant = 1

    predictions = sc.predict(X_test)
    self.assertEquals((100,), predictions.shape)
    np.testing.assert_array_equal(np.arange(100) % 2, predictions)


  def test_split_classifier_with_multiple_classifier(self):
    c0 = sklearn.dummy.DummyClassifier('constant', constant=0)    
    c1 = sklearn.dummy.DummyClassifier('constant', constant=1)    
    sc = SplitClassifier((c0, c1), lambda X: np.arange(len(X)) % 2)
    X = np.ones(shape=(100, 3))
    y = np.arange(100) % 2
    X_test = np.ones(shape=(100, 3))
    
    sc.fit(X, y)

    predictions = sc.predict(X_test)
    self.assertEquals((100,), predictions.shape)
    np.testing.assert_array_equal(np.arange(100) % 2, predictions)

  def test_split_classifier_with_3_indexes(self):
    c0 = sklearn.dummy.DummyClassifier('constant', constant=0)    
    c1 = sklearn.dummy.DummyClassifier('constant', constant=1)    
    c2 = sklearn.dummy.DummyClassifier('constant', constant=2)    
    sc = SplitClassifier((c0, c1, c2), lambda X: np.arange(len(X)) % 3)
    X = np.ones(shape=(100, 3))
    y = np.arange(100) % 3
    X_test = np.ones(shape=(100, 3))
    
    sc.fit(X, y)
    predictions = sc.predict(X_test)
    self.assertEquals((100,), predictions.shape)
    np.testing.assert_array_equal(np.arange(100) % 3, predictions)

  
  def test_split_classifier_with_fallback_classifier(self):
    c0 = sklearn.dummy.DummyClassifier('constant', constant=0)    
    c1 = sklearn.dummy.DummyClassifier('constant', constant=1)    
    fallback = sklearn.dummy.DummyClassifier('constant', constant=999)    
    
    def indexer(X):
      indexes = np.arange(len(X)) % 3
      indexes[indexes==2] = -1
      return indexes

    sc = SplitClassifier((c0, c1), indexer, fallback_classifier=fallback)
    X = np.ones(shape=(100, 3))
    y = np.arange(100) % 3    
    X_test = np.ones(shape=(100, 3))
    
    fallback.fit(X, np.array([999] * 100))
    sc.fit(X, y)
    predictions = sc.predict(X_test)
    self.assertEquals((100,), predictions.shape)
    expected = np.arange(100) % 3
    expected[expected==2] = 999
    np.testing.assert_array_equal(expected, predictions)