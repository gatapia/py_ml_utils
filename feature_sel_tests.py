import unittest, time
import numpy as np
from misc import * 
from sklearn import linear_model

class TestFeatureSelect(unittest.TestCase):

    def test_feature_select(self):
      boston_data = datasets.load_boston()
      X = boston_data['data']
      y = boston_data['target']
      clf = linear_model.LinearRegression()
      print 'original:', X.shape[1]
      t0 = time.time()
      feats = feature_select(clf, X, y)
      print 'feats:', feats
      print 'took:', time.time() - t0
      # Takes 0.0429999828339 seconds
      self.assertTrue([12, 5, 3, 7, 4, 1, 11] == feats[0])

if __name__ == '__main__':
    unittest.main()
