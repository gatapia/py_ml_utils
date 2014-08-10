import unittest, time
from feature_sel import * 
from sklearn import *

class TestFeatureSelect(unittest.TestCase):

    def test_feature_select(self):
      boston_data = datasets.load_boston()
      X = boston_data['data']
      y = boston_data['target']
      clf = linear_model.LinearRegression()
      t0 = time.time()
      feats = feature_select(clf, X, y)
      print 'took:', time.time() - t0
      sel_features = map(lambda f: f['feature'], feats)
      # Takes 0.0429999828339 seconds
      self.assertTrue([12, 5, 3, 7, 4, 1, 11] == sel_features)

if __name__ == '__main__':
    unittest.main()
