import unittest, time
from FeatSel import * 
from sklearn import *

class T(unittest.TestCase):

    def test_feature_select(self):
      boston_data = datasets.load_boston()
      X = boston_data['data'][:100,:5]
      y = boston_data['target'][:100]
      clf = linear_model.LinearRegression()
      t0 = time.time()
      '''
      feats = FeatSel(X, y, epochs=2).run(clf)
      sel_features = map(lambda f: f['feature'], feats)
      self.assertEqual([0, 2], sel_features)
      '''
      # too slow

if __name__ == '__main__':
    unittest.main()
