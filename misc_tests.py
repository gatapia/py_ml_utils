import unittest
import numpy as np
from misc import * 
from sklearn import linear_model

class TestMisc(unittest.TestCase):

    def test_mean_score(self):
      self.assertEqual('Mean: 2.000 (+/-0.577)', mean_score([1., 2., 3.]))

    def test_scale(self):
      arr = np.linspace(10, 100, 5)
      arr_scaled = scale(arr)
      exp = [-1.4142, -0.7071,  0., 0.7071,  1.4142]
      self.assertEqual(True, np.allclose(exp, arr_scaled))

    def test_scale_with_min_max(self):
      scaled = scale(np.matrix('1. 2.; 3. 4.'), (0, 1))
      exp = np.matrix('0. 0.; 1. 1.')
      self.assertEqual(True, np.allclose(exp, scaled))

    def test_do_n_sample_search(self):
      pass

    def test_cv(self):
      c = linear_model.LinearRegression()
      X = np.matrix('1. 1. 1. 1;2. 2. 2. 2;3. 3. 3. 3.')
      y = np.matrix('1. ;2. ;3.')
      cv = do_cv(c, X, y)      

    def test_gs(self):
      c = linear_model.LinearRegression()
      X = np.matrix('1. 1. 1. 1;2. 2. 2. 2;3. 3. 3. 3.')
      y = np.matrix('1. ;2. ;3.')
      cv = do_gs(c, X, y, {})

    def test_save_data(self):    
      pass

    def test_read_data(self):
      pass

    def test_save_data_gzip(self):    
      pass

    def test_read_data_gzip(self):
      pass

    def test_to_index(self):
      pass

if __name__ == '__main__':
    unittest.main()
