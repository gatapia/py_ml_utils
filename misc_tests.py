import unittest
import numpy as np
from misc import * 
from sklearn import linear_model

class T(unittest.TestCase):

  def test_mean_score(self):
    self.assertEqual('2.00000 (+/-0.57735)', mean_score([1., 2., 3.]))

  def test_scale(self):
    arr = np.linspace(10, 100, 5)
    arr_scaled = scale(arr)
    exp = [-1.4142, -0.7071,  0., 0.7071,  1.4142]
    self.assertTrue(np.allclose(exp, arr_scaled))

  def test_scale_with_min_max(self):
    scaled = scale(np.matrix('1. 2.; 3. 4.'), (0, 1))
    exp = np.matrix('0. 0.; 1. 1.')
    self.assertTrue(np.allclose(exp, scaled))

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
    gs = do_gs(c, X, y, {'normalize': [True, False]}, n_jobs=1)
    self.assertEqual(gs.best_params_, {'normalize': False})

  def test_one_hot_encode(self):
    df = pd.DataFrame({'col0': [1.1, 1.2, 1.3, 1.4, 1.5], 'col1':[1, 2, 3, 3, 1], 'col2': ['a', 'b', 'c', 'd', 'e']})
    df2 = one_hot_encode(df, [1], drop_originals=True)
    self.assertEqual(5, df2.shape[1])
    exp = [[1., 0., 0.],[0., 1., 0.],[0., 0., 1.],[0., 0., 1.],[1., 0., 0.]]
    self.assertTrue((np.array(exp) == df2[:, 2:]).all())

  def test_to_index(self):
    df = pd.DataFrame({'col0': [1.1, 1.2, 1.3, 1.4, 1.5], 'col1':[1, 2, 3, 3, 1], 'col2': ['a', 'b', 'c', 'd', 'e']})
    df2 = to_index(df, [1], drop_originals=True)
    self.assertEqual(3, df2.shape[1])
    exp = [0, 1, 2, 2, 0]
    self.assertTrue((df2.col1_indexes.values == exp).all())

  def test_save_data(self):    
    pass

  def test_read_data(self):
    pass

  def test_save_data_gzip(self):    
    pass

  def test_read_data_gzip(self):
    pass

if __name__ == '__main__':
  unittest.main()
