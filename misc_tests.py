import unittest
import numpy as np
from misc import * 
from sklearn import linear_model
from pandas_extensions import base_pandas_extensions_tester

class T(base_pandas_extensions_tester.BasePandasExtensionsTester):

  def test_cv(self):
    c = linear_model.LinearRegression()
    X = np.random.random(size=(100, 3))
    y = np.random.random(size=100)
    cv = do_cv(c, X, y)      

  def test_gs(self):
    c = linear_model.LinearRegression()
    X = np.random.random(size=(100, 3))
    y = np.random.random(size=100)
    gs = do_gs(c, X, y, {'fit_intercept': [True, False]}, n_jobs=1)
    self.assertEqual(gs.best_params_, {'fit_intercept': True})

  def test_start_stop(self):
    start('msg', 'id')
    msg = stop('msg', 'id')
    self.assertTrue(msg.startswith('msg, took: 0:00'))

    start('msg1', 'id1')
    start('msg2', 'id2')
    msg1 = stop('msg1', 'id1')
    msg2 = stop('msg2', 'id2')
    self.assertTrue(msg1.startswith('msg1, took: 0:00'))
    self.assertTrue(msg2.startswith('msg2, took: 0:00'))
