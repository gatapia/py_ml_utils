import unittest
import numpy as np
from misc import * 
from sklearn import linear_model

class T(unittest.TestCase):

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
    self.assertEqual(gs.best_params_, {'normalize': True})
