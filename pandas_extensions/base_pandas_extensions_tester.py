import unittest, random
import pandas as pd, numpy as np

class BasePandasExtensionsTester(unittest.TestCase):
  def setUp(self): 
    random.seed(0)
    np.random.seed(0) 

  def close(self, o1, o2):
    if type(o1) is dict: o1 = pd.DataFrame(o1)
    if type(o2) is dict: o2 = pd.DataFrame(o2)
    if type(o1) is dict: o1 = pd.DataFrame(o1)
    if type(o2) is dict: o2 = pd.DataFrame(o2)
    if hasattr(o1, 'values'): o1 = o1.values
    if hasattr(o2, 'values'): o2 = o2.values
    if not isinstance(o1, np.ndarray): np.array(o1)
    if not isinstance(o2, np.ndarray): np.array(o2)
    np.testing.assert_almost_equal(o1, o2, 3)

  def eq(self, o1, o2):
    if type(o1) is dict: o1 = pd.DataFrame(o1)
    if type(o2) is dict: o2 = pd.DataFrame(o2)
    if hasattr(o1, 'values'): o1 = o1.values
    if hasattr(o2, 'values'): o2 = o2.values
    if not isinstance(o1, np.ndarray): np.array(o1)
    if not isinstance(o2, np.ndarray): np.array(o2)
    np.testing.assert_array_equal(o1, o2)