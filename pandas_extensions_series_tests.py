import unittest, sklearn, datetime
import pandas as pd, numpy as np
from pandas_extensions import *

class T(unittest.TestCase):
  def test_one_hot_encode(self):
    s = pd.Series([1, 2, 3])
    s2 = s.one_hot_encode().todense()
    np.testing.assert_array_equal(s2, np.array([
      [1., 0., 0.], 
      [0., 1., 0.], 
      [0., 0., 1.]], 'object'))

  def test_binning(self):
    s = pd.Series([1., 2., 3.])    
    s2 = s.bin(2)
    self._eq(s2, ['(0.998, 2]', '(0.998, 2]', '(2, 3]'])

  def test_sigma_limits(self):
    min_v, max_v = pd.Series(np.random.normal(size=10000)).sigma_limits(2)
    self.assertAlmostEquals(min_v, -2., 1)
    self.assertAlmostEquals(max_v, 2., 1)

  def test_to_indexes(self):
    s = pd.Series(['a', 'b', 'c', 'a'])
    s2 = s.toidxs()
    self._eq(s2, [0, 1, 2, 0])

  def test_append_bottom(self):
    self._eq(pd.Series(['a', 'b', 'c', 'a']).append_bottom(['a']),
      ['a', 'b', 'c', 'a', 'a'])
    self._eq(pd.Series(['a', 'b', 'c', 'a']).append_bottom(np.array(['a'])),
      ['a', 'b', 'c', 'a', 'a'])
    self._eq(pd.Series(['a', 'b', 'c', 'a']).append_bottom(pd.Series(['a'])),
      ['a', 'b', 'c', 'a', 'a'])

  def test_missing(self):
    self._eq(pd.Series(['a', 'b', None, 'a'], name='c_col').missing('mode'),
      ['a', 'b', 'a', 'a'])
    self._eq(pd.Series(['a', 'b', None, 'a'], name='c_col').missing('NA'),
      ['a', 'b', 'NA', 'a'])

    self._eq(pd.Series([1, 2, None, 1], name='n_col').missing('mode'),
      [1, 2, 1, 1])
    self._eq(pd.Series([1, 2, None, 1], name='n_col').missing('mean'),
      [1, 2, 4./3, 1])
    self._eq(pd.Series([1, 2, None, 1], name='n_col').missing('median'),
      [1, 2, 1, 1])
    self._eq(pd.Series([1, 2, None, 1], name='n_col').missing('min'),
      [1, 2, 1, 1])
    self._eq(pd.Series([1, 2, None, 1], name='n_col').missing('max'),
      [1, 2, 2, 1])
    self._eq(pd.Series([1, 2, None, 1], name='n_col').missing('max+1'),
      [1, 2, 3, 1])

  def test_scale(self):
    self._eq(pd.Series([1, 2, 3]).scale(), [-1, 0, 1])    
    self._eq(pd.Series([1, 2, 3]).scale((0, 100)), [0, 50, 100])    

  def test_is_equals(self):
    self.assertTrue(pd.Series([1, 2, 3]).is_equals([1, 2, 3]))
    self.assertFalse(pd.Series([1, 2, 3]).is_equals([1, 2, 3.1]))

  def test_all_close(self):
    self.assertTrue(pd.Series([1, 2, 3]).all_close([1, 2, 3]))
    self.assertTrue(pd.Series([1, 2, 3]).all_close([1, 2, 3.00001]))
    self.assertFalse(pd.Series([1, 2, 3]).all_close([1, 2, 3.1]))

  def test_types_identification(self):
    self.assertTrue(pd.Series(name='c_col').is_categorical())
    self.assertTrue(pd.Series(name='c_col').is_categorical_like())
    self.assertTrue(pd.Series(name='b_col').is_binary())
    self.assertTrue(pd.Series(name='b_col').is_categorical_like())
    self.assertTrue(pd.Series(name='i_col').is_indexes())
    self.assertTrue(pd.Series(name='i_col').is_categorical_like())
    self.assertTrue(pd.Series(name='n_col').is_numerical())
    self.assertTrue(pd.Series(name='d_col').is_date())

    self.assertTrue(pd.Series(['a', 'b', 'c']).is_categorical())
    self.assertTrue(pd.Series(['a', 'b']).is_binary())
    self.assertTrue(pd.Series([1, 2, 3]).is_indexes())
    self.assertTrue(pd.Series(np.random.normal(size=10000)).is_numerical())
    self.assertTrue(pd.Series([datetime.datetime(2010, 1, 1)]).is_date())

  def test_is_valid_name(self):
    self.assertTrue(pd.Series(name='c_col').is_valid_name())
    self.assertTrue(pd.Series(name='b_col').is_valid_name())
    self.assertTrue(pd.Series(name='i_col').is_valid_name())
    self.assertTrue(pd.Series(name='n_col').is_valid_name())
    self.assertTrue(pd.Series(name='d_col').is_valid_name())

    self.assertFalse(pd.Series(name='x_col').is_valid_name())
    self.assertFalse(pd.Series(name='y_col').is_valid_name())
    self.assertFalse(pd.Series(name='random').is_valid_name())
    self.assertFalse(pd.Series().is_valid_name())

  def test_infer_name(self):
    self.assertEqual('c_col', pd.Series(['a', 'b', 'c'], name='col').infer_col_name().name)
    self.assertEqual('b_col', pd.Series(['a', 'b'], name='col').infer_col_name().name)
    self.assertEqual('b_col', pd.Series([10, 100], name='col').infer_col_name().name)
    self.assertEqual('n_col', pd.Series(np.random.normal(size=10000), name='col').infer_col_name().name)
    self.assertEqual('d_col', pd.Series([datetime.datetime(2010, 1, 1)], name='col').infer_col_name().name)


  def _eq(self, s1, s2):
    if hasattr(s1, 'values'): s1 = s1.values
    if hasattr(s2, 'values'): s2 = s2.values
    if not isinstance(s1, np.ndarray): np.array(s1)
    if not isinstance(s2, np.ndarray): np.array(s2)
    np.testing.assert_array_equal(s1, s2)