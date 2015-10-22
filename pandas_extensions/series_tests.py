import unittest, sklearn, datetime
import pandas as pd, numpy as np
from . import *
from . import base_pandas_extensions_tester

class T(base_pandas_extensions_tester.BasePandasExtensionsTester):
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
    self.eq(s2, ['(0.998, 2]', '(0.998, 2]', '(2, 3]'])

  def test_group_rare(self):
    s = pd.Series(['a', 'b', 'c'] * 100 + ['d', 'e', 'f'] * 10)    
    s2 = s.copy().group_rare()
    self.eq(s2, ['a', 'b', 'c'] * 100 + ['rare'] * 30)

    s2 = s.copy().group_rare(5)
    self.eq(s2, ['a', 'b', 'c'] * 100 + ['d', 'e', 'f'] * 10)

  def test_sigma_limits(self):
    min_v, max_v = pd.Series(np.random.normal(size=10000)).sigma_limits(2)
    self.assertAlmostEquals(min_v, -2., 1)
    self.assertAlmostEquals(max_v, 2., 1)

  def test_to_indexes(self):
    s = pd.Series(['a', 'b', 'c', 'a'])
    s2 = s.toidxs()
    self.eq(s2, [0, 1, 2, 0])

  def test_append_bottom(self):
    self.eq(pd.Series(['a', 'b', 'c', 'a']).append_bottom(['a']),
      ['a', 'b', 'c', 'a', 'a'])
    self.eq(pd.Series(['a', 'b', 'c', 'a']).append_bottom(np.array(['a'])),
      ['a', 'b', 'c', 'a', 'a'])
    self.eq(pd.Series(['a', 'b', 'c', 'a']).append_bottom(pd.Series(['a'])),
      ['a', 'b', 'c', 'a', 'a'])

  def test_missing(self):
    self.eq(pd.Series(['a', 'b', None, 'a'], name='c_col').missing('mode'),
      ['a', 'b', 'a', 'a'])
    self.eq(pd.Series(['a', 'b', None, 'a'], name='c_col').missing('NA'),
      ['a', 'b', 'NA', 'a'])

    self.eq(pd.Series([1, 2, None, 1], name='n_col').missing('mode'),
      [1, 2, 1, 1])
    self.eq(pd.Series([1, 2, None, 1], name='n_col').missing('mean'),
      [1, 2, 4./3, 1])
    self.eq(pd.Series([1, 2, None, 1], name='n_col').missing('median'),
      [1, 2, 1, 1])
    self.eq(pd.Series([1, 2, None, 1], name='n_col').missing('min'),
      [1, 2, 1, 1])
    self.eq(pd.Series([1, 2, None, 1], name='n_col').missing('max'),
      [1, 2, 2, 1])
    self.eq(pd.Series([1, 2, None, 1], name='n_col').missing('max+1'),
      [1, 2, 3, 1])

  def test_missing_with_iqm(self):
    s = pd.Series(np.random.random(100))
    s[50] = np.nan
    s = s.missing('iqm')
    self.close(s[50], 0.4451432058306285)

  def test_scale(self):
    self.eq(pd.Series([1, 2, 3]).scale(), [-1, 0, 1])    
    self.eq(pd.Series([1, 2, 3]).scale((0, 100)), [0, 50, 100])    

  def test_normalise(self):
    self.eq(pd.Series([1, 2, 3]).normalise(), [0, .5, 1])

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
    self.assertTrue(pd.Series(name='i_col').is_index())
    self.assertTrue(pd.Series(name='i_col').is_categorical_like())
    self.assertTrue(pd.Series(name='n_col').is_numerical())
    self.assertTrue(pd.Series(name='d_col').is_date())

    self.assertTrue(pd.Series(['a', 'b', 'c']).is_categorical())
    self.assertTrue(pd.Series(['a', 'b']).is_binary())
    self.assertTrue(pd.Series([1, 2, 3]).is_index())
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

  def test_categorical_outliers_with_new_value(self):
    cols = ['a', 'b', 'c', 'd'] * 100000 + ['f', 'g'] * 10000
    s = pd.Series(cols)
    s.categorical_outliers(0.1, 'others')

    exp = ['a', 'b', 'c', 'd'] * 100000 + ['others', 'others'] * 10000
    self.eq(exp, s)

  def test_categorical_outliers_with_mode(self):
    cols = ['a', 'b', 'c', 'd'] * 100000 + ['d', 'f', 'g'] * 10000
    s = pd.Series(cols)
    s.categorical_outliers(0.1, 'mode')
    
    exp = ['a', 'b', 'c', 'd'] * 100000 + ['d', 'd', 'd'] * 10000
    self.eq(exp, s)

  def test_compress_size_with_0_aggresiveness(self):
    s = pd.Series(np.random.normal(size=10000))
    self.assertEquals(str(s.dtype), 'float64')
    s2 = s.compress_size(aggresiveness=0)
    self.assertEquals(str(s2.dtype), 'float64')
    self.eq(s, s2)

  def test_compress_size_with_1_aggresiveness(self):
    s = pd.Series(np.random.normal(size=10000))
    self.assertEquals(str(s.dtype), 'float64')
    s2 = s.compress_size(aggresiveness=1)
    self.assertEquals(str(s2.dtype), 'float32')
    self.assertTrue(s.all_close(s2))

  def test_compress_size_with_2_aggresiveness(self):
    s = pd.Series(np.random.normal(size=10000))
    self.assertEquals(str(s.dtype), 'float64')
    s2 = s.compress_size(aggresiveness=2)
    self.assertEquals(str(s2.dtype), 'float16')
    self.assertTrue(s.all_close(s2, .01))

  def test_hashcode_float(self):
    np.random.seed(0)
    s = pd.Series(np.random.normal(size=10000))
    h1 = s.hashcode()
    np.random.seed(0)
    s1 = pd.Series(np.random.normal(size=10000))
    h11 = s.hashcode()
    np.random.seed(1)
    s2 = pd.Series(np.random.normal(size=10000))
    h2 = s2.hashcode()

    self.assertNotEqual(h1, h2)
    self.assertEqual(h1, h11)

  def test_hashcode_categorical(self):
    s = pd.Series(['a', 'b', 'c'])
    h1 = s.hashcode()
    s1 = pd.Series(['a', 'b', 'c'])
    h11 = s.hashcode()
    s2 = pd.Series(['a', 'b', 'd'])
    h2 = s2.hashcode()

    self.assertNotEqual(h1, h2)
    self.assertEqual(h1, h11)

  def test_add_noise(self):
    s = pd.Series(np.random.normal(size=10000))
    s2 = s.add_noise(.001)
    self.assertFalse(s.is_equals(s2))
    self.assertTrue(s.all_close(s2, .01))

  def test_add_causian(self):
    s = pd.Series(np.random.normal(size=10000))
    s2 = s.add_noise(.001, mode='gaussian')
    self.assertFalse(s.is_equals(s2))
    self.assertTrue(s.all_close(s2, .01))

  def test_to_count_of_binary_target(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    s.to_count_of_binary_target([0, 1, 1, 0, 1, 0])
    self.eq(s, [2, 2, 2, 1, 1., 0.])

  def test_to_ratio_of_binary_target(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    s.to_ratio_of_binary_target([0, 1, 1, 0, 1, 0])
    self.eq(s, [2/3., 2/3., 2/3., 1/2., 1/2., 0.])

  def test_to_count_of_samples(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    s.to_count_of_samples()
    self.eq(s, [3, 3, 3, 2, 2, 1])

  def test_to_ratio_of_samples(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    s.to_ratio_of_samples()
    self.eq(s, [1/2., 1/2., 1/2., 1/3., 1/3., 1/6.])  

  def test_to_stat(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.]), [2, 2, 2, 4.5, 4.5, 6])  

    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'median'), [2, 2, 2, 4.5, 4.5, 6])  

    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'min'), [1, 1, 1, 4, 4, 6])  

    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'max'), [3, 3, 3, 5, 5, 6])  

  def test_to_stat_iqm(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'])
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'iqm'), [2. ,  2. ,  2. ,  4.5,  4.5,  6.])  

  def test_to_stat_with_test(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'] * 2)
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.]), 
        [2, 2, 2, 4.5, 4.5, 6] * 2)  
    
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'] * 2)
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'median'), [2, 2, 2, 4.5, 4.5, 6] * 2)

    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'] * 2)
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'min'), [1, 1, 1, 4, 4, 6] * 2)

    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'] * 2)
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'max'), [3, 3, 3, 5, 5, 6] * 2)

    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'] * 2)
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.], 'iqm'), [2. ,  2. ,  2. ,  4.5,  4.5,  6.] * 2)  

  def test_to_stat_with_test_with_missing_vals(self):
    s = pd.Series(['a', 'a', 'a', 'b', 'b', 'c'] * 2 + ['d', 'd'])
    self.eq(s.to_stat([1., 2., 3., 4., 5., 6.]), 
        [2, 2, 2, 4.5, 4.5, 6] * 2 + [3.5, 3.5])  
    
  def test_to_stat_larger_data(self):
    s = pd.Series(np.random.random(100000) * 10).astype(int)
    y = np.random.random(100000)
    self.assertEqual(s.copy().to_stat(y, 'mean').mean(), 0.5022836997232796)
    self.assertEqual(s.copy().to_stat(y, 'median').mean(), 0.5043951662699959)
    self.assertEqual(s.copy().to_stat(y, 'min').mean(), 6.607534861049408e-05)
    self.assertEqual(s.copy().to_stat(y, 'max').mean(), 0.9999608457126161)

    s = pd.Series(np.random.random(100000) * 20).astype(int)
    y = np.random.random(100000)
    self.assertEqual(s.copy().to_stat(y, 'mean').mean(), 0.4985913593313108)
    self.assertEqual(s.copy().to_stat(y, 'median').mean(), 0.4975387552531616)
    self.assertEqual(s.copy().to_stat(y, 'min').mean(), 0.00020554116883401405)
    self.assertEqual(s.copy().to_stat(y, 'max').mean(), 0.999710657683425)

  def test_outliers(self):
    s = pd.Series(np.random.normal(size=200))
    min_1, max_1 = s.min(), s.max()
    s = s.outliers(2)
    min_2, max_2 = s.min(), s.max()
    self.assertTrue(min_1 < min_2)
    self.assertTrue(max_1 > max_2)

  def test_rank(self):
    s = pd.Series([5, 2, 7, 3, 5, 8, 1])
    rank = s.to_rank(False)
    self.eq(rank, [4.5, 2., 6., 3., 4.5, 7., 1.])
    
    rank = s.to_rank(True)
    self.close(rank, [0.583333,0.166667,0.833333,0.333333,0.583333,1.,0.])
  
  def test_boxcox(self):
    s = pd.Series([1, 2, 3, 4, 5])
    bc = s.boxcox()
    self.close(bc, [0.,0.88891531,1.64391666,2.32328256,2.95143041])

  def test_boxcox_with_negatives(self):
    s = pd.Series([1, 2, 3, 4, -1])
    bc = s.boxcox()
    self.close(bc, [2.151, 3.301,4.484,5.694, 0.])
  
  def test_floats_to_ints(self):
    s = pd.Series(np.random.normal(size=100))
    s = s.floats_to_ints()
    self.assertEqual(str(s.dtype), 'int32')
    self.assertEqual(s.mean(), 5980.74)

  def test_percentage_positive(self):
    s = pd.Series(np.random.normal(size=99) * 10)
    self.assertEqual(0.18181818181818182, (s > 10).percentage_positive())