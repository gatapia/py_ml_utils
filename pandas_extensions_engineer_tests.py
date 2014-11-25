import unittest
import pandas as pd
import numpy as np
from pandas_extensions import *
from ast_parser import explain
from sklearn import linear_model, preprocessing

class T(unittest.TestCase):
  def test_engineer_concat(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2': ['d', 'e', 'f']})    
    df.engineer('concat(c_1, c_2)')
    self.assertTrue(np.array_equal(df['c_concat(c_1,c_2)'].values, 
      np.array(['ad', 'be', 'cf'], 'object')))

  def test_engineer_concat_3_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2': ['d', 'e', 'f'], 'c_3': ['h', 'i', 'j']})    
    df.engineer('concat(c_3, c_1, c_2)')
    self.assertTrue(np.array_equal(df['c_concat(c_3,c_1,c_2)'].values, 
      np.array(['had', 'ibe', 'jcf'], 'object')))

  def test_engineer_concat_with_numerical_col(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3]})    
    df.engineer('concat(c_1,n_2)')
    self.assertTrue(np.array_equal(df['c_concat(c_1,n_2)'].values, 
      np.array(['a1', 'b2', 'c3'], 'object')))

  def test_engineer_concat_with_numerical_col_3_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6]})    
    df.engineer('concat(n_3,c_1,n_2)')
    self.assertTrue(np.array_equal(df['c_concat(n_3,c_1,n_2)'].values, 
      np.array(['4a1', '5b2', '6c3'], 'object')))

  def test_engineer_multiplication(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('mult(n_2, n_3)')
    self.assertTrue(np.array_equal(df['n_mult(n_2,n_3)'].values, 
      np.array([4, 10, 18], long)))

  def test_engineer_multiplication_3_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('mult(n_2, n_3, n_4)')
    self.assertTrue(np.array_equal(df['n_mult(n_2,n_3,n_4)'].values, 
      np.array([4*7, 80, 18*9], long)))

  def test_engineer_square_on_whole_data_frame(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('pow(2)')
    np.testing.assert_array_equal(df.values, 
      np.array([
        ['a', 1, 4, 7, 1*1, 4*4, 7*7],
        ['b', 2, 5, 8, 2*2, 5*5, 8*8],
        ['c', 3, 6, 9, 3*3, 6*6, 9*9],
        ], 'object'))

  def test_engineer_square_on_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('pow(n_3, 2)')
    np.testing.assert_array_equal(df.values, 
      np.array([
        ['a', 1, 4, 7, 4*4],
        ['b', 2, 5, 8, 5*5],
        ['c', 3, 6, 9, 6*6],
        ], 'object'))

  def test_engineer_log_on_whole_data_frame(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('lg()')
    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 1, 4, 7, math.log(1), math.log(4), math.log(7)],
        ['b', 2, 5, 8, math.log(2), math.log(5), math.log(8)],
        ['c', 3, 6, 9, math.log(3), math.log(6), math.log(9)],
        ], 'object')))

  def test_engineer_log_on_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('lg(n_3)')
    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 1, 4, 7, math.log(4)],
        ['b', 2, 5, 8, math.log(5)],
        ['c', 3, 6, 9, math.log(6)],
        ], 'object')))

  def test_engineer_sqrt_on_whole_data_frame(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('sqrt()')
    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 1, 4, 7, math.sqrt(1), math.sqrt(4), math.sqrt(7)],
        ['b', 2, 5, 8, math.sqrt(2), math.sqrt(5), math.sqrt(8)],
        ['c', 3, 6, 9, math.sqrt(3), math.sqrt(6), math.sqrt(9)],
        ], 'object')))

  def test_engineer_sqrt_on_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('sqrt(n_3)')
    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 1, 4, 7, math.sqrt(4)],
        ['b', 2, 5, 8, math.sqrt(5)],
        ['c', 3, 6, 9, math.sqrt(6)],
        ], 'object')))

  def test_engineer_rolling_sum_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_sum(n_1,3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 35, 40, 30, 29, 48], df['n_' + col])

  def test_engineer_rolling_mean_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_mean(n_1,3)'
    df.engineer(col)
    np.testing.assert_allclose([np.nan, np.nan, 11.66, 13.33, 10, 9.66, 16], df['n_' + col], rtol=1e-3)

  def test_engineer_rolling_median_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_median(n_1,3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 12, 13, 13, 12, 12], df['n_' + col])

  def test_engineer_rolling_min_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_min(n_1,3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 10, 12, 2, 2, 2], df['n_' + col])

  def test_engineer_rolling_max_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_max(n_1,3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 13, 15, 15, 15, 34], df['n_' + col])

  def test_engineer_rolling_std_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_std(n_1,3)'
    df.engineer(col)
    np.testing.assert_allclose([np.nan, np.nan, 1.528, 1.528, 7, 6.807, 16.371], df['n_' + col], rtol=1e-3)

  def test_engineer_rolling_var_on_single_col(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34]})
    col = 'rolling_var(n_1,3)'
    df.engineer(col)
    np.testing.assert_allclose([np.nan, np.nan, 2.333, 2.333, 49, 46.333, 268], df['n_' + col], rtol=1e-3)

  # Multiple Columns

  def test_engineer_rolling_sum_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_sum(3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 35, 40, 30, 29, 48], df['n_rolling_sum(n_1,3)'])
    np.testing.assert_array_equal([np.nan, np.nan, 6, 10, 10, 9, 8], df['n_rolling_sum(n_2,3)'])

  def test_engineer_rolling_mean_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_mean(3)'
    df.engineer(col)
    np.testing.assert_allclose([np.nan, np.nan, 11.66, 13.33, 10, 9.66, 16], df['n_rolling_mean(n_1,3)'], rtol=1e-3)
    np.testing.assert_allclose([np.nan, np.nan, 2, 3.333, 3.333, 3, 2.666], df['n_rolling_mean(n_2,3)'], rtol=1e-3)

  def test_engineer_rolling_median_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_median(3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 12, 13, 13, 12, 12], df['n_rolling_median(n_1,3)'])
    np.testing.assert_array_equal([np.nan, np.nan, 2, 3, 3, 2, 2], df['n_rolling_median(n_2,3)'])

  def test_engineer_rolling_min_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_min(3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 10, 12, 2, 2, 2], df['n_rolling_min(n_1,3)'])
    np.testing.assert_array_equal([np.nan, np.nan, 1, 2, 2, 2, 2], df['n_rolling_min(n_2,3)'])

  def test_engineer_rolling_max_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_max(3)'
    df.engineer(col)
    np.testing.assert_array_equal([np.nan, np.nan, 13, 15, 15, 15, 34], df['n_rolling_max(n_1,3)'])
    np.testing.assert_array_equal([np.nan, np.nan, 3, 5, 5, 5, 4], df['n_rolling_max(n_2,3)'])

  def test_engineer_rolling_std_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_std(3)'
    df.engineer(col)
    np.testing.assert_allclose([np.nan, np.nan, 1.528, 1.528, 7, 6.807, 16.371], df['n_rolling_std(n_1,3)'], rtol=1e-3)
    np.testing.assert_allclose([np.nan, np.nan, 1, 1.528, 1.528, 1.732, 1.1547], df['n_rolling_std(n_2,3)'], rtol=1e-3)

  def test_engineer_rolling_var_on_multi_cols(self):
    df = pd.DataFrame({'n_1': [10, 12, 13, 15, 2, 12, 34], 'n_2': [1, 2, 3, 5, 2, 2, 4]})
    col = 'rolling_var(3)'
    df.engineer(col)
    np.testing.assert_allclose([np.nan, np.nan, 2.333, 2.333, 49, 46.333, 268], df['n_rolling_var(n_1,3)'], rtol=1e-3)
    np.testing.assert_allclose([np.nan, np.nan, 1, 2.333, 2.333, 3, 1.333], df['n_rolling_var(n_2,3)'], rtol=1e-3)

  def test_engineer_method_chaining(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.\
      engineer('concat(c_1, c_2)').\
      engineer('concat(c_1, n_2)').\
      engineer('mult(n_2, n_3)').\
      engineer('lg(n_2)').\
      engineer('pow(n_3, 2)')

    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 'd', 1, 4, 7, 'ad', 'a1', 4, math.log(1), 4*4],
        ['b', 'e', 2, 5, 8, 'be', 'b2', 10, math.log(2), 5*5],
        ['c', 'f', 3, 6, 9, 'cf', 'c3', 18, math.log(3), 6*6]
        ], 'object')))

  def test_chaining_single_call_semi_col_sep(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('concat(c_1, c_2);concat(c_1, n_2);mult(n_2, n_3);lg(n_2);pow(n_3, 2)')

    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 'd', 1, 4, 7, 'ad', 'a1', 4, math.log(1), 4*4],
        ['b', 'e', 2, 5, 8, 'be', 'b2', 10, math.log(2), 5*5],
        ['c', 'f', 3, 6, 9, 'cf', 'c3', 18, math.log(3), 6*6]
        ], 'object')))

  def test_chaining_single_with_arr_arg(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1, 2, 3], 'n_3': [4, 5, 6], 'n_4': [7, 8, 9]})    
    df.engineer('concat(c_1, c_2);concat(c_1, n_2);mult(n_2, n_3);lg(n_2);pow(n_3, 2)'.split(';'))

    self.assertTrue(np.array_equal(df.values, 
      np.array([
        ['a', 'd', 1, 4, 7, 'ad', 'a1', 4, math.log(1), 4*4],
        ['b', 'e', 2, 5, 8, 'be', 'b2', 10, math.log(2), 5*5],
        ['c', 'f', 3, 6, 9, 'cf', 'c3', 18, math.log(3), 6*6]
        ], 'object')))

  def test_long_method_chains(self):
    df1 = pd.DataFrame({'n_1': [1, 2, 3], 'n_2': [4, 5, 6]})    
    df2 = pd.DataFrame({'n_1': [1, 2, 3], 'n_2': [4, 5, 6]})    
    df1.engineer('mult(lg(mult(n_1, n_2)), lg(pow(n_1, 3)))')
    df2.engineer('mult(n_1,n_2);pow(n_1,3)')
    df2.engineer('lg(pow(n_1,3));lg(mult(n_1, n_2))')
    df2.engineer('mult(lg(mult(n_1,n_2)),lg(pow(n_1, 3)))')

    np.testing.assert_array_equal(df1.columns.values.sort(), df2.columns.values.sort());
    np.testing.assert_array_equal(df1['n_mult(n_1,n_2)'].values, df2['n_mult(n_1,n_2)'].values);
    np.testing.assert_array_equal(df1['n_pow(n_1,3)'], df2['n_pow(n_1,3)']);
    np.testing.assert_array_equal(df1['n_lg(pow(n_1,3))'], df2['n_lg(pow(n_1,3))']);
    np.testing.assert_array_equal(df1['n_lg(mult(n_1,n_2))'], df2['n_lg(mult(n_1,n_2))']);
    np.testing.assert_array_equal(df1['n_mult(lg(mult(n_1,n_2)),lg(pow(n_1,3)))'], df2['n_mult(lg(mult(n_1,n_2)),lg(pow(n_1,3)))']);

if __name__ == '__main__':
  unittest.main()
