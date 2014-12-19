import unittest
import pandas as pd
import numpy as np
from pandas_extensions import *
from sklearn import linear_model, preprocessing

class T(unittest.TestCase):
  
  def test_series_one_hot_encode(self):
    s = pd.Series([1, 2, 3])
    s2 = s.one_hot_encode().todense()
    np.testing.assert_array_equal(s2, np.array([
      [1., 0., 0.], 
      [0., 1., 0.], 
      [0., 0., 1.]], 'object'))

  def test_series_binning(self):
    s = pd.Series([1., 2., 3.])    
    s2 = s.bin(2)
    self.assertTrue(np.array_equal(s2.values, np.array(
      ['(0.998, 2]', '(0.998, 2]', '(2, 3]'], 'object')))

  def test_categoricals(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})
    self.assertTrue(['c_1'] == df.categoricals())

  def test_appending_sparseness_2(self):
    df = pd.DataFrame({'n_1': [1., 2., 3.], 'n_2': [1., 2., 3.]}).to_sparse(fill_value=0)
    df['n_3'] = pd.Series([0, 0, 1]).to_sparse(fill_value=0)

    self.assertTrue(np.array_equal(df.values, np.array([
      [1., 1., 0.], 
      [2., 2., 0.], 
      [3., 3., 1.]], 'object')))

  def test_adding_sparse_col(self):
    df = pd.DataFrame({'n_1': [1., 2., 3.], 'n_2': [1., 2., 3.]})
    df['n_3'] = pd.Series([0, 0, 1]).to_sparse(fill_value=0)

    self.assertTrue(np.array_equal(df.values, np.array([
      [1., 1., 0.], 
      [2., 2., 0.], 
      [3., 3., 1.]], 'object')))

  def test_replacing_w_sparse_col(self):
    df = pd.DataFrame({'n_1': [1., 2., 3.], 'n_2': [1., 2., 3.], 'n_3': [0, 0, 1.]})
    df['n_3'] = pd.Series([0, 0, 1]).to_sparse(fill_value=0)
    
    self.assertTrue(np.array_equal(df.values, np.array([
      [1., 1., 0.], 
      [2., 2., 0.], 
      [3., 3., 1.]], 'object')))

  def test_one_hot_encode(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})
    df = df.one_hot_encode().toarray()    
    self.assertEquals((3, 4), df.shape)
    np.testing.assert_array_equal(df, [
      [1., 0., 0., 1.], 
      [0., 1., 0., 2.], 
      [0., 0., 1., 3.]])

  def test_one_hot_encode_2_cols(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2': ['d', 'e', 'f']})
    df.to_indexes(True)
    df = df.one_hot_encode().toarray()    
    self.assertEquals((3, 6), df.shape)
    np.testing.assert_array_equal(df, [
      [1., 0., 0., 1., 0, 0], 
      [0., 1., 0., 0., 1, 0], 
      [0., 0., 1., 0., 0, 1]])

  def test_one_hot_encode_with_multiple_columns(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 
      'c_2': [2, 2, 2],
      'n_1': [1., 2., 3.], 
      'c_3': ['d', 'e', 'f']})
    df = df.one_hot_encode().toarray()
    self.assertEqual((3, 8), df.shape)
    np.testing.assert_array_equal(df, [
      [1., 0., 0., 1, 1., 0., 0., 1.], 
      [0., 1., 0., 1, 0., 1., 0., 2.], 
      [0., 0., 1., 1, 0., 0., 1., 3.]])

    '''
    np.testing.assert_array_equal(df._one_hot_encoded_columns, 
      ['b_c_1[a]', 'b_c_1[b]', 'b_c_1[c]', 'b_c_2[2]', 
        'b_c_3[d]', 'b_c_3[e]', 'b_c_3[f]', 'n_1'])
    '''

  def test_binning(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})    
    df.bin(2)
    self.assertTrue((3, 3) == df.shape)
    self.assertTrue(np.array_equal(df.values, np.array([
      ['a', 1., '(0.998, 2]'], 
      ['b', 2., '(0.998, 2]'], 
      ['c', 3., '(2, 3]']], 'object')))

  def test_binning_with_remove(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})    
    df.bin(2, True)
    self.assertTrue((3, 2) == df.shape)
    self.assertTrue(np.array_equal(df.values, np.array([
      ['a', '(0.998, 2]'], 
      ['b', '(0.998, 2]'], 
      ['c', '(2, 3]']], 'object')))

  def test_remove_categoricals(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})    
    df.remove(categoricals=True)
    self.assertTrue(['n_1'] == df.columns)

  def test_remove_numericals(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})    
    df.remove(numericals=True)
    self.assertTrue(['c_1'] == df.columns)

  def test_remove_binaries(self):
    df = pd.DataFrame({'b_1':['a', 'b', 'c'], 'd_1': [1., 2., 3.]})    
    df.remove(binaries=True)
    self.assertTrue(['d_1'] == df.columns)

  def test_remove_dates(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'd_1': [1., 2., 3.]})    
    df.remove(dates=True)
    self.assertTrue(['c_1'] == df.columns)

  def test_combinations(self):
    df = pd.DataFrame({'c_1':[], 'c_2':[], 'c_3':[], 
      'n_1': [], 'n_2': [], 'n_3': []})    
    self.assertTrue([('c_1', 'c_2'), ('c_1', 'c_3'), ('c_2', 'c_3')] ==
      df.combinations(2, categoricals=True))

    combs = [('c_1', 'c_2'), ('c_1', 'c_3'), ('c_1', 'n_1'), 
      ('c_1', 'n_2'), ('c_1', 'n_3'), ('c_2', 'c_3'), ('c_2', 'n_1'),
      ('c_2', 'n_2'), ('c_2', 'n_3'), ('c_3', 'n_1'), ('c_3', 'n_2'), 
      ('c_3', 'n_3'), ('n_1', 'n_2'), ('n_1', 'n_3'), ('n_2', 'n_3')]
    self.assertTrue(combs == df.combinations(2, categoricals=True, numericals=True))

  def test_scale(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1., 2., 3.], 'n_3': [4., 5., 6.], 'n_4': [7., 8., 9.]})
    df.scale()
    np.testing.assert_array_equal(df.values, 
      np.array([
        ['a', 'd', -1, -1, -1],
        ['b', 'e', 0, 0, 0],
        ['c', 'f', 1, 1, 1]
        ], 'object'))

    df = pd.DataFrame({'n_2': [1., 2., 3., 4., 5.], 'n_3': [4., 5., 6., 7., 8.]})
    df.scale()
    np.testing.assert_allclose(df.values, 
      [
        [-1.26491106, -1.26491106],
        [-0.63245553, -0.63245553],
        [0, 0],
        [0.63245553, 0.63245553],
        [1.26491106, 1.26491106]
        ], 1e-6)

  def test_scale_with_min_max(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1., 2., 3.], 'n_3': [4., 5., 6.], 'n_4': [7., 8., 9.]})        
    df.scale([], (0., 2.))
    np.testing.assert_array_equal(df.values, 
      np.array([
        ['a', 'd', 0, 0, 0],
        ['b', 'e', 1, 1, 1],
        ['c', 'f', 2, 2, 2]
        ], 'object'))

    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1., 2., 3.], 'n_3': [4., 5., 6.], 'n_4': [7., 8., 9.]})        
    df.scale([], (10., 20.))
    np.testing.assert_array_equal(df.values, 
      np.array([
        ['a', 'd', 10, 10, 10],
        ['b', 'e', 15, 15, 15],
        ['c', 'f', 20, 20, 20]
        ], 'object'))

  def test_missing_vals_in_categoricals_mode(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})        
    df.missing(categorical_fill='mode')
    self.assertEqual('a', df['c_1'][4])

  def test_missing_vals_in_categoricals_mode_multiple_columns(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'c_2':['a', 'b', 'c', 'b', np.nan]})        
    df.missing(categorical_fill='mode')
    self.assertEqual('a', df['c_1'][4])
    self.assertEqual('b', df['c_2'][4])

  def test_missing_vals_in_categoricals_constant(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})        
    df.missing(categorical_fill='f')
    self.assertEqual('f', df['c_1'][4])

  def test_missing_vals_in_categoricals_constant_multiple_columns(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'c_2':['a', 'b', 'c', 'a', np.nan]})        
    df.missing(categorical_fill='f')
    self.assertEqual('f', df['c_1'][4])
    self.assertEqual('f', df['c_2'][4])

  def test_missing_vals_in_numericals_mode(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill='mode')
    self.assertEqual(1, df['n_2'][4])

  def test_missing_vals_in_numericals_mean(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill='mean')
    self.assertEqual(1.75, df['n_2'][4])

  def test_missing_vals_in_numericals_max(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill='max')
    self.assertEqual(3, df['n_2'][4])

  def test_missing_vals_in_numericals_min(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill='min')
    self.assertEqual(1, df['n_2'][4])

  def test_missing_vals_in_numericals_median(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill='median')
    self.assertEqual(1.5, df['n_2'][4])

  def test_missing_vals_in_numericals_constant(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'a', np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill=-10)
    self.assertEqual(-10, df['n_2'][4])

  def test_outliers(self):
    df = pd.DataFrame({'n_1':np.random.normal(size=200)})
    min_1, max_1 = df.n_1.min(), df.n_1.max()
    df.outliers(2)
    min_2, max_2 = df.n_1.min(), df.n_1.max()
    self.assertTrue(min_1 < min_2)
    self.assertTrue(max_1 > max_2)

  def test_missing_vals_in_numericals_mode_multiple_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})              
    df.missing(numerical_fill='mode')
    self.assertEqual(2, df['n_1'][4])
    self.assertEqual(1, df['n_2'][4])

  def test_missing_vals_in_numericals_mean_multiple_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})                          
    df.missing(numerical_fill='mean')
    self.assertEqual(2.75, df['n_1'][4])
    self.assertEqual(1.75, df['n_2'][4])

  def test_missing_vals_in_numericals_max_multiple_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})                         
    df.missing(numerical_fill='max')
    self.assertEqual(4, df['n_1'][4])
    self.assertEqual(3, df['n_2'][4])

  def test_missing_vals_in_numericals_min_multiple_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})                        
    df.missing(numerical_fill='min')
    self.assertEqual(2, df['n_1'][4])
    self.assertEqual(1, df['n_2'][4])

  def test_missing_vals_in_numericals_median_multiple_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})                         
    df.missing(numerical_fill='median')
    self.assertEqual(2.5, df['n_1'][4])
    self.assertEqual(1.5, df['n_2'][4])

  def test_missing_vals_in_numericals_constant_multiple_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 
      'n_2': [1, 2, 3, 1, np.nan]})                         
    df.missing(numerical_fill=-10)
    self.assertEqual(-10, df['n_1'][4])
    self.assertEqual(-10, df['n_2'][4])

  def test_outliers(self):
    df = pd.DataFrame({'n_1':np.random.normal(size=200)})
    min_1, max_1 = df.n_1.min(), df.n_1.max()
    df.outliers(2)
    min_2, max_2 = df.n_1.min(), df.n_1.max()
    self.assertTrue(min_1 < min_2)
    self.assertTrue(max_1 > max_2)

  def test_categorical_outliers_with_new_value(self):
    cols = ['a', 'b', 'c', 'd'] * 100000
    cols = cols + ['f', 'g'] * 10000
    df = pd.DataFrame({'c_1': cols})
    df.categorical_outliers(0.1, 'others')

    exp = ['a', 'b', 'c', 'd'] * 100000 
    exp = exp + ['others', 'others'] * 10000
    np.testing.assert_array_equal(exp, df.c_1.values.tolist())

  def test_categorical_outliers_with_mode(self):
    cols = ['a', 'b', 'c', 'd'] * 100000
    cols = cols + ['d', 'f', 'g'] * 10000
    df = pd.DataFrame({'c_1': cols})
    df.categorical_outliers(0.1, 'mode')
    
    exp = ['a', 'b', 'c', 'd'] * 100000
    exp = exp + ['d', 'd', 'd'] * 10000
    np.testing.assert_array_equal(exp, df.c_1.values.tolist())

  def test_append_right(self):
    df1 = pd.DataFrame({'c_1':['a', 'b'], 
      'n_1': [1, 2]})              
    df2 = pd.DataFrame({'c_2':['c', 'd'], 
      'n_2': [3, 4]})              
    df1 = df1.append_right(df2)
    self.assertTrue(np.array_equal(df1.values, 
      np.array([
        ['a', 1, 'c', 3],
        ['b', 2, 'd', 4]
        ], 'object')))

  def test_append_right_with_sparse(self):
    df1 = pd.DataFrame({'c':[1, 2, 3]})
    arr1 = sparse.coo_matrix([[4], [5], [6]])
    arr2 = df1.append_right(arr1)
    self.assertTrue(type(arr2) is sparse.coo.coo_matrix)
    arr2 = arr2.toarray()
    np.testing.assert_array_equal([[1, 4], [2, 5], [3, 6]], arr2)


  def test_append_bottom(self):
    df1 = pd.DataFrame({'c_1':['a', 'b'], 
      'n_1': [1., 2.]})              
    df2 = pd.DataFrame({'c_1':['c', 'd'], 
      'n_1': [3., 4.]})              
    dfappended = df1.append_bottom(df2)
    np.testing.assert_array_equal( 
      np.array([
        ['a', 1.],
        ['b', 2.],
        ['c', 3.],
        ['d', 4.]
        ], 'object'),
      dfappended.values)

  def test_shuffle(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c', 'd', 'e', 'f', 'g'], 'n_1': [1, 2, 3, 4, 5, 6, 7]})
    y = pd.Series([1L, 2L, 3L, 4L, 5L, 6L, 7L])
    df2, y2 = df.shuffle(y)
    
    # Originals did not change
    np.testing.assert_array_equal(df.values, np.array([['a', 1L], ['b', 2L], ['c', 3L], ['d', 4L], ['e', 5L], ['f', 6L], ['g', 7L]], dtype='object'))
    np.testing.assert_array_equal(y.values, [1, 2, 3, 4, 5, 6, 7])

    # Changed
    np.testing.assert_array_equal(df2.values, np.array([['g', 7L], ['a', 1L], ['d', 4L], ['b', 2L], ['c', 3L], ['e', 5L], ['f', 6L]], dtype='object'))
    np.testing.assert_array_equal(y2.values, [7, 1, 4, 2, 3, 5, 6])

  def test_to_indexes(self):
    df = pd.DataFrame({'c_1':['a', 'b'], 'c_2':['c', 'd'], 'n_1': [1, 2]})
    df.to_indexes(True)
    np.testing.assert_array_equal(df.values, [[1, 0, 0], [2, 1, 1]])

  def test_to_indexes_with_NA(self):
    df = pd.DataFrame({'c_1':['a', 'b', np.nan], 'c_2':['c', 'd', np.nan], 'n_1': [1, 2, 3]})
    df.to_indexes(True)
    np.testing.assert_array_equal(df.values, [[1, 0, 0], [2, 1, 1], [3, 255, 255]])
  
  def test_cv(self):
    df = pd.DataFrame({'n_1': [1, 2, 3, 4, 5, 6, 7]})
    y = pd.Series([1L, 2L, 3L, 4L, 5L, 6L, 7L])        
    df.cv(linear_model.LinearRegression(), y)

  def test_cv_ohe(self):
    df = pd.DataFrame({
      'c_1':['a', 'b', 'c', 'd', 'e', 'f', 'g'] * 10,  
      'n_1': [1, 2, 3, 4, 5, 6, 7] * 10})
    y = pd.Series([1, 0, 0, 1, 1, 0, 1] * 10)        
    df.cv_ohe(linear_model.LogisticRegression(), y)

  def test_pca(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, 2],
        'n_12': [2, 3, 4, 2, 3], 'n_22': [1, 2, 3, 1, 2]})        
    df = df.pca(2)
    np.testing.assert_allclose(df.values, 
      [
        [-1.6,  6.033501e-17],
        [0.4, -2.071991e-16],
        [2.4, -1.136632e-16],
        [-1.6, -3.344294e-16],
        [0.4, -2.071991e-16]
        ], 1e-6)

  def test_pca_with_whitening(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, 2],
        'n_12': [2, 3, 4, 2, 3], 'n_22': [1, 2, 3, 1, 2]})        
    df = df.pca(2, True)
    np.testing.assert_allclose(df.values, 
      [
        [-1.06904497,  0.29145945],
        [ 0.26726124, -1.00091381],
        [ 1.60356745, -0.54907124],
        [-1.06904497, -1.61552322],
        [ 0.26726124, -1.00091381]
        ], 1e-6)

  def test_remove_nas(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, np.nan]})        
    df.rmnas()
    np.testing.assert_array_equal(df.values, [[2, 1], [3, 2], [4, 3], [2, 1]])

    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 'n_2': [1, 2, np.nan, 1, 2]})        
    df.rmnas()
    np.testing.assert_array_equal(df.values, [[2, 1], [3, 2], [2, 1]])

  def test_remove_nas_w_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, np.nan]})        
    df.rmnas(['n_1'])
    np.testing.assert_array_equal(df.values, [[2, 1], [3, 2], [4, 3], [2, 1], [3, np.nan]])

    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 'n_2': [1, 2, np.nan, 1, 2]})        
    df.rmnas(['n_1'])
    np.testing.assert_array_equal(df.values, [[2, 1], [3, 2], [4, np.nan], [2, 1]])

  def test_to_vw(self):
    df = pd.DataFrame({'n_price': [0.23, 0.18, 0.53], 'n_sqft': [0.25, 0.15, 0.32], 'c_year': [2006, 1976, 1924]})
    y = pd.Series([0, 1, 0])
    weights = [1, 2, 0.5]
    file = 'test_vw_file.vw'    
    df.to_vw(file, y, weights)
    with open(file) as f: lines = f.readlines()
    os.remove(file)
    print lines
    np.testing.assert_array_equal([
      
      '-1.0 \'0 |n 0:0.23 1:0.25 |c 2\n',
      '1.0 2 \'1 |n 0:0.18 1:0.15 |c 3\n',
      '-1.0 0.5 \'2 |n 0:0.53 1:0.32 |c 4\n',
      ], lines)

  def test_to_libfm(self):
    df = pd.DataFrame({'n_price': [0.23, 0.18, 0.53], 'n_sqft': [0.25, 0.15, 0.32], 'c_year': [2006, 1976, 1924]})
    y = pd.Series([0, 1, 0])
    file = 'test_libfm_file.libfm'    
    df.to_libfm(file, y)
    with open(file) as f: lines = f.readlines()
    os.remove(file)
    np.testing.assert_array_equal([
      '0.0 0:0.23 1:0.25 2:1\n',
      '1.0 0:0.18 1:0.15 3:1\n',
      '0.0 0:0.53 1:0.32 4:1\n',
      ], lines)

  def test_to_svmlight(self):
    df = pd.DataFrame({'n_price': [0.23, 0.18, 0.53], 'n_sqft': [0.25, 0.15, 0.32], 'c_year': [2006, 1976, 1924]})
    y = pd.Series([0, 1, 0])
    file = 'test_libfm_file.svmlight'    
    df.to_svmlight(file, y)
    with open(file) as f: lines = f.readlines()
    os.remove(file)
    np.testing.assert_array_equal([
      '-1.0 0:0.23 1:0.25 2:1\n',
      '1.0 0:0.18 1:0.15 3:1\n',
      '-1.0 0:0.53 1:0.32 4:1\n',
      ], lines)

  def test_save_csv(self):
    rows = 10000
    '''
    Before optimisations took (rows=1M):    
    to csv takes:     8-9s
    to csv.gz takes:  319s
    
    After (rows=1M):
    to csv.gz takes: 24s

    '''
    df = pd.DataFrame({'col_1': range(rows), 'col_2': range(rows), 'col_3': range(rows)})    
    def impl(file):
      df.save_csv(file)      
      df2 = read_df(file)
      os.remove(file)    
      self.assertEquals(rows, df2.shape[0])
    impl('test.csv.gz')


  def test_describe_data(self):
    pass

if __name__ == '__main__':
  unittest.main()
