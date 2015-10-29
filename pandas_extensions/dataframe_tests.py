import unittest, sklearn, sklearn.linear_model, \
  sklearn.decomposition, sklearn.ensemble, datetime, scipy, os
import pandas as pd, numpy as np
from . import *
from . import base_pandas_extensions_tester

class T(base_pandas_extensions_tester.BasePandasExtensionsTester):  
  def test_categoricals(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'n_1': [1., 2., 3.]})
    self.assertEqual(['c_1'], df.categoricals())

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
    df.to_indexes(drop_origianls=True)
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
    self.eq(df, 
      np.array([
        ['a', 'd', -1, -1, -1],
        ['b', 'e', 0, 0, 0],
        ['c', 'f', 1, 1, 1]
        ], 'object'))

    df = pd.DataFrame({'n_2': [1., 2., 3., 4., 5.], 'n_3': [4., 5., 6., 7., 8.]})
    df.scale()
    self.close(df, [[-1.26491106, -1.26491106],
                    [-0.63245553, -0.63245553],
                    [0, 0],
                    [0.63245553, 0.63245553],
                    [1.26491106, 1.26491106]])

  def test_scale_with_min_max(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1., 2., 3.], 'n_3': [4., 5., 6.], 'n_4': [7., 8., 9.]})        
    df.scale(min_max=(0., 2.))
    self.eq(df, 
      np.array([
        ['a', 'd', 0, 0, 0],
        ['b', 'e', 1, 1, 1],
        ['c', 'f', 2, 2, 2]
        ], 'object'))

    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1., 2., 3.], 'n_3': [4., 5., 6.], 'n_4': [7., 8., 9.]})        
    df.scale(min_max=(10., 20.))
    self.eq(df, 
      np.array([
        ['a', 'd', 10, 10, 10],
        ['b', 'e', 15, 15, 15],
        ['c', 'f', 20, 20, 20]
        ], 'object'))

  def test_normalise(self):
    df = pd.DataFrame({'c_1':['a', 'b', 'c'], 'c_2':['d', 'e', 'f'], 
      'n_2': [1., 2., 3.], 'n_3': [4., 5., 6.], 'n_4': [7., 8., 9.]})        
    df.normalise()
    self.eq(df, 
      np.array([
        ['a', 'd', 0, 0, 0],
        ['b', 'e', .5, .5, .5],
        ['c', 'f', 1, 1, 1]
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
    df = pd.DataFrame({'n_1': np.random.normal(size=200)})
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
    arr1 = scipy.sparse.coo_matrix([[4], [5], [6]])
    arr2 = df1.append_right(arr1)
    self.assertTrue(type(arr2) is scipy.sparse.coo.coo_matrix)
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
    misc.cfg['sys_seed'] = 0
    df2, y2 = df.shuffle(y)
    
    # Originals did not change
    self.eq(df, np.array([['a', 1L], ['b', 2L], ['c', 3L], ['d', 4L], ['e', 5L], ['f', 6L], ['g', 7L]], dtype='object'))
    self.eq(y, [1, 2, 3, 4, 5, 6, 7])

    # Changed
    self.eq(df2, np.array([['g', 7L], ['a', 1L], ['d', 4L], ['b', 2L], ['c', 3L], ['e', 5L], ['f', 6L]], dtype='object'))
    self.eq(y2, [7, 1, 4, 2, 3, 5, 6])

  def test_to_indexes(self):
    df = pd.DataFrame({'c_1':['a', 'b'], 'c_2':['c', 'd'], 'n_1': [1, 2]})
    print df.values
    df.to_indexes(drop_origianls=True)
    self.eq(df, [[1, 0, 0], [2, 1, 1]])

  def test_to_indexes_with_NA(self):
    df = pd.DataFrame({'c_1':['a', 'b', np.nan], 'c_2':['c', 'd', np.nan], 'n_1': [1, 2, 3]})
    df.to_indexes(drop_origianls=True)
    self.eq(df, [[1, 0, 0], [2, 1, 1], [3, -1, -1]])
  
  def test_cv(self):
    df = pd.DataFrame({'n_1': [1, 2, 3, 4, 5, 6, 7]})
    y = pd.Series([1L, 2L, 3L, 4L, 5L, 6L, 7L])        
    df.cv(sklearn.linear_model.LinearRegression(), y)

  def test_cv_ohe(self):
    df = pd.DataFrame({
      'c_1':['a', 'b', 'c', 'd', 'e', 'f', 'g'] * 10,  
      'n_1': [1, 2, 3, 4, 5, 6, 7] * 10})
    y = pd.Series([1, 0, 0, 1, 1, 0, 1] * 10)        
    df.cv_ohe(sklearn.linear_model.LogisticRegression(), y)

  def test_pca(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, 2],
        'n_12': [2, 3, 4, 2, 3], 'n_22': [1, 2, 3, 1, 2]})        
    df = df.pca(2)
    self.close(df.values, [[ -1.60000000e+00,   6.11173774e-17],
                           [  4.00000000e-01,  -2.07359751e-16],
                           [  2.40000000e+00,  -9.99688998e-17],
                           [ -1.60000000e+00,  -3.14750603e-16],
                           [  4.00000000e-01,  -2.07359751e-16]])

  def test_remove_nas(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, np.nan]})        
    df.rmnas()
    self.eq(df, [[2, 1], [3, 2], [4, 3], [2, 1]])

    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 'n_2': [1, 2, np.nan, 1, 2]})        
    df.rmnas()
    self.eq(df, [[2, 1], [3, 2], [2, 1]])

  def test_remove_nas_w_columns(self):
    df = pd.DataFrame({'n_1': [2, 3, 4, 2, 3], 'n_2': [1, 2, 3, 1, np.nan]})        
    df.rmnas(['n_1'])
    self.eq(df, [[2, 1], [3, 2], [4, 3], [2, 1], [3, np.nan]])

    df = pd.DataFrame({'n_1': [2, 3, 4, 2, np.nan], 'n_2': [1, 2, np.nan, 1, 2]})        
    df.rmnas(['n_1'])
    self.eq(df, [[2, 1], [3, 2], [4, np.nan], [2, 1]])

  def test_to_vw(self):
    df = pd.DataFrame({'n_price': [0.23, 0.18, 0.53], 'n_sqft': [0.25, 0.15, 0.32], 'c_year': [2006, 1976, 1924]})
    y = pd.Series([0, 1, 0])
    weights = [1, 2, 0.5]
    file = 'test_vw_file.vw'    
    df.to_vw(file, y, weights)
    with open(file) as f: lines = f.readlines()
    os.remove(file)    
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
      '-1.0 1:0.23 2:0.25 3:1\n',
      '1.0 1:0.18 2:0.15 4:1\n',
      '-1.0 1:0.53 2:0.32 5:1\n',
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
      df2 = misc.read_df(file)
      os.remove(file)    
      self.assertEquals(rows, df2.shape[0])
    impl('test.csv.gz')

  def test_save_csv_with_existing_file(self):
    df = pd.DataFrame({'col_1': range(10)})    
    file = 'test.csv'
    if os.path.isfile(file): os.remove(file)
    df.save_csv(file)
    self.assertTrue(os.path.isfile(file))
    try:
      df.save_csv(file)
      self.fail()
    except:
      pass
    self.assertTrue(os.path.isfile(file))
    os.remove(file)

  def test_ensure_unique_names(self):
    df = pd.DataFrame({
      'c1':['a', 'b', 'c'] * 5,
      'c2':['b', 'c', 'b'] * 5,
      'c3': [10, 100, 100] * 5,
      'c4': np.random.normal(size=15),
      'c5': [datetime.datetime(2010, 1, 1)] * 15
      })
    df.columns = ['c'] * 5
    columns = df.ensure_unique_names().columns
    self.assertEqual(columns.tolist(), ['c', 'c_1', 'c_2', 'c_3', 'c_4'])

  def test_infer_col_names(self):
    columns = pd.DataFrame({
      'c1':['a', 'b', 'c'] * 5,
      'c2':['b', 'c', 'b'] * 5,
      'c3': [10, 100, 100] * 5,
      'c4': np.random.normal(size=15),
      'c5': [datetime.datetime(2010, 1, 1)] * 15
      }).infer_col_names().columns
    self.assertEqual(columns.tolist(), ['c_c1', 'b_c2', 'b_c3', 'n_c4', 'd_c5'])

  def test_infer_col_names_with_unknowns(self):
    df = pd.DataFrame({
      'c1':['a', 'b', 'c'] * 5,
      'c2':['b', 'c', 'b'] * 5,
      'c3': [10, 100, 100] * 5,
      'c4': np.random.normal(size=15),
      'c5': [datetime.datetime(2010, 1, 1)] * 15
      })
    df.columns = [None] * 5
    columns = df.infer_col_names().columns
    self.assertEqual(columns.tolist(), ['c_unknown', 'b_unknown', 'b_unknown_1', 'n_unknown', 'd_unknown'])

  def test_cats_to_count_of_samples(self):
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})
    df.cats_to_count_of_samples()
    self.eq(df, [[3, 2], [3, 2], [3, 3], [2, 3], [2, 3], [1, 1]])

  def test_cats_to_ratio_of_samples(self):
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})
    df.cats_to_ratio_of_samples()
    self.eq(df, [[3/6., 2/6.], [3/6., 2/6.], [3/6., 3/6.], [2/6., 3/6.], [2/6., 3/6.], [1/6., 1/6.]])

  def test_cats_to_count_of_binary_target(self):
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})
    y = [0, 1, 1, 0, 1, 0]
    df.cats_to_count_of_binary_target(y)
    self.eq(df, [[2, 1], [2, 1], [2, 2], [1, 2], [1, 2], [0, 0]])

  def test_cats_to_ratio_of_binary_target(self):
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})
    y = [0, 1, 1, 0, 1, 0]
    df.cats_to_ratio_of_binary_target(y)
    self.eq(df, [[2/3., 1/2.], [2/3., 1/2.], [2/3., 2/3.], [1/2., 2/3.], [1/2., 2/3.], [0, 0]])

  def test_cats_to_stats(self):
    y = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})    
    df.cats_to_stat(y)
    self.eq(df, [[2., 1.5], [2., 1.5], [2., 4], [4.5, 4], [4.5, 4], [6, 6]])

    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})    
    df.cats_to_stat(y, 'max')
    self.eq(df, [[3., 2], [3., 2], [3., 5], [5, 5], [5, 5], [6, 6]])

  def test_cats_to_stats_with_all(self):
    y = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})    
    df.cats_to_stat(y, 'all')
    print df
    self.eq(df.columns, ['n_c_1_mean', 'n_c_2_mean', 'n_c_1_iqm', 'n_c_2_iqm', 'n_c_1_median', 'n_c_2_median', 'n_c_1_min', 'n_c_2_min', 'n_c_1_max', 'n_c_2_max'])
    self.eq(df, [ [2., 1.5, 2.0, 1.5, 2, 1.5, 1, 1, 3, 2], 
                  [2., 1.5, 2.0, 1.5, 2, 1.5, 1, 1, 3, 2], 
                  [2.,   4, 2.0, 4.0, 2, 4, 1, 3, 3, 5], 
                  [4.5,  4, 4.5, 4.0, 4.5, 4, 4, 3, 5, 5], 
                  [4.5,  4, 4.5, 4.0, 4.5, 4, 4, 3, 5, 5], 
                  [6,    6, 6.0, 6.0, 6, 6, 6, 6, 6, 6]])

  def test_cats_to_stats_with_dict(self):
    y = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame({'c_1': ['a', 'a', 'a', 'b', 'b', 'c'], 'c_2': ['a', 'a', 'b', 'b', 'b', 'c']})    
    df.cats_to_stat(y, {'c_1': 'mean', 'c_2': 'iqm'})
    self.eq(df, [[2., 1.5], [2., 1.5], [2., 4], [4.5, 4], [4.5, 4], [6, 6]])

  def test_group_rare(self):
    df = pd.DataFrame({
      'c_1': ['a', 'b', 'c'] * 100 + ['d', 'e', 'f'] * 10,
      'c_2': ['a', 'b', 'c'] * 100 + ['d', 'e'] * 15,
      })    
    df2 = df.copy().group_rare()
    self.eq(df2, {
      'c_1': ['a', 'b', 'c'] * 100 + ['rare'] * 30,
      'c_2': ['a', 'b', 'c'] * 100 + ['rare'] * 30
      })

    df2 = df.copy().group_rare(limit=5)
    self.eq(df2, {
      'c_1': ['a', 'b', 'c'] * 100 + ['d', 'e', 'f'] * 10,
      'c_2': ['a', 'b', 'c'] * 100 + ['d', 'e'] * 15
      })

  def test_noise_filter_gaussian(self):
    df = pd.DataFrame({'n_1':np.random.normal(size=200)})
    df2 = df.copy().noise_filter('gaussian', 1, axis=-1)
    self.assertFalse(df.is_equal(df2))
    self.assertTrue(df.all_close(df2))

  def test_noise_filter_maximum(self):
    df = pd.DataFrame({'n_1':np.random.normal(size=200)})
    df2 = df.noise_filter('maximum', 1, axis=-1)    
    self.assertTrue(df.all_close(df2))

  def test_noise_filter_minimum(self):
    df = pd.DataFrame({'n_1':np.random.normal(size=200)})
    df2 = df.noise_filter('minimum', 1)
    self.assertTrue(df.all_close(df2))

  def test_noise_filter_uniform(self):
    df = pd.DataFrame({'n_1':np.random.normal(size=200)})
    df2 = df.noise_filter('uniform', 1)
    self.assertTrue(df.all_close(df2))
  
  def test_split(self):
    df = pd.DataFrame({'n_1': range(20)})
    y = [0, 1] * 10
    X_train, y_train, X_test, y_test = df.split(y)    
    self.eq(X_train, {'n_1': [5, 14,  9,  7, 16, 11,  3,  0, 15, 12]})
    self.eq(y_train, [1, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    self.eq(X_test, {'n_1': [18, 1, 19, 8, 10, 17, 6, 13, 4, 2]})
    self.eq(y_test, [0, 1, 1, 0, 0, 1, 0, 1, 0, 0])

    X_train, y_train, X_test, y_test = df.split(y, stratified=True)    
    self.eq(X_train, {'n_1': [8, 18, 5, 2, 7, 16, 4, 11, 19, 3]})
    self.eq(y_train, [0, 0, 1, 0, 1, 0, 0, 1, 1, 1])
    self.eq(X_test, {'n_1': [1, 14, 9, 6, 13, 17, 15, 12, 0, 10]})
    self.eq(y_test, [1, 0, 1, 0, 1, 1, 1, 0, 0, 0])

    X_train, y_train, X_test, y_test = df.split(y, stratified=True, train_fraction=.8)    
    self.eq(X_train, {'n_1': [5, 19, 3, 8, 14, 1, 4, 18, 2, 17, 11, 12, 16, 7, 13, 6]})
    self.eq(y_train, [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    self.eq(X_test, {'n_1': [0, 15, 10, 9]})
    self.eq(y_test, [0, 1, 0, 1])

  def test_tsne(self):
    df = pd.DataFrame(np.random.normal(size=(20, 5)))
    tsne = df.tsne()
    exp = [[  1.01495838e+02,  -1.03764146e+02],
           [  5.25378255e+01,   1.75595664e-01],
           [  1.81826568e+02,  -1.41299181e+00],
           [  6.72235498e+02,  -5.03992958e+02],
           [  1.95153600e+02,  -1.54963563e+02],
           [  5.42593037e+02,  -2.58430632e+02],
           [ -1.17398088e+01,   8.86143537e+01],
           [  7.68283241e-01,  -1.28952467e+02],
           [ -1.61562297e+02,   6.76526983e+01],
           [ -1.99880527e+02,   2.46638701e+02],
           [ -1.72567856e+03,  -6.29467935e+02],
           [ -5.84132372e+02,   6.41828851e+02],
           [ -7.04863185e+02,   2.53752017e+02],
           [  4.54027458e+01,   1.94803621e+02],
           [ -9.13778706e+01,   1.69451498e+02],
           [ -6.28457610e+01,  -1.47037042e+01],
           [ -1.02945979e+02,  -1.51704781e+02],
           [ -1.78403109e+02,  -5.35940696e+01],
           [  1.28801406e+02,   1.04813325e+02],
           [  1.59582183e+03,   4.41019932e+02]]
    self.assertTrue(tsne.all_close(exp))

  def test_kmeans(self):
    df = pd.DataFrame(np.random.normal(size=(20, 5)))
    kmeans = df.kmeans(2)
    self.eq(kmeans, [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

  def test_tree_features(self):
    df = pd.DataFrame(np.random.normal(size=(20, 5)))
    y = np.random.normal(size=20)
    tf = df.tree_features(sklearn.ensemble.RandomForestClassifier(2), y)
    exp = [[10,  7],
           [17,  7],
           [15, 13],
           [17, 18],
           [21,  5],
           [10,  7],
           [ 6,  3],
           [ 8,  5],
           [ 2, 13],
           [ 8,  1],
           [15, 11],
           [ 8,  3],
           [ 6,  3],
           [ 3, 11],
           [ 8,  3],
           [ 6,  1],
           [15,  2],
           [ 7,  3],
           [15, 13],
           [15, 13]]
    self.eq(tf, exp)

  def test_append_fit_transformer(self):
    df = pd.DataFrame({'n_1': np.random.normal(size=10)}).scale(min_max=(1, 2)).astype(int)
    df2 = df.append_fit_transformer(sklearn.preprocessing.OneHotEncoder())    
    exp = [[ 1.,  1.,  0.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  0.],
           [ 2.,  0.,  1.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  0.],
           [ 1.,  1.,  0.],]
    self.eq(df2, exp)

  def test_predict(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = pd.Series(np.random.normal(size=10)).normalise().astype(int)
    try: df.predict(lr, y)
    except: pass

    df = pd.DataFrame(np.random.normal(size=(12, 2)))
    predictions = df.predict(lr, y)
    self.eq([1, 0], predictions)

  def test_predict_proba(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = pd.Series(np.random.normal(size=10)).normalise().astype(int)
    try: df.predict_proba(lr, y)
    except: pass

    df = pd.DataFrame(np.random.normal(size=(12, 2)))
    predictions = pd.DataFrame(df.predict_proba(lr, y))
    self.close(predictions, [[ 0.133813,  0.866187], [ 0.827253,  0.172747]])

  def test_transform(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = pd.Series(np.random.normal(size=10)).normalise().astype(int)
    try: df.transform(lr, y)
    except: pass

    df = pd.DataFrame(np.random.normal(size=(12, 2)))
    predictions = pd.DataFrame(df.transform(lr, y))
    self.close(predictions, [[-2.55299], [0.864436]])

  def test_decision_function(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = pd.Series(np.random.normal(size=10)).normalise().astype(int)
    try: df.decision_function(lr, y)
    except: pass

    df = pd.DataFrame(np.random.normal(size=(12, 2)))
    predictions = df.decision_function(lr, y)
    self.close([1.867659, -1.56628], predictions)

  def test_self_predict(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = [0, 1] * 5
    predictions = df.self_predict(lr, y)
    self.eq([[0, 1, 0, 0, 1, 1, 0, 1, 0, 0]], predictions.T)

  def test_self_predict_proba(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = [0, 1] * 5
    predictions = df.self_predict_proba(lr, y)
    self.close([0.448,  0.552,  0.199,  0.38 ,  0.667,  0.66 ,  0.454,  0.522, 0.365,  0.326], predictions.T[1])

  def test_self_transform(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = [0, 1] * 5
    predictions = df.self_transform(lr, y)
    self.close([[0.4  ,  0.979,  1.868,  0.95 , -0.103,  1.454,  0.122,  0.334, -0.205, -0.854]], predictions.T)

  def test_self_decision_function(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    y = [0, 1] * 5
    predictions = df.self_decision_function(lr, y)
    self.close([[-0.21 ,  0.21 , -1.395, -0.49 ,  0.696,  0.663, -0.186,  0.09 ,-0.553, -0.726]], predictions.T)

  def test_nbytes(self):
    df = pd.DataFrame(np.random.normal(size=(10, 2)))
    self.assertEqual(256, df.nbytes())

    df = pd.DataFrame(np.random.normal(size=(1000, 2)))
    self.assertEqual(24016, df.nbytes())

  def test_compress_size(self):
    df = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=['n_1', 'n_2'])
    self.assertEqual(24016, df.nbytes())
    self.assertEqual(24016, df.compress_size(0).nbytes())
    self.assertEqual(16016, df.compress_size(1).nbytes())
    self.assertEqual(12016, df.compress_size(2).nbytes())

  def test_hashcode(self):
    np.random.seed(0)
    df = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=['n_1', 'n_2'])
    v1 = df.hashcode()

    np.random.seed(0)
    df = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=['n_1', 'n_2'])
    self.assertEqual(v1, df.hashcode())

    np.random.seed(1)
    df = pd.DataFrame(np.random.normal(size=(1000, 2)), columns=['n_1', 'n_2'])
    self.assertNotEqual(v1, df.hashcode())

  def test_trim_on_y(self):
    df = pd.DataFrame(np.random.normal(size=(1000, 2)))
    y = np.arange(1000)
    df2, y2 = df.trim_on_y(y, 100)
    self.assertEqual(900, df2.shape[0])
    self.assertEqual(900, len(y2))

    df2, y2 = df.trim_on_y(y, 100, 500)
    self.assertEqual(401, df2.shape[0])
    self.assertEqual(401, len(y2))

    df2, y2 = df.trim_on_y(y, None, 250)
    self.assertEqual(251, df2.shape[0])
    self.assertEqual(251, len(y2))

  def test_importances(self):    
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(100, 2)))
    y = np.random.normal(size=(100))
    imps = df.importances(lr, y)
    self.close([[0, 0.033174], [1, 0.009108]], imps)

  def test_numerical_stats(self):
    lr = sklearn.linear_model.LogisticRegression()
    df = pd.DataFrame(np.random.normal(size=(5, 1)), columns=['n_1'])
    stats = df.numerical_stats()
    exp = [[ 1.76405235,  1.76405235,  1.76405235, np.nan,  0., 1.76405235,  1.76405235  , np.nan  , np.nan , 1.76405235],
           [ 0.40015721,  0.40015721,  0.40015721, np.nan,  0., 0.40015721,   0.40015721 , np.nan ,  np.nan,  0.40015721],
           [ 0.97873798,  0.97873798,  0.97873798, np.nan,  0., 0.97873798,   0.97873798 , np.nan ,  np.nan,  0.97873798],
           [ 2.2408932 ,  2.2408932 ,  2.2408932 , np.nan,  0., 2.2408932 ,  2.2408932   , np.nan  , np.nan , 2.2408932 ],
           [ 1.86755799,  1.86755799,  1.86755799, np.nan,  0., 1.86755799,   1.86755799 , np.nan ,  np.nan,  1.86755799]]
    self.close(exp, stats)

  def test_smote(self):
    df = pd.DataFrame(np.random.normal(size=(100, 2)))
    y = [0, 0, 0, 1, 1] * 20
    df2, y2 = df.smote(y, 100, n_neighbors=2)
    self.assertEqual(140, len(df2))
    self.assertEqual(140, len(y2))

  def test_boxcox(self):
    df = pd.DataFrame(np.random.normal(size=(5, 3)), columns=['n_1', 'n_2', 'n_3'])
    df = df.scale(min_max=(1, 10))    
    df = df.boxcox()
    exp = [[ 2.86244796, 1.05187113, 6.53554472],
           [ 3.39513902, 1.70750321, 0.        ],
           [ 1.60306366, 0.,         3.03123573],
           [ 0.        , 0.75048413, 8.03130896],
           [ 1.179747  , 0.71532856, 4.82423476]]
    self.close(exp, df)

  def test_boxcox_on_negatives(self):
    df = pd.DataFrame(np.random.normal(size=(5, 3)), columns=['n_1', 'n_2', 'n_3'])
    df = df.boxcox()
    print df.values
    exp = [[ 0.60018495,  0.31885521,  2.19243151],
           [ 0.87374604,  0.52973271,  0.        ],
           [-0.05094574,  0.,          0.92972061],
           [-0.81723613,  0.21331878,  2.77613411],
           [-0.26593622,  0.20148432,  1.55562555]]
    self.close(exp, df)

  def test_to_dates(self):
    df = pd.DataFrame({ 'col1': ['1997-01-01', '2005-05-12'] })
    df.to_dates('col1')
    self.assertFalse('col1' in df)
    self.assertTrue('d_col1' in df)
    self.eq(df.d_col1, pd.to_datetime([datetime.datetime(1997, 1, 1), datetime.datetime(2005, 5, 12)]))

  def test_break_down_dates(self):
    df = pd.DataFrame({ 'd_col1': [1, 2] })
    df.d_col1 = pd.to_datetime([datetime.datetime(1997, 1, 1), datetime.datetime(2005, 5, 12)])
    df_0 = df.copy().break_down_dates(0)    
    self.eq(df_0.columns, ['c_d_col1_year', 'c_d_col1_month'])
    self.eq(df_0, [[1997, 1], [2005, 5]])

    df_1 = df.copy().break_down_dates(1)        
    self.eq(df_1.columns, ['c_d_col1_year','c_d_col1_month','c_d_col1_dayofweek','c_d_col1_quarter'])
    self.eq(df_1, [[1997, 1, 2, 1], [2005, 5, 3, 2]])

    df_2 = df.copy().break_down_dates(2)        
    self.eq(df_2.columns, ['c_d_col1_year','c_d_col1_month','c_d_col1_dayofweek','c_d_col1_quarter','c_d_col1_year_and_month','c_d_col1_weekday','c_d_col1_weekofyear'])
    self.eq(df_2, [[1997, 1, 2, 1, 199701, 2, 1], [2005, 5, 3, 2, 200505, 3, 19]])

  def test_summarise(self):
    pass
    # TODO
    '''
    df = pd.DataFrame(np.random.normal(size=(100, 2)), columns=['n_1', 'n_2'])
    f = 'dummy_file.test'
    if os.path.isfile(f): os.path.remove(f)
    
    df.summarise(filename=f)

    self.assertTrue(os.path.isfile(f))
    if os.path.isfile(f): os.path.remove(f)
    '''

  def test_floats_to_ints(self):
    df = pd.DataFrame(np.random.normal(size=(100, 2)), columns=['n_1', 'n_2'])
    df.floats_to_ints()
    self.eq(map(str, df.dtypes), ['int32', 'int32'])
    self.eq(df.mean(), [-95.74, 14277.78])

  def test_add_noise(self):
    df = pd.DataFrame(np.random.normal(size=(100, 2)), columns=['n_1', 'n_2'])
    df2 = df.copy().add_noise(level=.0001)
    self.assertFalse(df.is_equal(df2))
    self.close(df, df2)

    df2 = df.copy().add_noise(level=.0001, mode='gaussian')
    self.assertFalse(df.is_equal(df2))
    self.close(df, df2)

  def test_impute_categorical(self):
      df = pd.DataFrame({'n_1': np.random.random(1000)})
      df['b_to_impute'] = df.n_1 > 0.5
      df['b_to_impute'].ix[995:] = 'missing'
      df.impute_categorical('b_to_impute', 'missing')
      self.eq(df.b_to_impute[995:], df.n_1[995:] > 0.5)

  def test_custom_cache(self):
    df = pd.DataFrame({'n_1': np.random.random(100)})
    self.assertIsNone(df.custom_cache('missing'))
    self.assertEqual(123, df.custom_cache('key1', 123))
    self.assertEqual(123, df.custom_cache('key1'))
    df['new_column'] = np.random.random(100)
    self.assertIsNone(df.custom_cache('key1'))

  def test_subsample(self):
    df = pd.DataFrame({'n_1': np.random.random(100)})
    y = pd.Series(np.random.random(50))
    df2, y2 = df.subsample(y, size=.5)
    self.assertEquals(25, len(y2))
    self.assertEquals(25, len(df2))
    self.assertFalse(np.any(np.isnan(y2)))

