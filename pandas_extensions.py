'''
Naming Conventions for Features:
c_ = categorical
i_ = categoricals as indexes
n_ = numerical
b_ = binary
d_ = date

TODO:
- Time series computations 
  see: http://pandas.pydata.org/pandas-docs/stable/computation.html
- Assume all methods destructive
- Try pandas vs numpy sparse arrays
'''

import pandas as pd
import numpy as np
from misc import *
from ast_parser import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import utils, cross_validation
from scipy import sparse
import itertools, random, gzip
from scipy.ndimage.filters import *

'''
Series Extensions
'''
def _s_one_hot_encode(self):
  start('one_hot_encoding column')  

  arr = self.values
  col_ohe = OneHotEncoder().fit_transform(arr.reshape((len(arr), 1)))

  stop('done one_hot_encoding column converted to ' + 
      `col_ohe.shape[1]` + ' columns')  
  return col_ohe

def _s_bin(self, n_bins=100):
  return pd.Series(pd.cut(self, n_bins), index=self.index)

def _s_sigma_limits(self, sigma):
  delta = float(sigma) * self.std()
  m = self.mean()
  return (m - delta, m + delta)

def _s_to_indexes(self):
  c = self.name
  col = 'i_' + c
  cat = pd.Categorical.from_array(self)
  lbls = cat.codes if hasattr(cat, 'codes') else cat.labels    
  return pd.Series(lbls, index=self.index, \
      dtype=_get_optimal_numeric_type('int', 0, len(lbls) + 1))

'''
DataFrame Extensions
'''

def _df_categoricals(self): return filter(lambda c: c.startswith('c_'), self.columns)
def _df_indexes(self): return filter(lambda c: c.startswith('i_'), self.columns)
def _df_numericals(self): return filter(lambda c: c.startswith('n_'), self.columns)
def _df_binaries(self): return filter(lambda c: c.startswith('b_'), self.columns)
def _df_dates(self): return filter(lambda c: c.startswith('d_'), self.columns)

def _df_one_hot_encode(self, dtype=np.float):  
  if self.categoricals(): self.to_indexes(drop_origianls=True)    

  start('one_hot_encoding data frame with ' + `self.shape[1]` + \
    ' columns. \n\tNOTE: this resturns a sparse array and empties' + \
    ' the initial array.')  

  debug('separating categoricals from others')
  indexes = self.indexes()
  if not indexes: return self
  others = filter(lambda c: not c in indexes, self.columns)

  categorical_df = self[indexes]    
  others_df = sparse.coo_matrix(self[others].values)

  # Destroy original as it now just takes up memory
  self.drop(self.columns, 1, inplace=True) 
  gc.collect()

  ohe_sparse = None
  for i, c in enumerate(indexes):
    debug('one hot encoding column: ' + `c`)
    col_ohe = OneHotEncoder(categorical_features=[0], dtype=dtype).\
      fit_transform(categorical_df[[c]])
    if ohe_sparse == None: ohe_sparse = col_ohe
    else: ohe_sparse = sparse.hstack((ohe_sparse, col_ohe))
    categorical_df.drop(c, axis=1, inplace=True)
    gc.collect()
  
  matrix = ohe_sparse if not others else sparse.hstack((ohe_sparse, others_df))
  stop('done one_hot_encoding')
  return matrix.tocsr()

def _df_to_indexes(self, drop_origianls=False, sparsify=False):
  start('indexing categoricals in data frame. Note: NA gets turned into max index (255, 65535, etc)')  
  for c in self.categoricals():
    col = 'i_' + c
    cat = pd.Categorical.from_array(self[c])
    lbls = cat.codes if hasattr(cat, 'codes') else cat.labels    
    s = pd.Series(lbls, index=self[c].index, \
      dtype=_get_optimal_numeric_type('int', 0, len(lbls) + 1))
    modes = s.mode()
    mode = lbls[0]
    if len(modes) > 0: mode = modes.iget(0)
    self[col] = s.to_sparse(fill_value=int(mode)) if sparsify else s
    if drop_origianls: self.drop(c, 1, inplace=True)
  stop('done indexing categoricals in data frame')  
  return self

def _df_bin(self, n_bins=100, drop_origianls=False):
  start('binning data into ' + `n_bins` + ' bins')  
  for n in self.numericals():
    self['c_binned_' + n] = pd.Series(pd.cut(self[n], n_bins), index=self[n].index)
    if drop_origianls: self.drop(n, 1, inplace=True)
  stop('done binning data into ' + `n_bins` + ' bins')  
  return self

def _df_combinations(self, group_size=2, columns=[], categoricals=False, indexes=False,
    numericals=False, dates=False, binaries=False):
  cols = list(columns)
  if categoricals: cols = cols + self.categoricals()
  if indexes: cols = cols + self.indexes()
  if numericals: cols = cols + self.numericals()
  if dates: cols = cols + self.dates()
  if binaries: cols = cols + self.binaries()
  return list(itertools.combinations(cols, group_size))

def _df_remove_nas(self, columns=None):      
  self.dropna(0, 'any', subset=columns, inplace=True)
  return self

def _df_remove(self, columns=[], categoricals=False, numericals=False, 
    dates=False, binaries=False, missing_threshold=0.0):    
  cols = [columns] if type(columns) is str else list(columns)
  if categoricals: cols = cols + self.categoricals()
  if numericals: cols = cols + self.numericals()
  if dates: cols = cols + self.dates()
  if binaries: cols = cols + self.binaries()
  if missing_threshold > 0:
    for c in self.columns:
      nvc = self[c].isnull().value_counts()  
      if True in nvc and False in nvc and \
          nvc[True] / float(len(self)) > missing_threshold:
        cols.append(c)
  cols = set(cols)
  if len(cols) == 0: 
    raise Exception('At least one of categoricals, numericals, ' +
      'dates binaries should be set to True or columns array passed')

  debug('removing ' + `len(cols)` + ' columns from data frame')
  self.drop(cols, 1, inplace=True)
  return self

def _df_engineer(self, name, columns=None, quiet=False):  
  '''
  name(Array|string): Can list-like of names.  ';' split list of names 
  also supported
  '''
  if type(name) is str and ';' in name: name = name.split(';')
  if type(name) is list or type(name) is tuple: 
    for n in name: self.engineer(n)
    return self

  def func_to_string(c):
    func = c.func
    args = c.args
    return func + '(' + ','.join(map(lambda a: 
      func_to_string(a) if hasattr(a, 'func') else a, args)) + ')'
  
  def get_new_col_name(c):
    prefix = 'c_' if c.func == 'concat' else 'n_'    
    suffix = func_to_string(c)
    return suffix if suffix.startswith(prefix) else prefix + suffix
  
  c = explain(name)[0]
  func = c.func if not type(c) is str else None
  args = c.args if not type(c) is str else None

  new_name = get_new_col_name(c) if not type(c) is str else c
  if new_name in self.columns: return self # already created column  

  # Evaluate any embedded expressions in the 'name' expression
  for i, a in enumerate(args): 
    if hasattr(a, 'func'): 
      args[i] = get_new_col_name(a)
      self.engineer(func_to_string(a))

  if not quiet: debug('engineering feature: ' + name)
  if len(args) == 0 and (func == 'avg' or func == 'mult' or func == 'concat'):    
    combs = list(itertools.combinations(columns, 2)) if columns \
      else self.combinations(categoricals=func=='concat', indexes=func=='concat', numericals=func=='mult' or func=='avg')    
    for c1, c2 in combs: self.engineer(func + '(' + c1 + ',' + c2 + ')', quiet=True)
    return self
  elif func == 'concat': 
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = self[args[0]].astype(str) + self[args[1]].astype(str)
    if len(args) == 3: 
      self[new_name] = self[args[0]].astype(str) + self[args[1]].astype(str) + self[args[2]].astype(str)
  elif func  == 'mult':     
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = self[args[0]] * self[args[1]]
    if len(args) == 3: 
      self[new_name] = self[args[0]] * self[args[1]] * self[args[2]]
  elif func  == 'avg':     
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = (self[args[0]] + self[args[1]]) / 2
    if len(args) == 3: 
      self[new_name] = (self[args[0]] + self[args[1]] + self[args[2]]) / 3
  elif len(args) == 1 and func == 'pow':
    cols = columns if columns else self.numericals()
    for n in cols: self.engineer('pow(' + n + ', ' + args[0] + ')', quiet=True)
    return self
  elif len(args) == 0 and func == 'lg':
    cols = columns if columns else self.numericals()
    for n in cols: self.engineer('lg(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'sqrt':
    cols = columns if columns else self.numericals()
    for n in cols: self.engineer('sqrt(' + n + ')', quiet=True)    
    return self
  elif func == 'pow': 
    self[new_name] = np.power(self[args[0]], int(args[1]))
  elif func == 'lg': 
    self[new_name] = np.log(self[args[0]])
  elif func == 'sqrt': 
    self[new_name] = np.sqrt(self[args[0]])
  elif func.startswith('rolling_'):
    if len(args) == 1:
      cols = columns if columns else self.numericals()
      for n in cols: self.engineer(func + '(' + n + ', ' + args[0] + ')', quiet=True)
      return self
    else:      
      self[new_name] = getattr(pd, func)(self[args[0]], int(args[1]))
  else: raise Exception(name + ' is not supported')

  # Absolutely no idea why this is required but if removed 
  #   pandas_extensions_engineer_tests.py T.test_long_method_chains
  #   fails.  Its like this locks the results in before the next
  #   method is called and the next method appears to change the
  #   scale of the array? Who knows...
  self[new_name]

  return self
  
def _df_scale(self, columns=[], min_max=None):  
  start('scaling data frame')
  # If columns is not meant to be specified
  if min_max == None and len(columns) == 2:
    strtype = str(type(columns[0]))
    if strtype.startswith('int') or strtype.startswith('float'):
      min_max, columns = columns, []

  cols = columns if columns else self.numericals()
  for c in cols:
    if min_max:
      self[c] -= self[c].min()  
      self[c] /= self[c].max()
      self[c] *= (min_max[1] - min_max[0])
      self[c] += min_max[0]
    else:
      self[c] -= self[c].mean()
      self[c] /= self[c].std()
    gc.collect()
  stop('scaling data frame')
  return self

def _df_missing(self, categorical_fill='none', numerical_fill='none'):  
  start('replacing missing data categorical[' + `categorical_fill` + '] numerical[' + `numerical_fill` + ']')
  
  # Do numerical constants on whole DF for performance
  if type(numerical_fill) != str:
    self[self.numericals()] = self[self.numericals()].fillna(numerical_fill)
    self.replace([np.inf, -np.inf], numerical_fill, inplace=True)
    numerical_fill='none'

  # Do categorical constants on whole DF for performance
  if categorical_fill != 'none' and categorical_fill != 'mode':
    self[self.categoricals()] = self[self.categoricals()].fillna(categorical_fill)
    categorical_fill='none'

  # Get list of columns still left to fill
  categoricals_to_fill = []
  numericals_to_fill = []
  if categorical_fill != 'none': categoricals_to_fill += self.categoricals() + self.indexes()
  if numerical_fill != 'none': numericals_to_fill += self.numericals()

  # Prepare a dictionary of column -> fill values
  to_fill = {}
  for c in categoricals_to_fill: to_fill[c] = _get_col_aggregate(self[c], categorical_fill)
  for c in numericals_to_fill: 
    to_fill[c] = _get_col_aggregate(self[c], numerical_fill)
    self[c].replace([np.inf, -np.inf], to_fill[c], inplace=True)
  
  # Do fill in one step for performance
  if to_fill: self.fillna(value=to_fill, inplace=True)

  stop('done replacing missing data')
  return self

def _get_col_aggregate(col, mode):
  '''
  col: A pandas column
  mode: One of <constant>|mode|mean|median|min|max
  '''
  if type(mode) != str: return mode
  if mode == 'mode': return col.mode().iget(0) 
  if mode == 'mean': return col.mean()
  if mode == 'median': return col.median()
  if mode == 'min': return col.min()
  if mode == 'max': return col.max()
  if mode == 'max+1': return col.max()+1
  return mode

def _df_outliers(self, stds=3):  
  start('restraining outliers, standard deviations: ' + `stds`)
  for n in self.numericals(): 
    col = self[n]
    mean, offset = col.mean(), stds * col.std()
    min, max = mean - offset, mean + offset
    self[n] = col.clip(min, max)
  stop('done restraining outliers')
  return self

def _s_categorical_outliers(self, min_size=0.01, fill_mode='mode'):     
  threshold = float(len(self)) * min_size if type(min_size) is float else min_size 
  col = self.copy()
  fill = _get_col_aggregate(col, fill_mode)
  vc = col.value_counts()
  under = vc[vc <= threshold]    
  if under.shape[0] > 0: 
    debug('column [' + col.name + '] threshold[' + `threshold` + '] fill[' + `fill` + '] num of rows[' + `len(under.index)` + ']')
    col[col.isin(under.index)] = fill
  return col

def _s_compress(self, aggresiveness=0, sparsify=False):  
  def _get_optimal_numeric_type(dtype, min, max):
    dtype = str(dtype)
    is_int = dtype.startswith('int')
    if min >= 0 and is_int:
      '''
      uint8 Unsigned integer (0 to 255)
      uint16  Unsigned integer (0 to 65535)
      uint32  Unsigned integer (0 to 4294967295)
      uint64  Unsigned integer (0 to 18446744073709551615)
      '''
      if max <= 255: return 'uint8'
      if max <= 65535: return 'uint16'
      if max <= 4294967295: return 'uint32'
      if max <= 18446744073709551615: return 'uint64'
      raise Exception(`max` + ' is too large')
    elif is_int:
      '''
      int8 Byte (-128 to 127)
      int16 Integer (-32768 to 32767)
      int32 Integer (-2147483648 to 2147483647)
      int64 Integer (-9223372036854775808 to 9223372036854775807)
      '''
      if min >= -128 and max <= 127: return 'int8'
      if min >= -32768 and max <= 32767: return 'int16'
      if min >= -2147483648 and max <= 2147483647: return 'int32'
      if min >= -9223372036854775808 and max <= 9223372036854775807: return 'int64'
      raise Exception(`min` + ' and ' + `max` + ' are out of supported range')
    else:
      '''
      float16 Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
      float32 Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
      float64 Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
      '''
      if not dtype.startswith('float'): raise Exception('Unsupported type: ' + dtype)
      current = int(dtype[-2:])
      if aggresiveness == 0: return dtype
      if aggresiveness == 1: 
        if current == 64: return 'float32'
        elif current <= 32: return 'float16'
        elif current == 16: return 'float16'
        else: raise Exception('Unsupported type: ' + dtype)
      if aggresiveness == 2: return 'float16'      
  prefix = self.name[0:2]
  if prefix == 'n_' or prefix == 'i_':  
    if prefix == 'i_' or str(self.dtype).startswith('int'):        
      compressed = self.astype(_get_optimal_numeric_type('int', min(self), max(self)))
      return compressed if not sparsify else compressed.to_sparse(fill_value=int(self.mode()))    
    elif str(self.dtype).startswith('float'):
      return self.astype(_get_optimal_numeric_type(self.dtype, min(self), max(self)))
    else:
      raise Exception(self.name + ' expected "int" or "float" type got: ', str(self.dtype))
  else : 
    print self.name + ' is not supported, ignored during compression'
  return self

def _df_categorical_outliers(self, min_size=0.01, fill_mode='mode'):      
  start('binning categorical outliers, min_size: ' + `min_size`)

  for c in self.categoricals() + self.indexes():     
    self[c] = self[c].categorical_outliers(min_size, fill_mode)

  stop('done binning categorical outliers')
  return self

def _is_sparse(o):
  return type(o) is pd.sparse.frame.SparseDataFrame or \
    type(o) is pd.sparse.series.SparseSeries

def _df_append_right(self, df_or_s):  
  start('appending to the right.  note, this is a destructuve operation')
  if (type(df_or_s) is sparse.coo.coo_matrix):
    self_sparse = None
    for c in self.columns:
      debug('\tappending column: ' + c)
      c_coo = sparse.coo_matrix(self[[c]])
      self.drop([c], 1, inplace=True)
      gc.collect()
      if self_sparse == None: self_sparse = c_coo
      else: self_sparse = sparse.hstack((self_sparse, c_coo)) 
    self_sparse = sparse.hstack((self_sparse, df_or_s))
    stop('done appending to the right')
    return self_sparse
  elif _is_sparse(df_or_s) and not _is_sparse(self):
    debug('converting data frame to a sparse frame')
    self = self.to_sparse(fill_value=0)
  if type(df_or_s) is pd.Series: self[df_or_s.name] = df_or_s.values
  else: 
    self = pd.concat((self, df_or_s), 1)
  stop('done appending to the right')
  return self

def _df_append_bottom(self, df):  
  debug('warning: DataFrame.append_bottom always returns a new DataFrame')
  return pd.concat([self, df], ignore_index=True)

def _create_df_from_templage(template, data, index=None):
  df = pd.DataFrame(columns=template.columns, data=data, index=index)
  for c in template.columns:
    if template[c].dtype != df[c].dtype: 
      df[c] = df[c].astype(template[c].dtype)
  return df

def _create_s_from_templage(template, data):
  s = pd.Series(data)
  if template.dtype != s.dtype: s = s.astype(template.dtype)
  return s

def _df_subsample(self, y=None, size=0.5):  
  if type(size) is float:
    if size < 1.0: size = df.shape[0] * size
    size = int(size)
  if self.shape[0] <= size: return self if y is None else (self, y) # unchanged    

  start('subsample data frame')
  df = self.copy().shuffle(y)
  
  result = df[:size] if y is None else df[0][:size], df[1][:size]  
  start('done, subsample data frame')
  return result

def _df_shuffle(self, y=None):  
  start('shuffling data frame')
  df = self.copy()  
  if y is not None: 
    df = df[:y.shape[0]]
    df['__tmpy'] = y    

  index = list(df.index)
  random.seed(cfg['sys_seed'])
  random.shuffle(index)
  df = df.ix[index]
  df.reset_index(inplace=True, drop=True)

  result = df
  if y is not None:     
    y = pd.Series(df['__tmpy'], index=df.index)
    df.remove(['__tmpy'])
    result = (df, y)

  start('done, shuffling data frame')
  return result

def _df_noise_filter(self, type, *args, **kargs):  
  start('filtering data frame')
  
  filter = gaussian_filter1d if type == 'gaussian' \
    else maximum_filter1d if type == 'maximum' \
    else minimum_filter1d if type == 'minimum' \
    else uniform_filter1d if type == 'uniform' \
    else None
  if filter is None: raise Exception('filter: ' + type + ' is not supported')

  filtered = filter(self.values, *args, **kargs)
  return  _create_df_from_templage(self, filtered, self.index)

def _df_split(self, y, stratified=False, train_fraction=0.5):  
  train_size = int(self.shape[0] * train_fraction)
  test_size = int(self.shape[0] * (1.0-train_fraction))  
  start('splitting train_size: ' + `train_size` + ' test_size: ' + `test_size`)
  splitter = cross_validation.StratifiedShuffleSplit if stratified else \
    cross_validation.ShuffleSplit
  train_indexes, test_indexes = list(splitter(y, 1, test_size, train_size))[0]
  new_set = (
    self.iloc[train_indexes], 
    y.iloc[train_indexes], 
    self.iloc[test_indexes], 
    y.iloc[test_indexes]
  )
  stop('splitting done')
  return new_set

def _df_cv(self, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1):  
  return _df_cv_impl_(self, clf, y, n_samples, n_iter, scoring, n_jobs)

def _df_cv_ohe(self, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1):  
  return _df_cv_impl_(self.one_hot_encode(), clf, y, n_samples, n_iter, scoring, n_jobs)

def _df_cv_impl_(X, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1):  
  if hasattr(y, 'values'): y = y.values
  if n_samples is None: n_samples = len(y)
  else: n_samples = min(n_samples, len(y), X.shape[0])
  if len(y) < X.shape[0]: X = X[:len(y)]
  if utils.multiclass.type_of_target(y) == 'binary' and not (scoring or cfg['scoring']): 
    scoring = 'roc_auc'
  start('starting ' + `n_iter` + ' fold cross validation (' + 
      `n_samples` + ' samples) w/ metric: ' + `scoring or cfg['scoring']`)
  cv = do_cv(clf, X, y, n_samples, n_iter=n_iter, scoring=scoring, quiet=True, n_jobs=n_jobs)
  stop('done cross validation:\n  [CV]: ' + ("{0:.5f} (+/-{1:.5f})").format(cv[0], cv[1]))  
  return cv

def _df_pca(self, n_components, whiten=False):  
  new_X = PCA(n_components, whiten=whiten).fit_transform(self)
  columns = map(lambda i: 'n_pca_' + `i`, range(n_components))
  return pd.DataFrame(columns=columns, data=new_X)

def _df_predict(self, clf, y, X_test=None):    
  reseed(clf)
  X_train = self
  if X_test is None and self.shape[0] > len(y):
    X_test = self[len(y):]
    X_train = self[:len(y)]
  return clf.fit(X_train, y).predict(X_test)

def _df_predict_proba(self, clf, y, X_test=None):    
  reseed(clf)
  X_train = self
  if X_test is None and self.shape[0] > len(y):
    X_test = self[len(y):]
    X_train = self[:len(y)]
  return clf.fit(X_train, y).predict_proba(X_test)

def _df_self_predict(self, clf, y, n_chunks=5):    
  return __df_self_predict_impl(self, clf, y, n_chunks, False)

def _df_self_predict_proba(self, clf, y, n_chunks=5):    
  return __df_self_predict_impl(self, clf, y, n_chunks, True)

def __df_self_predict_impl(X, clf, y, n_chunks, predict_proba):    
  if y is not None and X.shape[0] != len(y): 
    raise Exception('self_predict should have enough y values to do full prediction.')
  start('self_predict with ' + `n_chunks` + ' starting')
  reseed(clf)
  chunk_size = int(math.ceil(X.shape[0] / float(n_chunks)))
  predictions = []
  iteration = 0
  while True:
    begin = iteration * chunk_size
    iteration += 1
    if begin >= X.shape[0]: break
    end = begin + chunk_size
    X_train = X[:begin].append_bottom(X[end:])
    X_test = X[begin:end]
    y2 = None if y is None else pd.concat((y[:begin], y[end:]), 0, ignore_index=True)    

    clf.fit(X_train, y2)    
    new_predictions = clf.predict_proba(X_test) if predict_proba else clf.predict(X_test)    
    if len(new_predictions.shape) > 1 and new_predictions.shape[1] == 1:
      new_predictions = new_predictions.T[1]
    if new_predictions.shape[0] == 1:      
      new_predictions = new_predictions.reshape(-1, 1)
    if iteration == 1:
      predictions = new_predictions
    elif predict_proba:
      predictions = np.vstack((predictions, new_predictions))
    else:
      predictions = np.hstack((predictions, new_predictions))
  stop('self_predict completed')
  return predictions


def _df_trim_on_y(self, y, sigma_or_min_y, max_y=None):    
  X = self.copy()  
  X['__tmpy'] = y.copy()
  if max_y is None:
    X = X[np.abs(X['__tmpy'] - X['__tmpy'].mean()) <= 
      (float(sigma) * X['__tmpy'].std())]
  else:
    X = X[(X['__tmpy'] >= sigma_or_min_y) & (X['__tmpy'] <= max_y)]
  y = X['__tmpy']
  return (X.drop(['__tmpy'], 1), y)

def _df_save_csv(self, file):   
  if file.endswith('.pickle'): 
    dump(file, self)
    return self
  if file.endswith('.gz'): file = gzip.open(file, "wb")
  self.to_csv(file, index=False)  
  return self

def _df_nbytes(self):    
  return self.index.nbytes + self.columns.nbytes + \
    sum(map(lambda c: self[c].nbytes, self.columns))

def _get_optimal_numeric_type(dtype, min, max, aggresiveness=0):
  dtype = str(dtype)
  is_int = dtype.startswith('int')
  if min >= 0 and is_int:
    '''
    uint8 Unsigned integer (0 to 255)
    uint16  Unsigned integer (0 to 65535)
    uint32  Unsigned integer (0 to 4294967295)
    uint64  Unsigned integer (0 to 18446744073709551615)
    '''
    if max <= 255: return 'uint8'
    if max <= 65535: return 'uint16'
    if max <= 4294967295: return 'uint32'
    if max <= 18446744073709551615: return 'uint64'
    raise Exception(`max` + ' is too large')
  elif is_int:
    '''
    int8 Byte (-128 to 127)
    int16 Integer (-32768 to 32767)
    int32 Integer (-2147483648 to 2147483647)
    int64 Integer (-9223372036854775808 to 9223372036854775807)
    '''
    if min >= -128 and max <= 127: return 'int8'
    if min >= -32768 and max <= 32767: return 'int16'
    if min >= -2147483648 and max <= 2147483647: return 'int32'
    if min >= -9223372036854775808 and max <= 9223372036854775807: return 'int64'
    raise Exception(`min` + ' and ' + `max` + ' are out of supported range')
  else:
    '''
    float16 Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    float32 Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    float64 Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    '''
    if not dtype.startswith('float'): raise Exception('Unsupported type: ' + dtype)
    current = int(dtype[-2:])
    if aggresiveness == 0: return dtype
    if aggresiveness == 1: 
      if current == 64: return 'float32'
      elif current <= 32: return 'float16'
      elif current == 16: return 'float16'
      else: raise Exception('Unsupported type: ' + dtype)
    if aggresiveness == 2: return 'float16'  

def _df_compress(self, aggresiveness=0, sparsify=False):  
  start('compressing dataset with ' + `len(self.columns)` + ' columns')    

  def _format_bytes(num):
      for x in ['bytes','KB','MB','GB']:
          if num < 1024.0 and num > -1024.0:
              return "%3.1f%s" % (num, x)
          num /= 1024.0
      return "%3.1f%s" % (num, 'TB')

  original_bytes = self.nbytes()
  # Binary fields compressed just like categoricals
  self.columns = map(lambda c: c.replace('b_', 'c_'), self.columns)
  self.missing(categorical_fill='missing', numerical_fill='none')
  self.toidxs(True)
  for idx, c in enumerate(self.columns): self[c] = self[c].s_compress(aggresiveness, sparsify)
  new_bytes = self.nbytes()
  diff_bytes = original_bytes - new_bytes
  stop('original: %s new: %s improvement: %s percentage: %.2f%%' % 
    (_format_bytes(original_bytes), _format_bytes(new_bytes), 
        _format_bytes(diff_bytes), diff_bytes * 100.0 / original_bytes))
  return self

def _s_hashcode(self):
  hashcode = hash(self.name)
  hashcode = hashcode * 17 + self.index.values.sum()
  hashcode = hashcode * 31 + hash(''.join(map(str, self[0:min(3, self.shape[0])].values)))
  return hashcode

def _df_hashcode(self):
  hashcode = self.index.values.sum()
  hashcode = hashcode * 17 + hash(''.join(self.columns.values))
  hashcode = hashcode * 31 + hash(''.join(map(str, self[0:min(3, self.shape[0])].values)))
  return hashcode

def __df_to_lines(df, 
    out_file_or_y=None, 
    y=None, 
    weights=None, 
    convert_zero_ys=True,
    output_categorical_value=True,
    tag_feature_sets=True):    
  columns_indexes = {}
  max_col = {'index':0}
  out_file = out_file_or_y if type(out_file_or_y) is str else None
  
  if y is None and out_file_or_y is not None and out_file is None: 
    y = out_file_or_y

  def get_col_index(name):
    if name not in columns_indexes:
      columns_indexes[name] = max_col['index']
      max_col['index'] += 1
    return str(columns_indexes[name])

  def impl(outfile):
    def add_cols(new_line, columns, is_numerical):
      if len(columns) == 0: return
      if tag_feature_sets: new_line.append('|' + ('n' if is_numerical else 'c'))
      for c in columns:        
        val = row[c] 
        if val == 0: continue        
        if not is_numerical:
          name = c + '_' + str(val)
          if output_categorical_value: line = get_col_index(name) + ':1'
          else: line = get_col_index(name)
        else:           
          line = get_col_index(c) + ':' + str(val)
        new_line.append(line)
      
    lines = []  
    for idx, row in _chunked_iterator(df):
      label = '1.0' if y is None or idx >= len(y) else str(float(y[idx]))
      if convert_zero_ys and label == '0.0': label = '-1.0'
      if weights is not None and idx < len(weights):      
        w = weights[idx]
        if w != 1: label += ' ' + `w`
        label += ' \'' + `idx`
      
      new_line = [label]      
      
      add_cols(new_line, df.numericals(), True)
      add_cols(new_line, df.categoricals() + df.indexes() + df.binaries(), False)

      line = ' '.join(new_line)
  
      if outfile: outfile.write(line + '\n')
      else: lines.append(line)
    return lines
  
  if out_file:
    with get_write_file_stream(out_file) as outfile:    
      return impl(outfile)
  else: 
    return impl(None)

def _df_to_vw(self, out_file_or_y=None, y=None, weights=None):    
  return __df_to_lines(self, out_file_or_y, y, weights, 
      convert_zero_ys=True,
      output_categorical_value=False,
      tag_feature_sets=True)  

def _df_to_svmlight(self, out_file_or_y=None, y=None):
  return __df_to_lines(self, out_file_or_y, y, None,
      convert_zero_ys=True,
      output_categorical_value=True,
      tag_feature_sets=False)

def _df_to_libfm(self, out_file_or_y=None, y=None):
  return __df_to_lines(self, out_file_or_y, y, None,
      convert_zero_ys=False,
      output_categorical_value=True,
      tag_feature_sets=False)


def _df_summarise(self, opt_y=None, filename='dataset_description', columns=None):
  from describe import describe
  describe.Describe(self, opt_y).show()  

def _chunked_iterator(df, chunk_size=1000000):
  start = 0
  while True:
    subset = df[start:start+chunk_size]
    start += chunk_size
    for r in subset.iterrows():
      yield r    
    if len(subset) < chunk_size: break

# Extensions
def extend_df(name, function):
  df = pd.DataFrame({})
  if not 'pd_extensions' in cfg and hasattr(df, name): raise Exception ('DataFrame already has a ' + name + ' method')
  setattr(pd.DataFrame, name, function)

def extend_s(name, function):
  s = pd.Series([])
  if not 'pd_extensions' in cfg and hasattr(s, name): raise Exception ('Series already has a ' + name + ' method')
  setattr(pd.Series, name, function)

# Data Frame Extensions  
extend_df('one_hot_encode', _df_one_hot_encode)
extend_df('to_indexes', _df_to_indexes)
extend_df('bin', _df_bin)
extend_df('remove', _df_remove)
extend_df('remove_nas', _df_remove_nas)
extend_df('engineer', _df_engineer)
extend_df('combinations', _df_combinations)
extend_df('missing', _df_missing)
extend_df('scale', _df_scale)
extend_df('outliers', _df_outliers)
extend_df('categorical_outliers', _df_categorical_outliers)
extend_df('append_right', _df_append_right)
extend_df('append_bottom', _df_append_bottom)
extend_df('shuffle', _df_shuffle)
extend_df('subsample', _df_subsample)
extend_df('split', _df_split)
extend_df('cv', _df_cv)
extend_df('cv_ohe', _df_cv_ohe)
extend_df('pca', _df_pca)
extend_df('noise_filter', _df_noise_filter)
extend_df('predict', _df_predict)
extend_df('predict_proba', _df_predict_proba)
extend_df('self_predict', _df_self_predict)
extend_df('self_predict_proba', _df_self_predict_proba)
extend_df('save_csv', _df_save_csv)
extend_df('to_vw', _df_to_vw)
extend_df('to_libfm', _df_to_libfm)
extend_df('to_svmlight', _df_to_svmlight)
extend_df('to_xgboost', _df_to_svmlight)
extend_df('hashcode', _df_hashcode)

extend_df('categoricals', _df_categoricals)
extend_df('indexes', _df_indexes)
extend_df('numericals', _df_numericals)
extend_df('dates', _df_dates)
extend_df('binaries', _df_binaries)
extend_df('trim_on_y', _df_trim_on_y)
extend_df('nbytes', _df_nbytes)
extend_df('compress', _df_compress)
extend_df('summarise', _df_summarise)

# Series Extensions  
extend_s('one_hot_encode', _s_one_hot_encode)
extend_s('bin', _s_bin)
extend_s('categorical_outliers', _s_categorical_outliers)
extend_s('sigma_limits', _s_sigma_limits)
extend_s('s_compress', _s_compress)
extend_s('hashcode', _s_hashcode)
extend_s('to_indexes', _s_to_indexes)

# Aliases
extend_s('catout', _s_categorical_outliers)
extend_s('ohe', _s_one_hot_encode)
extend_s('toidxs', _s_to_indexes)

extend_df('ohe', _df_one_hot_encode)
extend_df('toidxs', _df_to_indexes)
extend_df('rm', _df_remove)
extend_df('rmnas', _df_remove_nas)
extend_df('eng', _df_engineer)
extend_df('nas', _df_missing)
extend_df('catout', _df_categorical_outliers)

if not 'pd_extensions' in cfg: cfg['pd_extensions'] = True
