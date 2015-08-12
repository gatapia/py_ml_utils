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
from sklearn import utils, cross_validation, manifold, cluster
from scipy import sparse
import itertools, random, gzip
from scipy.ndimage.filters import *
from smote import * 

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
  cat = pd.Categorical.from_array(self)
  lbls = cat.codes if hasattr(cat, 'codes') else cat.labels    
  return pd.Series(lbls, index=self.index, dtype=_get_optimal_numeric_type('int', 0, len(lbls) + 1))

def _s_append_bottom(self, s):  
  return pd.concat([self, s], ignore_index=True)

def _s_missing(self, fill='none'):  
  start('replacing series missing data fill[' + `fill` + ']')
  
  val = _get_col_aggregate(self, fill)  
  self.fillna(val, inplace=True)
  if self.name.startswith('n_'): 
    self.replace([np.inf, -np.inf], val, inplace=True)  

  stop('replacing series missing data')
  return self

def _s_scale(self, min_max=None):  
  s = self
  if min_max:
    s -= s.min()  
    s /= s.max()
    s *= (min_max[1] - min_max[0])
    s += min_max[0]
  else:
    s -= s.mean()
    s /= s.std()

  return s

def _s_is_similar(self, other):
  shortest = float(min(len(self), len(other)))
  self, other = self[:int(shortest)], other[:int(shortest)]  
  name = self.name

  if name.startswith('d_'): 
    dbg('date similarity comparison not implemented')
    return True
  if name.startswith('c_') or name.startswith('b_'): self, other = self.to_indexes(), other.to_indexes()

  def _comp(v1, v2, prefix):
    v1, v2 = abs(v1), abs(v2)
    if v2 > v1: v1, v2 = v2, v1
    dissimilarity = (v1 - v2) / float(v1)
    if dissimilarity > .1:
      dbg(name, ':', prefix, 'below threshold [', v1, '], [', v2, '], dissimilarity[', dissimilarity, ']')
      return False
    return True

  if not _comp(np.sum(np.isfinite(self)), np.sum(np.isfinite(other)), 'null/inf'): return False

  if name.startswith('n_'):
    self, other = self.scale(), other.scale()
    rng, rng2 = self.max() - self.min(), other.max() - other.min()
    if not _comp(rng, rng2, 'range'): return False
    if not _comp(self.min(), other.min(), 'min'): return False
    if not _comp(self.max(), other.max(), 'min'): return False
    # if not _comp(self.kurtosis(), other.kurtosis(), 'kurtosis'): return False
  elif name.startswith('c_') or name.startswith('i_') or name.startswith('b_'): 
    vcs, vcs2 = self.value_counts(), other.value_counts()
    for val in vcs.keys():
      c = vcs[val]
      if c < shortest * .05: continue
      if val not in vcs2:
        dbg(name, 'categorical value:', val, 'not in second dataset')
        return False
      c2 = vcs2[val]
      if not _comp(c, c2, 'categorical value: ' + str(val)): return False
  else:
    dbg(name, ': is not supported')
    return False
  return True

'''
DataFrame Extensions
'''

def _df_categoricals(self): return filter(lambda c: c.startswith('c_'), self.columns)
def _df_indexes(self): return filter(lambda c: c.startswith('i_'), self.columns)
def _df_numericals(self): return filter(lambda c: c.startswith('n_'), self.columns)
def _df_binaries(self): return filter(lambda c: c.startswith('b_'), self.columns)
def _df_dates(self): return filter(lambda c: c.startswith('d_'), self.columns)

def _df_infer_col_names(self):
  start('infering column names')
  def _get_prefix(c):
    name = c.name
    dtype = str(c.dtype)
    if len(c.unique()) == 2: return 'b_'
    if dtype.startswith('float') or dtype.startswith('int'): return 'n_'
    if dtype.startswith('date'): return 'd_'
    return 'c_'

  new_cols = []
  for i in range(self.shape[1]):
    col_name = self.columns[i] if self.columns[i][1] == '_' else _get_prefix(self.ix[:,i]) + self.columns[i]
    orig_col_name = col_name
    idx = 1
    while col_name in new_cols:
      col_name = orig_col_name + '_' + `idx`
      idx += 1
    new_cols.append(col_name)
  self.columns = new_cols  
  start('done infering column names')
  return self

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
  
  matrix = ohe_sparse if len(others) == 0 else sparse.hstack((ohe_sparse, others_df))
  stop('done one_hot_encoding')
  return matrix.tocsr()

def _df_to_indexes(self, drop_origianls=False, sparsify=False):
  start('indexing categoricals in data frame. Note: NA gets turned into max index (255, 65535, etc)')  
  for c in self.categoricals() + self.binaries():
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

def _df_cats_to_count_ratios(self, y):
  all_cats = self.categoricals() + self.indexes() + self.binaries()
  renamed_cats = [c.replace('c_', 'n_') for c in all_cats]

  w_counts = _df_cats_to_counts(self, y)
  for orig_cat in renamed_cats:
    cols = [c for c in w_counts.columns if c.startswith(orig_cat + '_y:')]
    for c in cols:
      w_counts[c] /= float(len(y))
  return w_counts


def _df_cats_to_counts(self, y):
  start('converting categoricals to counts')
  self['__y'] = y.values if hasattr(y, 'values') else y
  unique_ys = y.unique()
  unique_ys.sort()
  all_cats = self.categoricals() + self.indexes() + self.binaries()
  for c in all_cats:
    piv = self[[c, '__y']].pivot_table(index=c, columns='__y', aggfunc=len, fill_value=0)
    piv.columns = [c.replace('c_', 'n_') + '_y:' + str(y_val) for y_val in unique_ys]
    self = self.merge(piv, 'left', left_on=c, right_index=True)  

  self = self.remove(['__y'] + all_cats)
  stop('done converting categoricals to counts')
  return self

def _df_cats_to_means(self, y):
  raise Exception('not implemented')

def _df_bin(self, n_bins=100, drop_origianls=False):
  start('binning data into ' + `n_bins` + ' bins')  
  for n in self.numericals():
    self['c_binned_' + n] = pd.Series(pd.cut(self[n], n_bins), index=self[n].index)
    if drop_origianls: self.drop(n, 1, inplace=True)
  stop('done binning data into ' + `n_bins` + ' bins')  
  return self

def _df_group_rare(self, columns=None, limit=30):
  start('grouping rare categorical columns, limit: ' + `limit`)  
  if columns is None: columns = self.categoricals()
  for c in columns:
    vcs = self[c].value_counts()
    rare = vcs[vcs < limit].keys()  
    self.loc[self[c].isin(rare), c] = 'rare'
  start('done grouping rare categorical')  
  return self

def _df_combinations(self, group_size=2, columns=[], categoricals=False, indexes=False,
    numericals=False, dates=False, binaries=False, permutations=False):
  cols = list(columns)
  if categoricals: cols = cols + self.categoricals()
  if indexes: cols = cols + self.indexes()
  if numericals: cols = cols + self.numericals()
  if dates: cols = cols + self.dates()
  if binaries: cols = cols + self.binaries()
  op = itertools.permutations if permutations else itertools.combinations
  return list(op(cols, group_size))

def _df_normalise(self, columns=None):
  start('normalising data [0-1]')  
  if columns is None: columns = self.numericals()
  for c in columns:
    self[c] -= self[c].min()
    self[c] /= self[c].max()
  stop('done normalising data')  
  return self

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
  cols = [c for c in cols if c in self.columns]
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
  if len(args) == 0 and (func == 'avg' or func == 'mult' or func == 'add' or func == 'concat'):    
    combs = list(itertools.combinations(columns, 2)) if columns is not None \
      else self.combinations(categoricals=func=='concat', indexes=func=='concat', numericals=func in ['mult', 'avg', 'add'])    
    for c1, c2 in combs: self.engineer(func + '(' + c1 + ',' + c2 + ')', quiet=True)
    return self
  if len(args) == 0 and (func == 'div' or func == 'subtract'):
    combs = list(itertools.combinations(columns, 2, permutations=True)) if columns is not None \
      else self.combinations(numericals=True, permutations=True)    
    for c1, c2 in combs: self.engineer(func + '(' + c1 + ',' + c2 + ')', quiet=True)
    return self
  elif func == 'concat': 
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = self[args[0]].astype(str) + self[args[1]].astype(str)
    if len(args) == 3: 
      self[new_name] = self[args[0]].astype(str) + self[args[1]].astype(str) + self[args[2]].astype(str)
  elif func  == 'mult' or func  == 'add':     
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    s1, s2 = self[args[0]], self[args[1]]
    if len(args) == 2: 
      self[new_name] = s1 * s2 if func == 'mult' else s1 + s2
    if len(args) == 3: 
      s3 = self[args[2]]
      self[new_name] = s1 * s2 * s3 if func == 'mult' else s1 + s2 + s3
  elif func  == 'div' or func  == 'subtract':     
    if len(args) != 2: raise Exception(name + ' only supports 2 columns')
    s1, s2 = self[args[0]].astype(float), self[args[1]].astype(float)
    self[new_name] = s1 / s2 if func == 'div' else s1 - s2    
  elif func  == 'avg':     
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = (self[args[0]] + self[args[1]]) / 2
    if len(args) == 3: 
      self[new_name] = (self[args[0]] + self[args[1]] + self[args[2]]) / 3
  elif len(args) == 1 and func == 'pow':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('pow(' + n + ', ' + args[0] + ')', quiet=True)
    return self
  elif len(args) == 1 and func == 'round':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('round(' + n + ', ' + args[0] + ')', quiet=True)
    return self
  elif len(args) == 0 and func == 'lg':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('lg(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'safe_lg':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('safe_lg(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'sqrt':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('sqrt(' + n + ')', quiet=True)    
    return self
  elif func == 'pow': 
    self[new_name] = np.power(self[args[0]], int(args[1]))
  elif func == 'round': 
    self[new_name] = self[args[0]].round(int(args[1]))
  elif func == 'lg': 
    self[new_name] = np.log(self[args[0]])
  elif func == 'safe_lg': 
    self[new_name] = np.log(self[args[0]] + 1 - self[args[0]].min())
  elif func == 'sqrt': 
    self[new_name] = np.sqrt(self[args[0]])
  elif func.startswith('rolling_'):
    if len(args) == 1:
      cols = columns if columns is not None else self.numericals()
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
  cols = columns if columns is not None else self.numericals()
  for c in cols:
    self[c] = self[c].scale(min_max)
  stop('scaling data frame')
  return self

def _df_missing(self, categorical_fill='none', numerical_fill='none', binary_fill='none'):  
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
  binaries_to_fill = []
  if categorical_fill != 'none': categoricals_to_fill += self.categoricals() + self.indexes()
  if numerical_fill != 'none': numericals_to_fill += self.numericals()
  if binary_fill != 'none': binaries_to_fill += self.binaries()

  # Prepare a dictionary of column -> fill values
  to_fill = {}
  for c in binaries_to_fill: to_fill[c] = _get_col_aggregate(self[c], binary_fill)
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
    dbg(self.name + ' is not supported, ignored during compression')
  return self

def _s_hashcode(self):
  index = tuple(self.index)
  values = tuple(tuple(x) for x in self.values)
  item = tuple([index, values])
  return hash(item)

def _s_to_ratio(self, y, positive_class=None):
  classes = y.unique()
  if len(classes) != 2: raise Exception('only binary target is supported')
  if positive_class is None: positive_class = classes[0]
  for val in self.unique():
    this_y = y[self == val]    
    ratio = len(this_y[this_y == positive_class]) / float(len(this_y))
    self[self==val] = ratio
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
  start('appending to the right.  note, this is a destructive operation')
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
    if type(df_or_s) is pd.DataFrame:
      columns = df_or_s.columns.tolist()
      right = df_or_s.values
    else:
      columns = [`i` + '_2' for i in range(df_or_s.shape[1])]
      right = df_or_s
    self = pd.DataFrame(np.hstack((self.values, right)), 
        columns=self.columns.tolist() + columns)
  stop('done appending to the right')
  return self

def _df_append_bottom(self, df):  
  # debug('warning: DataFrame.append_bottom always returns a new DataFrame')
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
  if stratified:
    train_indexes, test_indexes = list(cross_validation.StratifiedShuffleSplit(y, 1, test_size, train_size, random_state=cfg['sys_seed']))[0]  
  else:
    train_indexes, test_indexes = list(cross_validation.ShuffleSplit(len(y), 1, test_size, train_size, random_state=cfg['sys_seed']))[0]  
  new_set = (
    self.iloc[train_indexes], 
    y.iloc[train_indexes], 
    self.iloc[test_indexes], 
    y.iloc[test_indexes]
  )
  stop('splitting done')
  return new_set

def _df_cv(self, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1, fit_params=None):  
  return _df_cv_impl_(self, clf, y, n_samples, n_iter, scoring, n_jobs, fit_params)

def _df_cv_ohe(self, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1, fit_params=None):  
  return _df_cv_impl_(self.one_hot_encode(), clf, y, n_samples, n_iter, scoring, n_jobs, fit_params)

def _df_cv_impl_(X, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1, fit_params=None):    
  if hasattr(y, 'values'): y = y.values
  if n_samples is None: n_samples = len(y)
  else: n_samples = min(n_samples, len(y), X.shape[0])
  if len(y) < X.shape[0]: X = X[:len(y)]
  if utils.multiclass.type_of_target(y) == 'binary' and not (scoring or cfg['scoring']): 
    scoring = 'roc_auc'
  start('starting ' + `n_iter` + ' fold cross validation (' + 
      `n_samples` + ' samples) w/ metric: ' + `scoring or cfg['scoring']`)
  cv = do_cv(clf, X, y, n_samples, n_iter=n_iter, scoring=scoring, quiet=True, n_jobs=n_jobs, fit_params=fit_params)
  stop('done cross validation:\n  [CV]: ' + ("{0:.5f} (+/-{1:.5f})").format(cv[0], cv[1]))  
  return cv

def _df_pca(self, n_components, whiten=False):  
  new_X = PCA(n_components, whiten=whiten).fit_transform(self)
  columns = map(lambda i: 'n_pca_' + `i`, range(n_components))
  return pd.DataFrame(columns=columns, data=new_X)

def _df_tsne(self, n_components):  
  new_X = manifold.TSNE(n_components, method='barnes_hut').fit_transform(self)
  columns = map(lambda i: 'n_tsne_' + `i`, range(n_components))
  return pd.DataFrame(columns=columns, data=new_X)

def _df_kmeans(self, k):  
  return pd.Series(cluster.KMeans(k).fit_predict(self))

def _df_tree_features(self, tree_ensemble, y):
  def _make_tree_bins(clf, X):
    nd_mat = None
    X32 = np.array(X).astype(np.float32)
    for i, tt in enumerate(clf.estimators_):
      tt = tt.tree_ if hasattr(tt, 'tree_') else tt[0].tree_
      nds = tt.apply(X32)
      if i == 0:  nd_mat = nds.reshape(len(nds), 1)        
      else: nd_mat = np.hstack((nd_mat, nds.reshape(len(nds), 1)))
    return nd_mat

  def op(X, y, X2): 
    return _make_tree_bins(tree_ensemble.fit(X, y), X2)

  tree_features = self.self_chunked_op(y, op)
  tree_features.columns = ['i_c_tree_feature_' + `i+1` for i in range(tree_features.shape[1])]  
  return tree_features

def _df_append_fit_transformer(self, fit_transformer, method='fit_transform'):  
  new_X = getattr(fit_transformer, method)(self)
  columns = map(lambda i: 'n_new_col_' + `i`, range(new_X.shape[1]))
  new_df = pd.DataFrame(columns=columns, data=new_X)
  return self.copy().append_right(new_df)

def _df_predict(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'predict')

def _df_predict_proba(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'predict_proba')

def _df_transform(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'transform')

def _df_decision_function(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'decision_function')

def __df_clf_method_impl(X, clf, y, X_test, method):    
  reseed(clf)
  X_train = X
  if X_test is None and X.shape[0] > len(y):
    X_test = X[len(y):]
    X_train = X[:len(y)]    

  if len(X_train.shape) == 2 and X_train.shape[1] == 1: X_train = X_train.ix[:,0]
  if len(X_test.shape) == 2 and X_test.shape[1] == 1: X_test = X_test.ix[:,0]
  clf.fit(X_train, y)
  return getattr(clf, method)(X_test)

def _df_self_predict(self, clf, y, cv=5):    
  return _df_self_predict_impl(self, clf, y, cv, 'predict')

def _df_self_predict_proba(self, clf, y, cv=5):    
  return _df_self_predict_impl(self, clf, y, cv, 'predict_proba')

def _df_self_transform(self, clf, y, cv=5):    
  return _df_self_predict_impl(self, clf, y, cv, 'transform')

def _df_self_decision_function(self, clf, y, cv=5):    
  return _df_self_predict_impl(self, clf, y, cv, 'decision_function')

def _df_self_predict_impl(X, clf, y, cv, method):    
  if y is not None and X.shape[0] != len(y): X = X[:len(y)]
  start('self_' + method +' with ' + `cv` + ' chunks starting')
  reseed(clf)
      
  def op(X, y, X2):
    if len(X.shape) == 2 and X.shape[1] == 1: 
      if hasattr(X, 'values'): X = X.values
      X = X.T[0]
    if len(X2.shape) == 2 and X2.shape[1] == 1: 
      if hasattr(X2, 'values'): X2 = X2.values
      X2 = X2.T[0]
    
    clf.fit(X, y)  
    new_predictions = getattr(clf, method)(X2)
    if new_predictions.shape[0] == 1:      
      new_predictions = new_predictions.reshape(-1, 1)
    return new_predictions    
  
  predictions = _df_self_chunked_op(X, y, op, cv)
  stop('self_predict completed')
  return predictions.values

def _df_self_chunked_op(self, y, op, cv=5):    
  if y is not None and hasattr(y, 'values'): y = y.values
  X = self
  if cv is None: cv = 5
                          # cross_validation.KFold(len(y), cv, shuffle=True, random_state=cfg['sys_seed'])
  if type(cv) is int: cv = cross_validation.StratifiedKFold(y, cv, shuffle=True, random_state=cfg['sys_seed'])
  indexes=None
  chunks=None
  for train_index, test_index in cv:
    X_train = X.iloc[train_index]
    y_train = y[train_index]
    X_test = X.iloc[test_index]
    predictions = op(X_train, y_train, X_test)
    indexes = test_index if indexes is None else np.concatenate((indexes, test_index))
    chunks = predictions if chunks is None else np.concatenate((chunks, predictions))
  df = pd.DataFrame(data=chunks, index=indexes)
  return df.sort()  

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

def _df_save_csv(self, file, header=True):   
  if file.endswith('.pickle'): 
    dump(file, self)
    return self
  if file.endswith('.gz'): file = gzip.open(file, "wb")
  self.to_csv(file, index=False, header=header)  
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

def _df_hashcode(self, opt_y=None):
  if opt_y is not None: self['_tmpy_'] = opt_y
  index = tuple(self.index)
  columns = tuple(self.columns)
  values = tuple(tuple(x) for x in self.values)
  item = tuple([index, columns, values])
  hash_val = hash(item)
  if opt_y is not None: self.remove('_tmpy_')
  return hash_val

def __df_to_lines(df, 
    out_file_or_y=None, 
    y=None, 
    weights=None, 
    convert_zero_ys=True,
    output_categorical_value=True,
    tag_feature_sets=True,
    col_index_start=0,
    sort_feature_indexes=False):    
  columns_indexes = {}
  max_col = {'index':0}
  out_file = out_file_or_y if type(out_file_or_y) is str else None
  
  if y is None and out_file_or_y is not None and out_file is None: 
    y = out_file_or_y
  if y is not None and hasattr(y, 'values'): y = y.values

  def get_col_index(name):
    if name not in columns_indexes:
      columns_indexes[name] = max_col['index']
      max_col['index'] += 1
    return str(col_index_start + columns_indexes[name])

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
          else: 
            line = get_col_index(name)
        else:           
          line = get_col_index(c) + ':' + str(val)
        new_line.append(line)
      
    lines = []  
    for idx, row in enumerate(_chunked_iterator(df)):
      label = '1.0' if y is None or idx >= len(y) else str(float(y[idx]))
      if convert_zero_ys and label == '0.0': label = '-1.0'
      if weights is not None and idx < len(weights):      
        w = weights[idx]
        if w != 1: label += ' ' + `w`
        label += ' \'' + `idx`
      
      new_line = []
      add_cols(new_line, df.numericals(), True)
      add_cols(new_line, df.categoricals() + df.indexes() + df.binaries(), False)
      if sort_feature_indexes:
        new_line = sorted(new_line, key=lambda v: int(v.split(':')[0]))
      line = ' '.join([label] + new_line)
  
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
      tag_feature_sets=False,
      col_index_start=1,
      sort_feature_indexes=True)

def _df_to_libfm(self, out_file_or_y=None, y=None):
  return __df_to_lines(self, out_file_or_y, y, None,
      convert_zero_ys=False,
      output_categorical_value=True,
      tag_feature_sets=False)

def _df_to_libffm(self, out_file_or_y=None, y=None):
  out_file = out_file_or_y if type(out_file_or_y) is str else None  
  if y is None and out_file_or_y is not None and out_file is None: 
    y = out_file_or_y
  if hasattr(y, 'values'): y = y.values
  
  lines = []    
  outfile = None
  if out_file is not None: outfile = get_write_file_stream(out_file)
  categoricals = self.categoricals() + self.indexes() + self.binaries()
  if len(categoricals) > 0: raise Exception('categoricals not currently supported')
  numericals = self.numericals()
  for idx, row in enumerate(_chunked_iterator(self)):
    label = '0' if y is None or idx >= len(y) else str(int(y[idx]))
    new_line = [label]     
    for col_idx, c in enumerate(numericals):
      cis = str(col_idx)
      new_line.append(cis + ':0:' + `0 if row[c] == 0  else row[c]`)
    line = ' '.join(new_line)
    if outfile is not None: outfile.write(line + '\n')
    else: lines.append(line)
  if outfile:
    outfile.close()
    return None
  else: return lines

def _df_summarise(self, opt_y=None, filename='dataset_description', columns=None):
  from describe import describe
  describe.Describe().show_dataset(self, opt_y)  

def _df_importances(self, clf, y):
  clf.fit(self[:len(y)], y)
  if hasattr(clf, 'feature_importances_'): imps = clf.feature_importances_ 
  else: imps = map(abs, clf.coef_[0])
  top_importances_indexes = np.argsort(imps)[::-1]
  top_importances_values = np.array(imps)[top_importances_indexes]
  top_importances_features = self.columns[top_importances_indexes]
  return zip(top_importances_features, top_importances_values)

def _df_is_similar(self, other):
  in_X = self.columns - other.columns
  if len(in_X) > 0:
    dbg('columns found in main dataset not in second: ', in_X)
    return False
  in_X2 = other.columns - self.columns

  if len(in_X) > 0:
    dbg('columns found in second dataset not in main: ', in_X2)
    return False
  
  return np.all([is_similar_s(self[c], other[c]) for c in self.columns])

def _df_numerical_stats(self, columns=None):
  X2 = self[columns if columns is not None else self.numericals()]
  self['n_min'] = X2.min(1)
  self['n_max'] = X2.max(1)
  self['n_kurt'] = X2.kurt(1)
  self['n_mad'] = X2.mad(1)
  self['n_mean'] = X2.mean(1)
  self['n_median'] = X2.median(1)
  self['n_sem'] = X2.sem(1)
  self['n_std'] = X2.std(1)
  self['n_sum'] = X2.sum(1)
  return self

def _df_smote(self, y, percentage_multiplier, n_neighbors, opt_target=None):
  seed(0)
  vcs = y.value_counts(dropna=False)
  if len(vcs) != 2: raise Exception('DataFrame.smote only works on binary classifiers')
  min_value = opt_target if opt_target is not None else vcs.argmin()
  minorities = self[y == min_value]

  new_minorities = SMOTE(minorities.values, percentage_multiplier, n_neighbors)
  new_len = self.shape[0] + new_minorities.shape[0]
  y2 = pd.Series(np.append(y.values, np.array([min_value] * len(new_minorities))), index=np.arange(new_len))
  minorities_df = pd.DataFrame(new_minorities, columns=self.columns)
  new_df = self.copy().append_bottom(minorities_df)
  new_df.index = np.arange(new_len)
  return (new_df, y2)

def _df_to_ratio(self, y, positive_class=None):
  start('converting all categoricals to ratios')
  for c in self.categoricals() + self.indexes() + self.binaries():
    self[c].to_ratio(y, positive_class)
  stop('done converting all categoricals to ratios')
  return self

def _chunked_iterator(df, chunk_size=1000000):
  start = 0
  while True:
    subset = df[start:start+chunk_size]
    start += chunk_size
    for r in subset.iterrows():
      yield r[1]
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
extend_df('normalise', _df_normalise)
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
extend_df('tsne', _df_tsne)
extend_df('kmeans', _df_kmeans)
extend_df('tree_features', _df_tree_features)
extend_df('append_fit_transformer', _df_append_fit_transformer)
extend_df('noise_filter', _df_noise_filter)
extend_df('predict', _df_predict)
extend_df('predict_proba', _df_predict_proba)
extend_df('transform', _df_transform)
extend_df('decision_function', _df_decision_function)
extend_df('self_predict', _df_self_predict)
extend_df('self_predict_proba', _df_self_predict_proba)
extend_df('self_transform', _df_self_transform)
extend_df('self_decision_function', _df_self_decision_function)
extend_df('self_chunked_op', _df_self_chunked_op)
extend_df('save_csv', _df_save_csv)
extend_df('to_vw', _df_to_vw)
extend_df('to_libfm', _df_to_libfm)
extend_df('to_libffm', _df_to_libffm)
extend_df('to_svmlight', _df_to_svmlight)
extend_df('to_xgboost', _df_to_svmlight)
extend_df('hashcode', _df_hashcode)
extend_df('importances', _df_importances)

extend_df('categoricals', _df_categoricals)
extend_df('indexes', _df_indexes)
extend_df('numericals', _df_numericals)
extend_df('dates', _df_dates)
extend_df('binaries', _df_binaries)
extend_df('trim_on_y', _df_trim_on_y)
extend_df('nbytes', _df_nbytes)
extend_df('compress', _df_compress)
extend_df('summarise', _df_summarise)

extend_df('cats_to_count_ratios', _df_cats_to_count_ratios)
extend_df('cats_to_counts', _df_cats_to_counts)
extend_df('infer_col_names', _df_infer_col_names)
extend_df('group_rare', _df_group_rare)
extend_df('is_similar', _df_is_similar)
extend_df('numerical_stats', _df_numerical_stats)
extend_df('smote', _df_smote)
extend_df('to_ratio', _df_to_ratio)

# Series Extensions   
extend_s('one_hot_encode', _s_one_hot_encode)
extend_s('missing', _s_missing)
extend_s('bin', _s_bin)
extend_s('categorical_outliers', _s_categorical_outliers)
extend_s('sigma_limits', _s_sigma_limits)
extend_s('s_compress', _s_compress)
extend_s('hashcode', _s_hashcode)
extend_s('to_indexes', _s_to_indexes)
extend_s('append_bottom', _s_append_bottom)
extend_s('scale', _s_scale)
extend_s('is_similar', _s_is_similar)
extend_s('to_ratio', _s_to_ratio)

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
