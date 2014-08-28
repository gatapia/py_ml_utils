import pandas as pd
import numpy as np
from misc import *
from sklearn.preprocessing import OneHotEncoder
import itertools, logging, time, datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
t0 = time.time()

def debug(msg): log.info(msg)

def start(msg): 
  global t0
  t0 = time.time()
  log.info(msg)

def stop(msg): 
  global t0
  log.info(msg + (', took (h:m:s): %s' % 
    datetime.timedelta(seconds=time.time() - t0)))
  t0 = time.time()

'''
Series Extensions
'''
def _s_one_hot_encode(self):
  start('one_hot_encoding column')  
  df = pd.get_dummies(self)
  stop('done one_hot_encoding column converted to ' + `df.shape[1]` + ' columns')  
  return df

def _s_bin(self, n_bins=100):
  return pd.Series(pd.cut(self, n_bins))

'''
DataFrame Extensions
'''

def _df_categoricals(self):
  return filter(lambda c: c.startswith('c_'), self.columns)

def _df_numericals(self):
  return filter(lambda c: c.startswith('n_'), self.columns)

def _df_binaries(self):
  return filter(lambda c: c.startswith('b_'), self.columns)

def _df_dates(self):
  return filter(lambda c: c.startswith('d_'), self.columns)

def _df_one_hot_encode(self):
  start('one_hot_encoding data frame with ' + `self.shape[1]` + ' columns')  
  df = self.copy()
  df = df.to_indexes(True)
  cols = list(df.columns)
  cats = map(lambda c: 'n_' + c + '_indexes', self.categoricals())  
  cat_idxs = map(lambda c: cols.index(c), cats)
  np_arr = OneHotEncoder(categorical_features=cat_idxs).fit_transform(df)
  stop('done one_hot_encoding data frame with ' + `np_arr.shape[1]` + ' columns')  
  return np_arr

def _df_to_indexes(self, drop_origianls=False):
  start('indexing categoricals in data frame')  
  for c in self.categoricals():
    self['n_' + c + '_indexes'] = pd.Series(pd.Categorical.from_array(self[c]).labels)
    if drop_origianls: self.drop(c, 1, inplace=True)
  stop('done indexing categoricals in data frame')  
  return self

def _df_bin(self, n_bins=100, drop_origianls=False):
  start('binning data into ' + `n_bins` + ' bins')  
  for n in self.numericals():
    self['c_binned_' + n] = pd.cut(self[n], n_bins)
    if drop_origianls: self.drop(n, 1, inplace=True)
  stop('done binning data into ' + `n_bins` + ' bins')  
  return self

def _df_combinations(self, group_size=2, categoricals=False, numericals=False, 
    dates=False, binaries=False):
  def cmb(columns): return list(itertools.combinations(columns, group_size))
  
  cols = []  
  if categoricals: cols = cols + self.categoricals()
  if numericals: cols = cols + self.numericals()
  if dates: cols = cols + self.dates()
  if binaries: cols = cols + self.binaries()
  return cmb(cols)

def _df_remove(self, columns=[], categoricals=False, numericals=False, 
    dates=False, binaries=False, missing_threshold=0.0):  
  cols = list(columns)
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

  debug('removing from data frame: ' + `cols`)
  self.drop(cols, 1, inplace=True)
  return self

def _df_engineer(self, name, quiet=False):
  if not quiet: debug('engineering feature: ' + name)
  if '(:)' == name:       
    for c1, c2 in self.combinations(categoricals=True):  
      self.engineer(c1 + '(:)' + c2, quiet=True)
  elif '(:)' in name: 
    def to_obj(col):
      if not col in self: raise Exception('could not find "' + col + '" in data frame')
      return self[col] if self[col].dtype == 'object' else self[col].astype('str')
    new_name = name if name.startswith('c_') else 'c_' + name
    columns = name.split('(:)')
    if len(columns) < 2 or len(columns) > 3: 
      raise Exception(name + ' only supports 2 or 3 columns')
    if len(columns) == 2: 
      self[new_name] = to_obj(columns[0]) + to_obj(columns[1])
    if len(columns) == 3: 
      self[new_name] = to_obj(columns[0]) + to_obj(columns[1]) + to_obj(columns[2])
  elif '(*)' == name: 
    for n1, n2 in self.combinations(numericals=True):  
      self.engineer(n1 + '(*)' + n2, quiet=True)
  elif '(*)' in name: 
    columns = name.split('(*)')
    if len(columns) < 2 or len(columns) > 3: 
      raise Exception(name + ' only supports 2 or 3 columns')
    if len(columns) == 2: 
      self[name] = self[columns[0]] * self[columns[1]]
    if len(columns) == 3: 
      self[name] = self[columns[0]] * self[columns[1]] * self[columns[2]]
  elif '(^2)' == name:
    for n in self.numericals():
      self[n + '(^2)'] = self[n] * self[n]
  elif '(lg)' == name:
    for n in self.numericals():
      self[n + '(lg)'] = np.log(self[n])
    self.replace([np.inf, -np.inf], 0, inplace=True)
    self.fillna(0., inplace=True)
  elif '(^2)' in name:
    n = name.split('(')[0]
    self[name] = self[n] * self[n]
  elif '(lg)' in name:
    n = name.split('(')[0]
    self[name] = np.log(self[n]).replace([np.inf, -np.inf], 0).fillna(0.)
  else: raise Exception(name + ' is not supported')
  return self
  
def _df_scale(self, min_max=None, drop_origianls=False):  
  start('scaling data frame')
  pp = preprocessing
  scaler = pp.MinMaxScaler(min_max) if min_max else pp.StandardScaler()
  cols = self.numericals()
  new_cols = map(lambda n: n + '_scaled' if drop_origianls else n, cols)
  self[new_cols] = pd.DataFrame(scaler.fit_transform(self[cols].values), index=self.index)
  stop('scaling data frame')
  return self

def _df_missing(self, categorical_fill='none', numerical_fill='none'):  
  start('replacing missing data categorical[' + `categorical_fill` + '] numerical[' + `numerical_fill` + ']')
  for c in self.columns: 
    fill_mode = 'none'
    if c in self.categoricals(): fill_mode = categorical_fill
    elif c in self.numericals(): fill_mode = numerical_fill    
    if fill_mode == 'none': continue
    self[c] = self[c].fillna(_get_col_aggregate(self[c], fill_mode))
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

def _df_categorical_outliers(self, min_size=0.01):    
  threshold = float(len(self)) * min_size if type(min_size) is float else min_size
  start('binning categorical outliers, threshold: ' + `threshold`)

  for c in self.categoricals(): 
    vc = self[c].value_counts()
    for v, cnt in zip(vc.index.values, vc.values):      
      if cnt < threshold: 
        self[c][self[c] == v] = 'others'
  stop('done binning categorical outliers')
  return self

def _df_append_right(self, df_or_s):  
  if ((type(df_or_s) is pd.sparse.frame.SparseDataFrame or 
      type(df_or_s) is pd.sparse.series.SparseSeries) and 
      not type(self) is pd.sparse.frame.SparseDataFrame):
    debug('converting data frame to a sparse frame')
    self = self.to_sparse(fill_value=0)
  if type(df_or_s) is pd.Series: self[df_or_s.name] = df_or_s.values
  else: self = pd.concat((self, df_or_s), 1)
  return self

def _df_append_bottom(self, df):  
  debug('warning: DataFrame.append_bottom always returns a new DataFrame')
  return pd.concat((self, df), 0)

def _df_shuffle(self, y):  
  start('shuffling data frame')
  new_X, new_y = utils.shuffle(self, y, random_state=sys_seed)
  start('done, shuffling data frame')
  return (pd.DataFrame(columns=self.columns, data=new_X), pd.Series(new_y))

# Data Frame Extensions  
pd.DataFrame.one_hot_encode = _df_one_hot_encode
pd.DataFrame.to_indexes = _df_to_indexes
pd.DataFrame.bin = _df_bin
pd.DataFrame.remove = _df_remove
pd.DataFrame.engineer = _df_engineer
pd.DataFrame.combinations = _df_combinations
pd.DataFrame.missing = _df_missing
pd.DataFrame.scale = _df_scale
pd.DataFrame.outliers = _df_outliers
pd.DataFrame.categorical_outliers = _df_categorical_outliers
pd.DataFrame.append_right = _df_append_right
pd.DataFrame.append_bottom = _df_append_bottom
pd.DataFrame.shuffle = _df_shuffle

pd.DataFrame.categoricals = _df_categoricals
pd.DataFrame.numericals = _df_numericals
pd.DataFrame.dates = _df_dates
pd.DataFrame.binaries = _df_binaries

# Series Extensions  
pd.Series.one_hot_encode = _s_one_hot_encode
pd.Series.bin = _s_bin

# Aliases
pd.DataFrame.ohe = _df_one_hot_encode
pd.DataFrame.toidxs = _df_to_indexes
pd.DataFrame.rm = _df_remove
pd.DataFrame.eng = _df_engineer
pd.DataFrame.nas = _df_missing