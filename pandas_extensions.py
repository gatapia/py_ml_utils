import pandas as pd
from misc import *
import itertools

'''
Series Extensions
'''
def _s_one_hot_encode(self):
  return pd.get_dummies(self)

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

def _df_one_hot_encode(self, drop_origianls=False):
  for c in self.categoricals():
    c_df = pd.get_dummies(self[c])
    c_df.columns = map(lambda c2: 'b_' + c + '[' + c2 + ']', c_df.columns)
    if drop_origianls: self.drop(c, 1, inplace=True)
    self = pd.concat([self, c_df], axis=1)
  return self

def _df_bin(self, n_bins=100, drop_origianls=False):
  for n in self.numericals():
    self['c_binned_' + n] = pd.cut(self[n], n_bins)
    if drop_origianls: self.drop(n, 1, inplace=True)
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

def _df_remove(self, categoricals=False, numericals=False, 
    dates=False, binaries=False):
  if sum([categoricals, numericals, dates, binaries]) == 0: 
    raise Exception('At least one of categoricals, numericals, ' +
      'dates binaries should be set to True')
  cols = []
  if categoricals: cols = cols + self.categoricals()
  if numericals: cols = cols + self.numericals()
  if dates: cols = cols + self.dates()
  if binaries: cols = cols + self.binaries()
  self.drop(cols, 1, inplace=True)

def _df_engineer(self, name):
  if '(:)' in name: 
    def to_obj(col):
      return self[col] if self[col].dtype == 'object' else self[col].astype('str')
    new_name = name if name.startswith('c_') else 'c_' + name
    columns = name.split('(:)')
    if len(columns) < 2 or len(columns) > 3: 
      raise Exception(name + ' only supports 2 or 3 columns')
    if len(columns) == 2: 
      self[new_name] = to_obj(columns[0]) + to_obj(columns[1])
    if len(columns) == 3: 
      self[new_name] = to_obj(columns[0]) + to_obj(columns[1]) + to_obj(columns[2])
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
  elif '(^2)' in name:
    n = name.split('(')[0]
    self[name] = self[n] * self[n]
  elif '(lg)' in name:
    n = name.split('(')[0]
    self[name] = np.log(self[n])
  else: raise Exception(name + ' is not supported')
  return self
  
def _df_scale(self, min_max=None, drop_origianls=False):  
  pp = preprocessing
  scaler = pp.MinMaxScaler(min_max) if min_max else pp.StandardScaler()
  cols = self.numericals()
  new_cols = map(lambda n: n + '_scaled' if drop_origianls else n, cols)
  self[new_cols] = pd.DataFrame(scaler.fit_transform(self[cols].values), index=self.index)
  return self

def _df_missing(self, categorical_fill='none', numerical_fill='none'):  
  for c in self.columns: 
    fill_mode = 'none'
    if c in self.categoricals(): fill_mode = categorical_fill
    elif c in self.numericals(): fill_mode = numerical_fill    
    if fill_mode == 'none': continue
    self[c] = self[c].fillna(_get_col_aggregate(self[c], fill_mode))
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
  for n in self.numericals(): 
    col = self[n]
    mean, offset = col.mean(), stds * col.std()
    min, max = mean - offset, mean + offset
    self[n] = col.clip(min, max)
  return self

def _df_categorical_outliers(self, min_size=0.01):  
  threshold = float(len(self)) * min_size
  for c in self.categoricals(): 
    vc = self[c].value_counts()
    for v, cnt in zip(vc.index.values, vc.values):      
      if cnt < threshold: 
        print 'v:', v, 'cnt:', cnt, 'threshold:', threshold
        self[c][self[c] == v] = 'others'
  return self

def _df_append_right(self, df_or_s):  
  if type(df_or_s) is pd.Series: self[df_or_s.name] = df_or_s.values
  else: 
    for col in df_or_s.columns: self[col] = df_or_s[col].values
  return self

def _df_append_bottom(self, df):  
  print 'warning: DataFrame.append_bottom always returns a new DataFrame'
  return pd.concat((self, df), 0)

# Data Frame Extensions  
pd.DataFrame.one_hot_encode = _df_one_hot_encode
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

pd.DataFrame.categoricals = _df_categoricals
pd.DataFrame.numericals = _df_numericals
pd.DataFrame.dates = _df_dates
pd.DataFrame.binaries = _df_binaries

# Series Extensions  
pd.Series.one_hot_encode = _s_one_hot_encode
pd.Series.bin = _s_bin
