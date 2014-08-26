import pandas as pd
from misc import *
import itertools

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
    c_df.columns = map(lambda c2: c + '[' + c2 + ']', c_df.columns)
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

    columns = name.split('(:)')
    if len(columns) < 2 or len(columns) > 3: 
      raise Exception(name + ' only supports 2 or 3 columns')
    if len(columns) == 2: 
      self[name] = to_obj(columns[0]) + to_obj(columns[1])
    if len(columns) == 3: 
      self[name] = to_obj(columns[0]) + to_obj(columns[1]) + to_obj(columns[2])
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
  

# Data Frame Extensions  
pd.DataFrame.one_hot_encode = _df_one_hot_encode
pd.DataFrame.bin = _df_bin
pd.DataFrame.remove = _df_remove
pd.DataFrame.engineer = _df_engineer
pd.DataFrame.combinations = _df_combinations

pd.DataFrame.categoricals = _df_categoricals
pd.DataFrame.numericals = _df_numericals
pd.DataFrame.dates = _df_dates
pd.DataFrame.binaries = _df_binaries