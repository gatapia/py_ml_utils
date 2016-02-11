import pandas as pd, numpy as np
import sklearn, datetime, utils, scipy
from .. import misc

def _s_one_hot_encode(self):
  misc.start('one_hot_encoding column')  

  arr = self.values
  col_ohe = sklearn.preprocessing.OneHotEncoder().fit_transform(arr.reshape((len(arr), 1)))

  misc.stop('done one_hot_encoding column converted to ' + 
      `col_ohe.shape[1]` + ' columns')  
  return col_ohe

def _s_bin(self, n_bins=100):
  return pd.Series(pd.cut(self, n_bins), index=self.index)

def _s_group_rare(self, limit=30, rare_val=None):
  vcs = self.value_counts()
  rare = vcs[vcs <= limit].keys()  
  if rare_val is None:
    rare_val = 'rare' 
    if self.is_numerical(): rare_val = -1
    elif self.is_index(): 
      print 'self.max():', self.max(), self
      self.max() + 1

  self[self.isin(rare)] = rare_val
  return self

def _s_sigma_limits(self, sigma):
  '''
  returns the minimum and maximum values in the series between the
  specified sigma.  This can be used to truncate outliers.
  '''
  delta = float(sigma) * self.std()
  m = self.mean()
  return (m - delta, m + delta)

def _s_to_indexes(self):
  cat = pd.Categorical.from_array(self)
  lbls = cat.codes if hasattr(cat, 'codes') else cat.labels    
  return pd.Series(lbls, index=self.index, dtype=utils.get_optimal_numeric_type('int', 0, len(lbls) + 1))

def _s_append_bottom(self, s):  
  if type(s) is not pd.Series: s = pd.Series(s)
  return pd.concat([self, s], ignore_index=True)

def _s_missing(self, fill='none'):  
  misc.start('replacing series missing data fill[' + `fill` + ']')
  val = utils.get_col_aggregate(self, fill)    
  self.fillna(val, inplace=True)
  self.replace([np.inf, -np.inf], val, inplace=True)  

  misc.stop('replacing series missing data')
  return self

def _s_normalise(self):  
  return self.scale(min_max=(0, 1))

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

def _s_is_valid_name(self):
  if not self.name: return False
  for v in ['c_', 'n_', 'i_', 'd_', 'b_']:
    if self.name.startswith(v): return True
  return False

def _s_is_categorical(self):
  if self.is_valid_name(): return self.name.startswith('c_')
  return sklearn.utils.multiclass.type_of_target(self) == 'multiclass'

def _s_is_index(self):
  if self.is_valid_name(): return self.name.startswith('i_')
  return sklearn.utils.multiclass.type_of_target(self) == 'multiclass' and str(self.dtype).startswith('int')

def _s_is_binary(self):
  if self.is_valid_name(): return self.name.startswith('b_')
  uniques = self.unique()
  if len(uniques) == 1: 
    misc.dbg('\n!!! columns: ' + self.name + ' has only 1 unique value and hence has no information and should be removed\n')
  return len(uniques) == 2

def _s_is_categorical_like(self):
  return self.is_categorical() or self.is_index() or self.is_binary()

def _s_is_numerical(self):
  if self.is_valid_name(): return self.name.startswith('n_')
  return sklearn.utils.multiclass.type_of_target(self) == 'continuous'

def _s_is_date(self):
  if self.is_valid_name(): return self.name.startswith('d_')
  return str(self.dtype).startswith('date') or \
      type(self[0]) is pd.Timestamp or \
      type(self[0]) is datetime.datetime

def _s_is_equals(self, s):
  if type(s) is not pd.Series: s = pd.Series(s)
  return np.all(self == s);

def _s_all_close(self, s, tol=1e-08):
  return np.allclose(self, s, atol=tol)

def _s_infer_col_name(self):
  if self.is_valid_name(): return self
  name = self.name or 'unknown'
  dtype = str(self.dtype)
  if self.is_date(): prefix = 'd_'
  elif self.is_binary(): prefix = 'b_'
  elif self.is_numerical(): prefix = 'n_'  
  else: prefix = 'c_'
  self.name = prefix + name
  return self

def _s_outliers(self, stds=3):  
  mean, offset = self.mean(), stds * self.std()
  min, max = mean - offset, mean + offset
  return self.clip(min, max)

def _s_categorical_outliers(self, min_size=0.01, fill_mode='mode'):     
  threshold = float(len(self)) * min_size if type(min_size) is float else min_size 
  fill = utils.get_col_aggregate(self, fill_mode)
  vc = self.value_counts()
  under = vc[vc <= threshold]    
  if under.shape[0] > 0: 
    misc.debug('column [' + str(self.name) + '] threshold[' + `threshold` + '] fill[' + `fill` + '] num of rows[' + `len(under.index)` + ']')
    self[self.isin(under.index)] = fill
  return self

def _s_compress_size(self, aggresiveness=0, sparsify=False):      
  '''
  Always returns a new Series as inplace type change is not allowed
  '''
  c = self.copy()
  if c.is_numerical() or c.is_index():
    if c.is_index() or str(c.dtype).startswith('int'):        
      c = c.astype(utils.get_optimal_numeric_type('int', min(c), max(c)))
      return c if not sparsify else c.to_sparse(fill_value=int(c.mode()))    
    elif str(c.dtype).startswith('float'):
      return c.astype(utils.get_optimal_numeric_type(c.dtype, min(c), max(c), aggresiveness=aggresiveness))
    else:
      raise Exception(str(c.name) + ' expected "int" or "float" type got: ', str(c.dtype))
  else : 
    misc.dbg(c.name + ' is not supported, ignored during compression')
  return c

def _s_hashcode(self):
  index = tuple(self.index)
  values = tuple(self.values)
  item = tuple([index, values])
  return hash(item)

def _s_add_noise(self, level=0.40, mode='random'):
  misc.reseed(None)
  if mode == 'random':
    return self + self * (level * np.random.random(size=self.size) - (level/2.))
  if mode =='gaussian':
    return self + np.random.normal(scale=level, size=self.size)
  raise Exception('mode: ' + mode + ' is not supported.')

def _s_to_count_of_samples(self):
  if not self.is_categorical_like(): raise Exception('only supported for categorical like columns')
  vc = self.value_counts(dropna=False)
  for val in vc.keys(): self[self==val] = vc[val]
  return self

def _s_to_ratio_of_samples(self):
  if not self.is_categorical_like(): raise Exception('only supported for categorical like columns')
  vc = self.value_counts(dropna=False)
  for val in vc.keys():
    num = vc[val]
    self[self==val] = num / float(len(self))
  return self

def _s_to_count_of_binary_target(self, y, positive_class=None):
  if not self.is_categorical_like(): raise Exception('only supported for categorical like columns')
  if type(y) is not pd.Series: y = pd.Series(y)
  classes = y.unique()
  if len(classes) != 2: raise Exception('only binary target is supported')
  if positive_class is None: positive_class = np.max(classes)
  for val in self.unique():
    this_y = y[self == val]    
    self[self==val] = len(this_y[this_y == positive_class])
  return self

def _s_to_ratio_of_binary_target(self, y, positive_class=None):
  if not self.is_categorical_like(): raise Exception('only supported for categorical like columns')
  if type(y) is not pd.Series: y = pd.Series(y)
  classes = y.unique()
  if len(classes) != 2: raise Exception('only binary target is supported')
  if positive_class is None: positive_class = np.max(classes)
  for val in self.unique():
    this_y = y[self == val]    
    ratio = len(this_y[this_y == positive_class]) / float(len(this_y))
    self[self==val] = ratio
  return self

def _s_to_stat(self, y, stat='mean', 
      missing_value='missing', missing_treatment='missing-category', 
      noise_level=None):
  # if not self.is_categorical_like(): raise Exception('only supported for categorical like columns')
  if type(y) is not pd.Series: y = pd.Series(y)  
  train = self[:len(y)] 
  test = self[len(y):]
  df = pd.DataFrame({'c_1' : train, 'n_y': y.values})
  
  def iqm(x): return np.mean(np.percentile(x, [75 ,25]))

  s = df.groupby('c_1')['n_y'].\
      transform(iqm if stat == 'iqm' else stat)
  if len(test) > 0:   
    _, not_in_train = train.difference_with(test, quiet=True)  
    transformer = dict(zip(train, s))

    test[test.isin(not_in_train)] = missing_value if \
        missing_treatment == 'missing-category' and missing_value in transformer else 'use-whole-set'

    if (missing_treatment != 'missing-category' or missing_value not in transformer):
      transformer['use-whole-set'] = utils.get_col_aggregate(y, stat)
    s =  s.append_bottom(test.map(transformer))  
    
  if noise_level > 0: s = s.add_noise(noise_level, 'gaussian')
  return s

def _s_to_rank(self, normalise=True):
  r = self.rank()
  if normalise: r = r.normalise()
  return r

def _s_boxcox(self):
  minv = self.min()
  if minv <= 0: self = 1 + (self - minv)
  return scipy.stats.boxcox(self)[0]

def _s_vc(self):
  counts = self.value_counts(dropna=False)  
  misc.dbg(counts, '\n', len(counts), 'uniques')

def _s_floats_to_ints(self, decimals=5):
  if not str(self.dtype).startswith('float'): return self
  return (self * (10 ** decimals)).astype(np.int32)

def _s_percentage_positive(self, positive_val=True):
  return float(len(self[self==positive_val])) / len(self)

def _s_viz(self):
  from viz.describe_series import DescribeSeries
  return DescribeSeries(self)

def _s_difference_with(self, other, quiet=False):
  self_s = set(self.unique())
  other_s = set(other.unique())
  intersection = len(self_s.intersection(other_s))
  actual_diffs = self_s.difference(other_s)
  difference = len(actual_diffs)
  if not quiet: misc.dbg('same:', intersection, 
        'diff:', difference, 
        '%% diff: %.1f' % (100. * difference / (intersection + difference)))
  in_self = self_s - other_s
  in_other = other_s - self_s
  if not quiet and len(in_self) < 50: misc.dbg('in left: %s:' % in_self)
  if not quiet and len(in_other) < 50: misc.dbg('in right: %s:' % in_other)
  return (in_self, in_other)

'''
Add new methods manually using:
pandas_extensions._extend_s('group_rare', _s_group_rare)
'''    