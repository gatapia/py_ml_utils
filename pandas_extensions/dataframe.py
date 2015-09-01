import pandas as pd, numpy as np
import itertools, random, gzip, gc, ast_parser, scipy, \
  sklearn, sklearn.manifold, sklearn.cluster, smote
from .. import misc
import utils

def _df_categoricals(self): return filter(lambda c: c.startswith('c_'), self.columns)
def _df_indexes(self): return filter(lambda c: c.startswith('i_'), self.columns)
def _df_binaries(self): return filter(lambda c: c.startswith('b_'), self.columns)
def _df_categorical_like(self): return self.categoricals() + self.indexes() + self.binaries()
def _df_numericals(self): return filter(lambda c: c.startswith('n_'), self.columns)
def _df_dates(self): return filter(lambda c: c.startswith('d_'), self.columns)

def _df_is_equal(self, other):
  if type(other) is not pd.DataFrame: other = pd.DataFrame(other)
  return np.all(self == other);

def _df_all_close(self, other, tol=1e-08):
  if type(other) is not pd.DataFrame: other = pd.DataFrame(other)
  return np.allclose(self, other, atol=tol)

def _df_ensure_unique_names(self):
  uniques = []
  for i, n in enumerate(self.columns):
    unique_idx = 1
    tmp = n
    while tmp in uniques:
      tmp = n + '_' + `unique_idx`
      unique_idx += 1
    uniques.append(tmp)
  self.columns = uniques
  return self

def _df_infer_col_names(self):
  misc.start('infering column names')  
  self.columns = [self.ix[:,i].infer_col_name().name for i in range(self.shape[1])]
  return self.ensure_unique_names()

def _df_one_hot_encode(self, dtype=np.float):  
  if self.categoricals(): self.to_indexes(drop_origianls=True)    

  misc.start('one_hot_encoding data frame with ' + `self.shape[1]` + \
    ' columns. \n\tNOTE: this resturns a sparse array and empties' + \
    ' the initial array.')  

  misc.debug('separating categoricals from others')
  indexes = self.indexes()
  if not indexes: return self
  others = filter(lambda c: not c in indexes, self.columns)

  categorical_df = self[indexes]    
  others_df = scipy.sparse.coo_matrix(self[others].values)

  # Destroy original as it now just takes up memory
  self.drop(self.columns, 1, inplace=True) 
  gc.collect()

  ohe_sparse = None
  for i, c in enumerate(indexes):
    misc.debug('one hot encoding column: ' + `c`)
    col_ohe = sklearn.preprocessing.OneHotEncoder(categorical_features=[0], dtype=dtype).\
      fit_transform(categorical_df[[c]])
    if ohe_sparse == None: ohe_sparse = col_ohe
    else: ohe_sparse = scipy.sparse.hstack((ohe_sparse, col_ohe))
    categorical_df.drop(c, axis=1, inplace=True)
    gc.collect()
  
  matrix = ohe_sparse if len(others) == 0 else scipy.sparse.hstack((ohe_sparse, others_df))
  misc.stop('done one_hot_encoding')
  return matrix.tocsr()

def _df_to_indexes(self, drop_origianls=False, sparsify=False):
  misc.start('indexing categoricals in data frame. Note: NA gets turned into max index (255, 65535, etc)')  
  for c in self.categoricals() + self.binaries():
    col = 'i_' + c
    cat = pd.Categorical.from_array(self[c])
    lbls = cat.codes if hasattr(cat, 'codes') else cat.labels    
    s = pd.Series(lbls, index=self[c].index, \
      dtype=utils.get_optimal_numeric_type('int', 0, len(lbls) + 1))
    modes = s.mode()
    mode = lbls[0]
    if len(modes) > 0: mode = modes.iget(0)
    self[col] = s.to_sparse(fill_value=int(mode)) if sparsify else s
    if drop_origianls: self.drop(c, 1, inplace=True)
  misc.stop('done indexing categoricals in data frame')  
  return self

def _df_cats_to_count_of_binary_target(self, y, positive_class=None):
  for c in self.categorical_like():
    self[c] = self[c].to_count_of_binary_target(y, positive_class)
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_ratio_of_binary_target(self, y, positive_class=None):
  for c in self.categorical_like():
    self[c] = self[c].to_ratio_of_binary_target(y, positive_class)
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_count_of_samples(self):
  for c in self.categorical_like():
    self[c] = self[c].to_count_of_samples()
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_ratio_of_samples(self):
  for c in self.categorical_like():
    self[c] = self[c].to_ratio_of_samples()
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_stat(self, y, stat='mean', remove_originals=True):
  if stat == 'all': stat = ['mean', 'median', 'min', 'max']
  if type(stat) is str: stat = [stat]
  cols = self.categorical_like()
  for s in stat:
    for c in cols: 
      self['n_' + c + '_' + s] = self[c].to_stat(y, s).astype(float)
  if remove_originals: self.remove(cols)
  return self

def _df_bin(self, n_bins=100, drop_origianls=False):
  misc.start('binning data into ' + `n_bins` + ' bins')  
  for n in self.numericals():
    self['c_binned_' + n] = self[n].bin(n_bins)
    if drop_origianls: self.drop(n, 1, inplace=True)
  misc.stop('done binning data into ' + `n_bins` + ' bins')  
  return self

def _df_group_rare(self, columns=None, limit=30):
  misc.start('grouping rare categorical columns, limit: ' + `limit`)  
  if columns is None: columns = self.categorical_like()
  for c in columns: self[c].group_rare(limit)
  misc.start('done grouping rare categorical')  
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

  misc.debug('removing ' + `len(cols)` + ' columns from data frame')
  cols = [c for c in cols if c in self.columns]
  self.drop(cols, 1, inplace=True)
  return self
 
def _df_scale(self, columns=None, min_max=None):  
  misc.start('scaling data frame')
  cols = columns if columns is not None else self.numericals()
  for c in cols: self[c] = self[c].scale(min_max)
  misc.stop('scaling data frame')
  return self

def _df_normalise(self, columns=None):
  return self.scale(columns, min_max=(0, 1))

def _df_missing(self, categorical_fill='none', numerical_fill='none', binary_fill='none'):  
  misc.start('replacing missing data categorical[' + `categorical_fill` + '] numerical[' + `numerical_fill` + ']')
  
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
  for c in binaries_to_fill: to_fill[c] = utils.get_col_aggregate(self[c], binary_fill)
  for c in categoricals_to_fill: to_fill[c] = utils.get_col_aggregate(self[c], categorical_fill)
  for c in numericals_to_fill: 
    to_fill[c] = utils.get_col_aggregate(self[c], numerical_fill)
    self[c].replace([np.inf, -np.inf], to_fill[c], inplace=True)
  
  # Do fill in one step for performance
  if to_fill: self.fillna(value=to_fill, inplace=True)

  misc.stop('done replacing missing data')
  return self

def _df_outliers(self, stds=3):  
  misc.start('restraining outliers, standard deviations: ' + `stds`)
  for n in self.numericals(): self[n] = self[n].outliers(stds)
  misc.stop('done restraining outliers')
  return self

def _df_categorical_outliers(self, min_size=0.01, fill_mode='mode'):      
  misc.start('grouping categorical outliers, min_size: ' + `min_size`)
  for c in self.categorical_like(): self[c].categorical_outliers(min_size, fill_mode)
  misc.stop('done grouping categorical outliers')
  return self

def _df_append_right(self, df_or_s):  
  misc.start('appending to the right.  note, this is a destructive operation')
  if (type(df_or_s) is scipy.sparse.coo.coo_matrix):
    self_sparse = None
    for c in self.columns:
      misc.debug('\tappending column: ' + c)
      c_coo = scipy.sparse.coo_matrix(self[[c]])
      self.drop([c], 1, inplace=True)
      gc.collect()
      if self_sparse == None: self_sparse = c_coo
      else: self_sparse = scipy.sparse.hstack((self_sparse, c_coo)) 
    self_sparse = scipy.sparse.hstack((self_sparse, df_or_s))
    misc.stop('done appending to the right')
    return self_sparse
  elif utils.is_sparse(df_or_s) and not utils.is_sparse(self):
    misc.debug('converting data frame to a sparse frame')
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
  misc.stop('done appending to the right')
  return self

def _df_append_bottom(self, df):  
  return pd.concat([self, df], ignore_index=True)

def _create_s_from_templage(template, data):
  s = pd.Series(data)
  if template.dtype != s.dtype: s = s.astype(template.dtype)
  return s

def _df_subsample(self, y=None, size=0.5):  
  if type(size) is float:
    if size < 1.0: size = df.shape[0] * size
    size = int(size)
  if self.shape[0] <= size: return self if y is None else (self, y) # unchanged    

  misc.start('subsample data frame')
  df = self.copy().shuffle(y)
  
  result = df[:size] if y is None else df[0][:size], df[1][:size]  
  misc.start('done, subsample data frame')
  return result

def _df_shuffle(self, y=None):    
  misc.start('shuffling data frame')  
  df = self.copy()  
  if y is not None: 
    df = df[:y.shape[0]]
    df['__tmpy'] = y    

  index = list(df.index)
  misc.reseed(None)
  random.shuffle(index)
  df = df.ix[index]
  df.reset_index(inplace=True, drop=True)

  result = df
  if y is not None:     
    y = pd.Series(df['__tmpy'], index=df.index)
    df.remove(['__tmpy'])
    result = (df, y)

  misc.start('done, shuffling data frame')
  return result

def _df_noise_filter(self, type, *args, **kargs):  
  import scipy.ndimage.filters as filters

  misc.start('noise filtering data frame, filter type: ' + type)  
  filter = filters.gaussian_filter1d if type == 'gaussian' \
    else filters.maximum_filter1d if type == 'maximum' \
    else filters.minimum_filter1d if type == 'minimum' \
    else filters.uniform_filter1d if type == 'uniform' \
    else None
  if filter is None: raise Exception('filter: ' + type + ' is not supported')

  filtered = filter(self.values, *args, **kargs)
  df = utils.create_df_from_templage(self, filtered, self.index)
  misc.stop('noise filtering done')
  return df

def _df_split(self, y, stratified=False, train_fraction=0.5):  
  if type(y) is not pd.Series: y = pd.Series(y)
  train_size = int(self.shape[0] * train_fraction)
  test_size = int(self.shape[0] * (1.0-train_fraction))  
  misc.start('splitting train_size: ' + `train_size` + ' test_size: ' + `test_size`)
  if stratified:
    train_indexes, test_indexes = list(sklearn.cross_validation.StratifiedShuffleSplit(y, 1, test_size, train_size, random_state=misc.cfg['sys_seed']))[0]  
  else:
    train_indexes, test_indexes = list(sklearn.cross_validation.ShuffleSplit(len(y), 1, test_size, train_size, random_state=misc.cfg['sys_seed']))[0]  
  new_set = (
    self.iloc[train_indexes], 
    y.iloc[train_indexes], 
    self.iloc[test_indexes], 
    y.iloc[test_indexes]
  )
  misc.stop('splitting done')
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
  if sklearn.utils.multiclass.type_of_target(y) == 'binary' and not (scoring or misc.cfg['scoring']): 
    scoring = 'roc_auc'
  misc.start('starting ' + `n_iter` + ' fold cross validation (' + 
      `n_samples` + ' samples) w/ metric: ' + `scoring or misc.cfg['scoring']`)
  cv = misc.do_cv(clf, X, y, n_samples, n_iter=n_iter, scoring=scoring, quiet=True, n_jobs=n_jobs, fit_params=fit_params)
  misc.stop('done cross validation:\n  [CV]: ' + ("{0:.5f} (+/-{1:.5f})").format(cv[0], cv[1]))  
  return cv

def _df_pca(self, n_components, whiten=False):  
  new_X = sklearn.decomposition.PCA(n_components, whiten=whiten).fit_transform(self)
  columns = map(lambda i: 'n_pca_' + `i`, range(n_components))
  return pd.DataFrame(columns=columns, data=new_X)

def _df_tsne(self, n_components=2):  
  # barnes_hut not in sklearn master yet. but put in once there
  # new_X = sklearn.manifold.TSNE(n_components, method='barnes_hut').fit_transform(self)
  new_X = sklearn.manifold.TSNE(n_components).fit_transform(self)
  columns = map(lambda i: 'n_tsne_' + `i`, range(n_components))
  return pd.DataFrame(columns=columns, data=new_X)

def _df_kmeans(self, k):  
  return pd.Series(sklearn.cluster.KMeans(k).fit_predict(self))

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

def _df_append_fit_transformer(self, fit_transformer, method='transform'):    
  if 'fit' not in method: fit_transformer.fit(self)
  new_X = getattr(fit_transformer, method)(self)
  if utils.is_sparse(new_X): new_X = new_X.todense()
  columns = map(lambda i: 'n_new_col_' + `i`, range(new_X.shape[1]))
  new_df = pd.DataFrame(new_X, columns=columns)
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
  misc.reseed(clf)
  X_train = X
  if X_test is None:
    if X.shape[0] > len(y):
      X_test = X[len(y):]
      X_train = X[:len(y)]
    else: raise Exception('No X_test provided and X does not appear to have test set appended.  Did you mean to call self_' + method + ' instead?')

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
  if type(y) is not pd.Series: y = pd.Series(y)
  if y is not None and X.shape[0] != len(y): X = X[:len(y)]
  misc.start('self_' + method +' with ' + `cv` + ' chunks starting')
  misc.reseed(clf)
      
  def op(X, y, X2):
    if len(X.shape) == 2 and X.shape[1] == 1: 
      if hasattr(X, 'values'): X = X.values
      X = X.T[0]
    if len(X2.shape) == 2 and X2.shape[1] == 1: 
      if hasattr(X2, 'values'): X2 = X2.values
      X2 = X2.T[0]
    
    this_clf = sklearn.base.clone(clf)
    this_clf.fit(X, y)  
    new_predictions = getattr(this_clf, method)(X2)
    if new_predictions.shape[0] == 1:      
      new_predictions = new_predictions.reshape(-1, 1)
    return new_predictions    
  
  predictions = _df_self_chunked_op(X, y, op, cv)
  misc.stop('self_predict completed')
  return predictions.values

def _df_self_chunked_op(self, y, op, cv=5):    
  if y is not None and hasattr(y, 'values'): y = y.values
  X = self
  if cv is None: cv = 5
  if type(cv) is int: cv = sklearn.cross_validation.StratifiedKFold(y, cv, shuffle=True, random_state=misc.cfg['sys_seed'])
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

def _df_trim_on_y(self, y, sigma, max_y=None):    
  X = self.copy()  
  X['__tmpy'] = y.copy()
  X = X[np.abs(X['__tmpy'] - X['__tmpy'].mean()) <= 
      (float(sigma) * X['__tmpy'].std())]
  if max_y is not None: X = X[X['__tmpy'] <= max_y]
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

def _df_compress_size(self, aggresiveness=0, sparsify=False):  
  misc.start('compressing dataset with ' + `len(self.columns)` + ' columns')    

  def _format_bytes(num):
      for x in ['bytes','KB','MB','GB']:
          if num < 1024.0 and num > -1024.0:
              return "%3.1f%s" % (num, x)
          num /= 1024.0
      return "%3.1f%s" % (num, 'TB')

  original_bytes = self.nbytes()
  self.missing(categorical_fill='missing', numerical_fill='none')
  self.toidxs(True)
  for idx, c in enumerate(self.columns): 
    self[c] = self[c].compress_size(aggresiveness, sparsify)

  new_bytes = self.nbytes()
  diff_bytes = original_bytes - new_bytes
  misc.stop('original: %s new: %s improvement: %s percentage: %.2f%%' % 
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

def _df_summarise(self, opt_y=None, filename='dataset_description', 
    columns=None, start_notebook=True):
  from describe import Describe
  Describe(filename).show_dataset(self, opt_y, start_notebook=start_notebook)  

def _df_importances(self, clf, y):
  clf.fit(self[:len(y)], y)
  if hasattr(clf, 'feature_importances_'): imps = clf.feature_importances_ 
  else: imps = map(abs, clf.coef_[0])
  top_importances_indexes = np.argsort(imps)[::-1]
  top_importances_values = np.array(imps)[top_importances_indexes]
  top_importances_features = self.columns[top_importances_indexes]
  return zip(top_importances_features, top_importances_values)

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
  if type(y) is not pd.Series: y = pd.Series(y)
  misc.reseed(None)
  vcs = y.value_counts(dropna=False)
  if len(vcs) != 2: raise Exception('DataFrame.smote only works on binary classifiers')
  min_value = opt_target if opt_target is not None else vcs.argmin()
  minorities = self[y == min_value]

  new_minorities = smote.SMOTE(minorities.values, percentage_multiplier, n_neighbors)
  new_len = self.shape[0] + new_minorities.shape[0]
  y2 = pd.Series(np.append(y.values, np.array([min_value] * len(new_minorities))), index=np.arange(new_len))
  minorities_df = pd.DataFrame(new_minorities, columns=self.columns)
  new_df = self.copy().append_bottom(minorities_df)
  new_df.index = np.arange(new_len)
  return (new_df, y2)

def _df_boxcox(self):
  for n in self.numericals(): self[n] = self[n].boxcox()
  return self

def _df_break_down_dates(self, aggresiveness=3, remove_originals=True):
  date_cols = self.dates()
  for d in date_cols:
    s = self[d]
    self['c_' + d + '_year'] = s.dt.year
    self['c_' + d + '_month'] = s.dt.month
    if aggresiveness >= 1:
      self['c_' + d + '_dayofweek'] = s.dt.dayofweek
      self['c_' + d + '_quarter'] = s.dt.quarter
    if aggresiveness >= 2:
      self['c_' + d + '_year_and_month'] = self['c_' + d + '_year'] * 100
      self['c_' + d + '_year_and_month'] += self['c_' + d + '_month']
      self['c_' + d + '_weekday'] = s.dt.weekday
      self['c_' + d + '_weekofyear'] = s.dt.weekofyear
  if remove_originals: self.remove(date_cols)
  return self

def _df_to_dates(self, columns, remove_originals=True):
  if type(columns) is str: columns = [columns]
  for d in columns: self['d_' + d] = pd.to_datetime(self[d])
  if remove_originals: self.remove(columns)
  return self

'''
Add new methods manually using:
pandas_extensions._extend_df('to_dates', _df_to_dates)
'''  