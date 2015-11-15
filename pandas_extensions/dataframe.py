import pandas as pd, numpy as np
import itertools, random, gzip, gc, ast_parser, scipy, \
  sklearn, sklearn.manifold, sklearn.cluster, os
from .. import misc
from ..lib import smote
import utils

def _df_categoricals(self): return filter(lambda c: self[c].is_categorical(), self.columns)
def _df_indexes(self): return filter(lambda c: self[c].is_index(), self.columns)
def _df_binaries(self): return filter(lambda c: self[c].is_binary(), self.columns)
def _df_categorical_like(self): return self.categoricals() + self.indexes() + self.binaries()
def _df_numericals(self): return filter(lambda c: self[c].is_numerical(), self.columns)
def _df_dates(self): return filter(lambda c: self[c].is_date(), self.columns)

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

def _df_one_hot_encode(self, columns=None, dtype=np.float):  
  if self.categoricals(): 
    self.to_indexes(columns=columns)    
    if columns is not None: columns = ['i_' for c in columns]

  misc.start('one_hot_encoding data frame with ' + `self.shape[1]` + \
    ' columns. \n\tNOTE: this resturns a sparse array and empties' + \
    ' the initial array.')  

  misc.debug('separating categoricals from others')
  if columns is None: columns = self.indexes()  
  if len(columns) == 0: 
    misc.stop('no columns found to one hot encode, returning original data frame')
    return self
  others = filter(lambda c: not c in columns, self.columns)

  categorical_df = self[columns]    
  others_df = scipy.sparse.coo_matrix(self[others].values)

  # Destroy original as it now just takes up memory
  self.drop(self.columns, 1, inplace=True) 
  gc.collect()

  ohe_sparse = None
  for i, c in enumerate(columns):
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

def _df_to_indexes(self, columns=None, drop_origianls=True, sparsify=False):
  misc.start('indexing categoricals in data frame')  
  if columns is None or len(columns) == 0: columns = self.categoricals() + self.binaries()
  for c in columns: self['i_' + c] = self[c].astype('category').cat.codes
  if drop_origianls: self.drop(columns, 1, inplace=True)
  misc.stop('done indexing categoricals in data frame')  
  return self

def _df_cats_to_count_of_binary_target(self, y, columns=None, positive_class=None):
  if columns is None or len(columns) == 0: columns = self.categorical_like()
  for c in columns:
    self[c] = self[c].to_count_of_binary_target(y, positive_class)
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_ratio_of_binary_target(self, y, columns=None, positive_class=None):
  if columns is None or len(columns) == 0: columns = self.categorical_like()
  for c in columns:
    self[c] = self[c].to_ratio_of_binary_target(y, positive_class)
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_count_of_samples(self, columns=None):
  if columns is None or len(columns) == 0: columns = self.categorical_like()
  for c in columns:
    self[c] = self[c].to_count_of_samples()
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_ratio_of_samples(self, columns=None):
  if columns is None or len(columns) == 0: columns = self.categorical_like()
  for c in columns:
    self[c] = self[c].to_ratio_of_samples()
    self[c].name = c.replace('c_', 'n_')
  return self

def _df_cats_to_stat(self, y, stat='mean', 
    remove_originals=True, columns=None, 
    missing_value='missing', missing_treatment='missing-category', noise_level=None):
  '''
  stat: can be string 'mean', 'iqm', 'median', 'min', 'max' or
    'all' which creates a group of columns for each of the stats.
    stat can also be a dictionary of column names to their stat.
  '''
  misc.start('converting categoricals to stat: ' + str(stat))
  if type(stat) is dict:
    cols = stat.keys()
    for c in cols:
      s = stat[c]
      self['n_' + c + '_' + s] = self[c].to_stat(y, s, 
          missing_value=missing_value, missing_treatment=missing_treatment, noise_level=noise_level).astype(float)
  else:
    if stat == 'all': stat = ['mean', 'iqm', 'median', 'min', 'max']    
    if type(stat) is str: stat = [stat]
    cols = columns if columns is not None else self.categorical_like()
    for s in stat:
      for c in cols: 
        self['n_' + c + '_' + s] = self[c].to_stat(y, s, missing_value=missing_value, 
            missing_treatment=missing_treatment, noise_level=noise_level).astype(float)
  if remove_originals: self.remove(cols)
  misc.stop('done converting categoricals')
  return self

def _df_bin(self, n_bins=100, drop_origianls=False):
  misc.start('binning data into ' + `n_bins` + ' bins')  
  for n in self.numericals():
    self['c_binned_' + n] = self[n].bin(n_bins)
    if drop_origianls: self.drop(n, 1, inplace=True)
  misc.stop('done binning data into ' + `n_bins` + ' bins')  
  return self

def _df_group_rare(self, columns=None, limit=30, rare_val=None):
  misc.start('grouping rare categorical columns, limit: ' + `limit`)  
  if columns is None: columns = self.categorical_like()
  for c in columns: self[c].group_rare(limit, rare_val=rare_val)
  misc.stop('done grouping rare categorical')  
  return self

def _df_combinations(self, group_size=2, columns=[], categoricals=False, indexes=False,
    numericals=False, dates=False, binaries=False, permutations=False):
  cols = list(columns)
  misc.start('calculating combinations')
  if categoricals: cols = cols + self.categoricals()
  if indexes: cols = cols + self.indexes()
  if numericals: cols = cols + self.numericals()
  if dates: cols = cols + self.dates()
  if binaries: cols = cols + self.binaries()
  op = itertools.permutations if permutations else itertools.combinations
  misc.stop('done calculating combinations')
  return list(op(cols, group_size))

def _df_remove_nas(self, columns=None):      
  misc.start('removing missing values')
  self.dropna(0, 'any', subset=columns, inplace=True)
  misc.stop('done removing missing values')
  return self

def _df_remove(self, columns=[], categoricals=False, numericals=False, 
    dates=False, binaries=False, missing_threshold=0.0):    
  cols = [columns] if type(columns) is str else list(columns)
  misc.start('removing columns')
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

  cols = [c for c in cols if c in self.columns]
  self.drop(cols, 1, inplace=True)
  misc.stop('done removing [' + str(len(cols)) + '] columns')
  return self
 
def _df_scale(self, columns=None, min_max=None):  
  misc.start('scaling data frame')
  cols = columns if columns is not None else self.numericals()
  for c in cols: self[c] = self[c].scale(min_max)
  misc.stop('scaling data frame')
  return self

def _df_normalise(self, columns=None):
  return self.scale(columns, min_max=(0, 1))

def _df_missing(self, categorical_fill='none', numerical_fill='none'):  
  misc.start('replacing missing data categorical[' + `categorical_fill` + '] numerical[' + `numerical_fill` + ']')
  
  # Do numerical constants on whole DF for performance
  if type(numerical_fill) != str:
    self[self.numericals()] = self[self.numericals()].fillna(numerical_fill)
    self.replace([np.inf, -np.inf], numerical_fill, inplace=True)
    numerical_fill='none'

  # Do categorical constants on whole DF for performance
  if categorical_fill != 'none' and categorical_fill != 'mode':
    self[self.categorical_like()] = self[self.categorical_like()].fillna(categorical_fill)
    categorical_fill='none'

  # Get list of columns still left to fill
  categoricals_to_fill = []
  numericals_to_fill = []
  binaries_to_fill = []
  if categorical_fill != 'none': categoricals_to_fill += self.categorical_like()
  if numerical_fill != 'none': numericals_to_fill += self.numericals()

  # Prepare a dictionary of column -> fill values
  to_fill = {}  
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
  misc.start('creating new concatenated dataframe')
  X = pd.concat([self, df], ignore_index=True)
  misc.stop('done creating new concatenated dataframe')
  return X

def _df_subsample(self, y=None, size=0.5):  
  if y is not None: self = self[:len(y)]
  if type(size) is float:
    if size < 1.0: size = self.shape[0] * size
    size = int(size)
  if self.shape[0] <= size: return self if y is None else (self, y) # unchanged    

  misc.start('subsample data frame, size: ' + str(size))
  df = self.copy().shuffle(y)
  
  result = df[:size] if y is None else df[0][:size], df[1][:size]  
  misc.start('done, subsample data frame')
  return result

def _df_col_subsample_columns(self, fraction, random_state=None):
  if random_state is not None: misc.seed(random_state)
  else: misc.reseed(None)
  column_idxs  = range(self.shape[1])
  np.random.shuffle(column_idxs)
  n_cols = max(1, int(fraction * len(column_idxs)))
  return self.columns[column_idxs[:n_cols]]

def _df_col_subsample(self, fraction, random_state=None):
  return self[self.col_subsample_columns(fraction, random_state)]

def _df_shuffle(self, y=None):    
  misc.start('shuffling data frame')  
  df = self.copy()  
  if y is not None: 
    df = df[:y.shape[0]]
    df['__tmpy'] = y.values

  index = list(df.index)
  misc.reseed(None)
  random.shuffle(index)
  df = df.ix[index]
  df.reset_index(inplace=True, drop=True)

  result = df
  if y is not None:     
    y = pd.Series(df['__tmpy'].values)
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

def _df_split_train_test(self, y):  
  return (self[:len(y)], self[len(y):])

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

def _df_cv(self, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1, fit_params=None, prefix=None):  
  return _df_cv_impl_(self, clf, y, n_samples, n_iter, scoring, n_jobs, fit_params, prefix)

def _df_cv_ohe(self, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1, fit_params=None, prefix=None):  
  return _df_cv_impl_(self.one_hot_encode(), clf, y, n_samples, n_iter, scoring, n_jobs, fit_params, prefix)

def _df_cv_impl_(X, clf, y, n_samples=None, n_iter=3, scoring=None, n_jobs=-1, fit_params=None, prefix=None):    
  if hasattr(y, 'values'): y = y.values
  if n_samples is None: n_samples = len(y)
  else: n_samples = min(n_samples, len(y), X.shape[0])
  if len(y) < X.shape[0]: X = X[:len(y)]
  if sklearn.utils.multiclass.type_of_target(y) == 'binary' and not (scoring or misc.cfg['scoring']): 
    scoring = 'roc_auc'
  score_name = scoring or misc.cfg['scoring']
  if hasattr(score_name, '__name__'): score_name = score_name.__name__
  misc.start('starting ' + `n_iter` + ' fold cross validation (' + 
      `n_samples` + ' samples) w/ metric: ' + str(score_name))
  cv = misc.do_cv(clf, X, y, n_samples, n_iter=n_iter, scoring=scoring, 
    quiet=prefix is None, n_jobs=n_jobs, fit_params=fit_params, prefix=prefix)
  misc.stop('done cross validation:\n  [CV]: ' + ("{0:.5f} (+/-{1:.5f})").format(cv[0], cv[1]))  
  return cv

def _df_pca(self, n_components, whiten=False):  
  misc.start('pca')
  new_X = sklearn.decomposition.PCA(n_components, whiten=whiten).fit_transform(self)
  columns = map(lambda i: 'n_pca_' + `i`, range(n_components))
  misc.start('done pca')
  return pd.DataFrame(columns=columns, data=new_X)

def _df_tsne(self, n_components=2):  
  misc.start('tsne')
  # barnes_hut not in sklearn master yet. but put in once there
  # new_X = sklearn.manifold.TSNE(n_components, method='barnes_hut').fit_transform(self)
  new_X = sklearn.manifold.TSNE(n_components).fit_transform(self)
  columns = map(lambda i: 'n_tsne_' + `i`, range(n_components))
  misc.start('done tsne')
  return pd.DataFrame(columns=columns, data=new_X)

def _df_kmeans(self, k):  
  misc.start('kmeans, k: ' + str(k))  
  s = pd.Series(sklearn.cluster.KMeans(k).fit_predict(self))
  misc.stop('done kmeans')
  return s

def _df_tree_features(self, tree_ensemble, y):
  misc.start('tree_features using: ' + tree_ensemble.__class__.__name__)
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
  misc.stop('done tree_features')
  return tree_features

def _df_append_fit_transformer(self, fit_transformer, method='transform'):    
  misc.start('append_fit_transformer')
  if 'fit' not in method: fit_transformer.fit(self)
  new_X = getattr(fit_transformer, method)(self)
  if utils.is_sparse(new_X): new_X = new_X.todense()
  columns = map(lambda i: 'n_new_col_' + `i`, range(new_X.shape[1]))
  new_df = pd.DataFrame(new_X, columns=columns)
  X = self.copy().append_right(new_df)
  misc.start('done append_fit_transformer')
  return X

def _df_predict(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'predict')

def _df_predict_proba(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'predict_proba')

def _df_transform(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'transform')

def _df_decision_function(self, clf, y, X_test=None):    
  return __df_clf_method_impl(self, clf, y, X_test, 'decision_function')

def __df_clf_method_impl(X, clf, y, X_test, method):    
  misc.start('clf_method_impl: ' + method)
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
  val = getattr(clf, method)(X_test)
  misc.start('done clf_method_impl: ' + method)
  return val

def _df_self_predict(self, clf, y, cv=5): return misc.self_predict(clf, self, y, cv)

def _df_self_predict_proba(self, clf, y, cv=5): return misc.self_predict_proba(clf, self, y, cv, 'predict_proba')

def _df_self_transform(self, clf, y, cv=5): return misc.self_transform(clf, self, y, cv, 'transform')

def _df_self_chunked_op(self, y, op, cv=5): return misc.self_chunked_op(self, y, op, cv)

def _df_trim_on_y(self, y, min_y=None, max_y=None):    
  X = self.copy()  
  X['__tmpy'] = y.copy()
  if min_y is not None: X = X[X['__tmpy'] >= min_y]
  if max_y is not None: X = X[X['__tmpy'] <= max_y]
  y = X['__tmpy']
  return (X.remove('__tmpy'), y)

def _df_save_csv(self, file, header=True, force=False):   
  if os.path.isfile(file) and not force:
    raise Exception('File: ' + file + ' already exists.  To overwrite set force=True')
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
  self.toidxs(drop_origianls=True)
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

def _df_importances(self, clf, y):
  misc.start('calculating feature importances using: ' + clf.__class__.__name__)
  clf.fit(self[:len(y)], y)
  if hasattr(clf, 'feature_importances_'): imps = clf.feature_importances_ 
  else: imps = map(abs, clf.coef_[0])
  top_importances_indexes = np.argsort(imps)[::-1]
  top_importances_values = np.array(imps)[top_importances_indexes]
  top_importances_features = self.columns[top_importances_indexes]
  misc.stop('done calculating feature importances')
  return zip(top_importances_features, top_importances_values)

def _df_numerical_stats(self, columns=None):
  misc.start('adding row numerical stats')
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
  misc.stop('done row numerical stats')
  return self

def _df_smote(self, y, percentage_multiplier, n_neighbors, opt_target=None):
  misc.start('smoting by : ' + str(percentage_multiplier) + '%')
  if type(y) is not pd.Series: y = pd.Series(y)
  misc.reseed(None)
  vcs = y.value_counts(dropna=False)
  if len(vcs) != 2: raise Exception('DataFrame.smote only works on binary classifiers')
  min_value = opt_target if opt_target is not None else vcs.argmin()
  minorities = self[y == min_value]

  new_minorities = smote.SMOTE(minorities.values, percentage_multiplier, n_neighbors)
  new_len = self.shape[0] + new_minorities.shape[0]
  new_y_data = np.append(y.values, np.array([min_value] * len(new_minorities)))
  y2 = pd.Series(new_y_data, index=np.arange(new_len))
  minorities_df = pd.DataFrame(new_minorities, columns=self.columns)
  new_df = self.copy().append_bottom(minorities_df)
  new_df.index = np.arange(new_len)
  misc.stop('done smote')
  return (new_df, y2)

def _df_boxcox(self):
  misc.start('box-cox converting numerical columns')
  for n in self.numericals(): self[n] = self[n].boxcox()
  misc.stop('done box-cox conversion')
  return self

def _df_break_down_dates(self, aggresiveness=3, remove_originals=True):
  date_cols = self.dates()
  misc.start('breaking down date columns; aggresiveness: ' + str(aggresiveness) + ' columns: ' + str(date_cols))
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
  misc.stop('done breaking down date columns')
  return self

__df_to_dates_cache = {}
def _df_to_dates(self, columns, remove_originals=True, cache=True):  
  if type(columns) is str: columns = [columns]
  misc.start('converting to pd.datetime: ' + str(columns))
  for d in columns: 
    if cache:
      colid = str(d) + ':' + str(self[d].hashcode())
      if colid in __df_to_dates_cache: self['d_' + d] = __df_to_dates_cache[colid]
      else: self['d_' + d] = __df_to_dates_cache[colid] = pd.to_datetime(self[d])
    else: self['d_' + d] = pd.to_datetime(self[d])
  if remove_originals: self.remove(columns)
  misc.stop('done converting to pd.datetime')
  return self

def _df_floats_to_ints(self, decimals=5):
  misc.start('float_to_int decimals: ' + `decimals`)
  for c in self.numericals(): self[c] = self[c].floats_to_ints(decimals)
  misc.stop('float_to_int done')
  return self

def _df_viz(self):
  from .viz.describe_dataframe import DescribeDataFrame
  return DescribeDataFrame(self)

def _df_describe_similarity(self, other):
  single_val = []
  diffs = []
  for c in self.columns:    
    if self[c].is_categorical_like() or self[c].is_date():
      tr_d = set(self[c].unique())
      te_d = set(other[c].unique())
      intersection = len(tr_d.intersection(te_d))
      actual_diffs = tr_d.difference(te_d)
      difference = len(actual_diffs)
      if difference > 0:
        diffs.append({'c': c, 'i': intersection, 'ad': actual_diffs, 'd': difference, 'p': (100. * difference / (intersection + difference))})
      elif intersection == 1:
        single_val.append(c)
        
  if len(single_val) > 0:
    for c in single_val:
      misc.dbg('column:', c, 'has only one value? consider removing')  
    misc.dbg('\n\n')

  for r in sorted(diffs, key=lambda r: r['p'], reverse=True):
    misc.dbg(r['c'], 
        'same:', r['i'], 
        'diff:', r['d'], 
        '%% diff: %.1f' % (r['p']))
    if r['d'] <=3:
      misc.dbg('\tactual differences: %s:' % r['ad'])

def _df_impute_categorical(self, column, missing_value=np.nan):
  X = self.copy()
  to_impute = X[X[column]==missing_value].remove(column)
  to_impute_indexes = to_impute.index
  to_train = X[X[column]!=missing_value]
  y = to_train[column]
  to_train.remove(column)
  clf = sklearn.linear_model.LogisticRegression()
  imputed_values = clf.fit(to_train, y).predict(to_impute)
  self.ix[to_impute_indexes, column] = imputed_values
  return self

def _df_custom_cache(self, name, value=None):
  hashcode = str(self.hashcode())
  prop_name = hashcode + ':' + name
  if '__custom_cache' not in self.__dict__:
    self.__dict__['__custom_cache'] = {}
  if value is not None:     
    self.__custom_cache[prop_name] = value
  for k in self.__custom_cache.keys():
    this_hash = k.split(':')[0]
    if this_hash != hashcode: 
      del self.__custom_cache[k]
  return self.__custom_cache[prop_name] if prop_name in self.__custom_cache else None  

def _df_add_noise(self, columns=None, level=.4, mode='random'):
  if columns is None or len(columns) == 0: columns = self.numericals()
  if len(columns) == 0: misc.dbg('not adding noise: no numerical columns found')
  for n in columns: self[n] = self[n].add_noise(level, mode)
  return self

'''
Add new methods manually using:
pandas_extensions._extend_df('group_rare', _df_group_rare)
'''  
