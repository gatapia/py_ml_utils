import sys
sys.path.append('utils/lib')
import numpy as np
import pandas as pd
import scipy as scipy
import cPickle as pickle
from collections import Counter
import gzip, time, math, datetime, random, os, gc, logging
from sklearn import preprocessing, grid_search, utils, metrics, cross_validation
from scipy.stats import sem 
from scipy.stats.mstats import mode
from sklearn.externals import joblib

cfg = {
  'sys_seed':0,
  'debug':True,
  'scoring': None,
  'indent': 0
}

random.seed(cfg['sys_seed'])
np.random.seed(cfg['sys_seed']) 
NA = 99999.0
logging.basicConfig(level=logging.DEBUG, 
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
t0 = time.time()

def debug(msg): 
  if not cfg['debug']: return
  log.info(msg)

def start(msg): 
  if not cfg['debug']: return
  global t0
  t0 = time.time()
  log.info(msg)

def stop(msg): 
  if not cfg['debug']: return
  global t0
  log.info(msg + (', took (h:m:s): %s' % 
    datetime.timedelta(seconds=time.time() - t0)))
  t0 = time.time()

def reseed(clf):
  clf.random_state = cfg['sys_seed']
  random.seed(cfg['sys_seed'])
  np.random.seed(cfg['sys_seed']) 
  return clf

def model_name(clf):
  name = type(clf).__name__
  if hasattr(clf, 'base_classifier'): 
    name += '[' + model_name(clf.base_classifier) + ']'
  return name

def get_col_aggregate(col, mode):
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
  raise Exception('Unsupported aggregate mode: ' + `mode`)

def mean_score(scores):
  return ("{0:.5f} (+/-{1:.5f})").format(np.mean(scores), sem(scores))

def scale(X, min_max=None):  
  pp = preprocessing
  scaler = pp.MinMaxScaler(min_max) if min_max else pp.StandardScaler()
  return scaler.fit_transform(X)

def fillnas(X, categoricals=[], categorical_fill='mode', numerical_fill='mean', inplace=False):
  if not inplace: X = X.copy()
  for c in X.columns: 
    fill_mode = categorical_fill if c in categoricals else numerical_fill
    if fill_mode != 'none':
      X[c] = X[c].fillna(get_col_aggregate(X[c], fill_mode))
  return X

def one_hot_encode(X, columns, drop_originals=True):
  if type(columns[0]) is int: columns = map(lambda c: X.columns[c], columns)
  X = to_index(X.copy(), columns, drop_originals=True)
  new_cols = map(lambda c: c + '_indexes', columns)
  column_indexes = map(list(X.columns.values).index, new_cols)
  X_categoricals = X[column_indexes]
  X_enc = preprocessing.OneHotEncoder(sparse=False).fit_transform(X_categoricals)
  X_all = np.append(X.values, X_enc, 1)
  return np.delete(X_all, column_indexes, 1) if drop_originals else X_all

# Does a search through n_samples_arr to test what n_samples is acceptable
#   for cross validation.  No use using very high n_samples if not required
def do_n_sample_search(clf, X, y, n_samples_arr):
  reseed(clf)

  scores = []
  sems = []
  for n_samples in n_samples_arr:
    cv = do_cv(clf, X, y, n_samples, quiet=True)
    dbg("n_samples:", n_samples, "cv:", cv)
    scores.append(cv[0])
    sems.append(cv[1])
  max_score_idx = scores.index(max(scores))
  min_sem_idx = sems.index(min(sems))
  dbg("best score n_samples:", n_samples_arr[max_score_idx], "score:", scores[max_score_idx])
  dbg("best sem n_samples:", n_samples_arr[min_sem_idx], "sem:", sems[min_sem_idx])
  return (scores, sems)


def do_cv(clf, X, y, n_samples=None, n_iter=3, test_size=0.1, quiet=False, scoring=None, stratified=False, fit_params=None, reseed_classifier=True, n_jobs=-1):
  t0 = time.time()
  if reseed_classifier: reseed(clf)
  
  if n_samples is None: n_samples = len(y)
  elif type(n_samples) is float: n_samples = int(n_samples)
  
  try:
    if (n_samples > X.shape[0]): n_samples = X.shape[0]
  except: pass
  cv = cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=cfg['sys_seed']) \
    if not(stratified) else cross_validation.StratifiedShuffleSplit(y, n_iter, train_size=n_samples, test_size=test_size, random_state=cfg['sys_seed'])

  if n_jobs == -1 and cfg['cv_n_jobs'] > 0: n_jobs = cfg['cv_n_jobs']

  test_scores = cross_validation.cross_val_score(
      clf, X, y, cv=cv, scoring=scoring or cfg['scoring'], 
      fit_params=fit_params, n_jobs=n_jobs)
  if not(quiet): 
    dbg('%s took: %.2fm' % (mean_score(test_scores), (time.time() - t0)/60))
  return (np.mean(test_scores), sem(test_scores))

def split(X, y, test_split=0.1):
  X, y = utils.shuffle(X, y, random_state=cfg['sys_seed'])  
  num_split = math.floor(X.shape[0] * test_split) if type(test_split) is float else test_split
  test_X, test_y = X[:num_split], y[:num_split]
  X, y = X[num_split:], y[num_split:]
  return X, y, test_X, test_y

def proba_scores(y_true, y_preds, scoring=metrics.roc_auc_score):
  for i, y_pred in enumerate(y_preds):
    print 'classifier [%d]: %.4f' % (i+1, scoring(y_true, y_pred))

  print 'mean: %.4f' % (scoring(y_true, np.mean(y_preds, axis=0)))
  print 'max: %.4f' % (scoring(y_true, np.max(y_preds, axis=0)))
  print 'min: %.4f' % (scoring(y_true, np.min(y_preds, axis=0)))
  print 'median: %.4f' % (scoring(y_true, np.median(y_preds, axis=0)))

def score(clf, X, y, test_split=0.1, auc=False):
  X, y, test_X, test_y = split(X, y, test_split)
  reseed(clf)
  clf.fit(X, y)
  predictions = clf.predict_proba(test_X).T[1] if auc else clf.predict(test_X)
  return show_score(test_y, predictions)

def _to_np_arr(arrays):
  return map(lambda a: a.values if hasattr(a, 'values') else a, arrays)

def show_score(y_true, y_pred):  
  y_true, y_pred = _to_np_arr((y_true, y_pred))
  if (utils.multiclass.type_of_target(y_true) == 'binary' and
      utils.multiclass.type_of_target(y_pred) == 'continuous'):
    auc = metrics.roc_auc_score(y_true, y_pred)
    dbg('auc: ', auc)
    return auc

  if (utils.multiclass.type_of_target(y_true) == 'continuous' and
      utils.multiclass.type_of_target(y_pred) == 'continuous'):
    r2 = metrics.r2_score(y_true, y_pred)
    dbg('r2: ', r2)
    return r2

  accuracy = metrics.accuracy_score(y_true, y_pred)
  matrix = metrics.confusion_matrix(y_true, y_pred)
  report = metrics.classification_report(y_true, y_pred)
  dbg('Accuracy: ', accuracy, '\n\nMatrix:\n', matrix, '\n\nReport\n', report)
  return accuracy

def do_gs(clf, X, y, params, n_samples=1000, n_iter=3, 
    n_jobs=-2, scoring=None, fit_params=None, 
    random_iterations=None):
  start('starting grid search')
  if type(n_samples) is float: n_samples = int(n_samples)
  reseed(clf)
  cv = cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, random_state=cfg['sys_seed'])
  if random_iterations is None:
    gs = grid_search.GridSearchCV(clf, params, cv=cv, 
      n_jobs=n_jobs, verbose=2, scoring=scoring or cfg['scoring'], fit_params=fit_params)
  else:
    gs = grid_search.RandomizedSearchCV(clf, params, random_iterations, cv=cv, 
      n_jobs=n_jobs, verbose=2, scoring=scoring or cfg['scoring'], 
      fit_params=fit_params, refit=False)
  X2, y2 = utils.shuffle(X, y, random_state=cfg['sys_seed'])  
  gs.fit(X2[:n_samples], y2[:n_samples])
  stop('done grid search')
  dbg(gs.best_params_, gs.best_score_)  
  return gs

def dump(file, data):  
  if not os.path.isdir('data/pickles'): os.makedirs('data/pickles')
  if not '.' in file: file += '.pickle'
  joblib.dump(data, 'data/pickles/' + file);  

def load(file, opt_fallback=None):
  full_file = 'data/pickles/' + file
  if not '.' in full_file: full_file += '.pickle'
  if os.path.isfile(full_file): return joblib.load(full_file);
  if opt_fallback is None: return None
  data = opt_fallback()
  dump(file, data)
  return data
  
def get_write_file_stream(file):
  return gzip.GzipFile(file, 'wb') if file.endswith('.gz') else open(file, "wb")

def save_data(file, data):
  if (file.endswith('.gz')):
    f = gzip.GzipFile(file, 'wb')
    f.write(pickle.dumps(data, 0))
    f.close()
  else:
    f = open(file, "wb" )
    pickle.dump(data, f)
    f.close()

def read_data(file):
  if (file.endswith('z')):
    f = gzip.GzipFile(file, 'rb')
    buffer = ""
    while True:
      data = f.read()
      if data == "": break
      buffer += data
    object = pickle.loads(buffer)
    f.close()
    return object
  else:
    f = open(file, "rb" )
    data = pickle.load(f)
    f.close()
    return data

def read_df(file, nrows=None):
  if file.endswith('.pickle'): return load(file)
  
  compression = 'gzip' if file.endswith('.gz') else None
  nrows = None if nrows == None else int(nrows)
  return pd.read_csv(file, compression=compression, nrows=nrows);

def read_lines(file, ignore_header=False):
  with open(file) as f:
    if ignore_header: f.readline()
    return f.readlines()

def to_csv_gz(data_dict, file, columns=None):
  if file.endswith('.gz'): file = gzip.open(file, "wb")
  df = data_dict
  if type(df) is not pd.DataFrame: df = pd.DataFrame(df)
  df.to_csv(file, index=False, columns=columns)  

def gzip_file(in_name, out_name):  
  f_in = open(in_name, 'rb')
  f_out = gzip.open(out_name, 'wb')
  f_out.writelines(f_in)
  f_out.close()
  f_in.close()
  os.remove(in_name) 

def to_index(df_or_series, columns=[], drop_originals=False, inplace=False):
  if type(df_or_series) is pd.Series:
    labels = pd.Categorical.from_array(df_or_series).codes
    return pd.Series(labels)

  if not inplace: df_or_series = df_or_series.copy()

  for col in columns:
    if type(col) is int: col = df_or_series.columns[col]
    if not col in df_or_series.columns: continue
    
    df_or_series[col + '_indexes'] = to_index(df_or_series[col])
    if drop_originals: 
      df_or_series.drop(col, 1, inplace=True)
      gc.collect()
  return df_or_series

def dbg(*args):
  if cfg['debug']: print args