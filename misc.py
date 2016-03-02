from __future__ import print_function
import sys, gzip, time, datetime, random, os, logging, gc, \
    scipy, sklearn, sklearn.cross_validation, sklearn.grid_search,\
    sklearn.utils, sklearn.externals.joblib, inspect
import numpy as np, pandas as pd, xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from pandas import Series, DataFrame

def debug(msg): 
  if not cfg['debug']: return
  log.info(msg)

_message_timers = {}
def start(msg, id=None): 
  if not cfg['debug']: return
  if id is None:
    s = inspect.stack()
    if len(s) > 0 and len(s[1]) > 2: id = s[1][3]
    else: id = 'global'
  _message_timers[id] = time.time()
  log.info(msg)

def stop(msg, id=None): 
  if not cfg['debug']: return
  if id is None:
    s = inspect.stack()
    if len(s) > 0 and len(s[1]) > 2: id = s[1][3]
    else: id = 'global'
  took = datetime.timedelta(seconds=time.time() - _message_timers[id]) \
    if id in _message_timers else 'unknown'
  msg += (', took: %s' % str(took))
  log.info(msg)
  if id in _message_timers: del _message_timers[id]
  return msg

def reseed(clf):
  if clf is not None: clf.random_state = cfg['sys_seed']
  random.seed(cfg['sys_seed'])
  np.random.seed(cfg['sys_seed']) 
  return clf

def seed(seed):
  cfg['sys_seed'] = seed
  reseed(None)

def do_cv(clf, X, y, n_samples=None, n_iter=3, test_size=None, quiet=False, 
      scoring=None, stratified=False, n_jobs=-1, fit_params=None, prefix='CV'):
  if not quiet: start('starting ' + prefix)
  reseed(clf)
  
  if n_samples is None: n_samples = len(y)
  if X.shape[0] > len(y): X = X[:len(y)]
  elif type(n_samples) is float: n_samples = int(n_samples)
  if scoring is None: scoring = cfg['scoring']
  if test_size is None: test_size = 1./n_iter
  
  try:
    if (n_samples > X.shape[0]): n_samples = X.shape[0]
  except: pass

  if cfg['custom_cv'] is not None:
    cv = cfg['custom_cv']
  elif stratified:
    cv = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter, train_size=n_samples, test_size=test_size, random_state=cfg['sys_seed'])
  else:
    cv = sklearn.cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=cfg['sys_seed'])

  if n_jobs == -1 and cfg['cv_n_jobs'] > 0: n_jobs = cfg['cv_n_jobs']

  test_scores = sklearn.cross_validation.cross_val_score(
      clf, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, 
      fit_params=fit_params)
  score_desc = ("{0:.5f} (+/-{1:.5f})").format(np.mean(test_scores), scipy.stats.sem(test_scores))
  if not quiet: stop('done %s: %s' % (prefix, score_desc))
  return (np.mean(test_scores), scipy.stats.sem(test_scores))

def score_classifier_vals(prop, vals, clf, X, y, n_iter=3):
  results = []
  for v in vals:  
    clf = sklearn.base.clone(clf)
    target_clf = clf.base_classifier if hasattr(clf, 'base_classifier') else clf
    setattr(target_clf, prop, v)    
    score = do_cv(clf, X, y, n_iter=n_iter, prefix='CV - prop[%s] val[%s]' % (prop, str(v)))
    results.append({'prop': prop, 'v':v, 'score': score})  
  sorted_results = sorted(results, key=lambda r: r['score'][0], reverse=True)
  best = {'prop': prop, 'value': sorted_results[0]['v'], 'score': sorted_results[0]['score']}
  dbg('\n\n\n\n', best)
  return sorted_results

def score_operations_on_cols(clf, X, y, columns, operations, operator, n_iter=5):
  best = X.cv(clf, y, n_iter=n_iter)
  if not cfg['scoring_higher_better']: best *= -1
  results = []
  for c in columns:
    if c not in X: continue
    col_best = best
    col_best_op = 'no-op'
    for op in operations:
      X2 = operator(X.copy(), c, op)      
      score = X2.cv(clf, y, n_iter=n_iter)
      if not cfg['scoring_higher_better']: score *= -1
      if score[0] < col_best[0]:
        col_best = score
        col_best_op = str(op)
    r = {'column': c, 'best': col_best_op, 'score': col_best[0], 'improvement': best[0] - col_best[0]}
    results.append(r)
    dbg(r)
  return results

def do_gs(clf, X, y, params, n_samples=1.0, n_iter=3, 
    n_jobs=-2, scoring=None, fit_params=None, 
    random_iterations=None):
  start('starting grid search')
  if type(n_samples) is float: n_samples = int(len(y) * n_samples)
  reseed(clf)
  cv = sklearn.cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, random_state=cfg['sys_seed'])
  if random_iterations is None:
    gs = sklearn.grid_search.GridSearchCV(clf, params, cv=cv, 
      n_jobs=n_jobs, verbose=2, scoring=scoring or cfg['scoring'], fit_params=fit_params)
  else:
    gs = sklearn.grid_search.RandomizedSearchCV(clf, params, random_iterations, cv=cv, 
      n_jobs=n_jobs, verbose=2, scoring=scoring or cfg['scoring'], 
      fit_params=fit_params, refit=False)
  X2, y2 = sklearn.utils.shuffle(X, y, random_state=cfg['sys_seed'])  
  gs.fit(X2[:n_samples], y2[:n_samples])
  stop('done grid search')
  dbg(gs.best_params_, gs.best_score_)  
  return gs

def dump(file, data, force=False):  
  if not os.path.isdir('data/pickles'): os.makedirs('data/pickles')
  if not '.' in file: file += '.pickle'
  if os.path.isfile(file) and not force:
    raise Exception('file: ' + file + ' already exists. Set force=True to overwrite.')
  sklearn.externals.joblib.dump(data, 'data/pickles/' + file);  

def load(file, opt_fallback=None):
  start('loading file: ' + file)
  full_file = 'data/pickles/' + file
  if not '.' in full_file: full_file += '.pickle'
  if os.path.isfile(full_file): 
    if full_file.endswith('.npy'): return np.load(full_file)
    else: return sklearn.externals.joblib.load(full_file);
  if opt_fallback is None: return None
  data = opt_fallback()
  dump(file, data)
  stop('done loading file: ' + file)
  return data
  
def read_df(file, nrows=None):
  start('reading dataframe: ' + file)
  if file.endswith('.pickle'): 
    df = load(file)
  else:

    sep = '\t' if '.tsv' in file else ','
    if file.endswith('.7z'):
      import libarchive
   
      with libarchive.reader(file) as reader:
        df = pd.read_csv(reader, nrows=nrows, sep=sep);
    elif file.endswith('.zip'):
      import zipfile
      zf = zipfile.ZipFile(file)
      if len(zf.filelist) != 1: raise Exception('zip files with multiple files not supported')
      with zf.open(zf.filelist[0].filename) as reader:
        df = pd.read_csv(reader, nrows=nrows, sep=sep);
    else:
      compression = 'gzip' if file.endswith('.gz') else None
      nrows = None if nrows == None else int(nrows)  
      df = pd.read_csv(file, compression=compression, nrows=nrows, sep=sep);
  stop('done reading dataframe')
  return df

def optimise(predictions, y, scorer):
  def scorer_func(weights):
    means = np.average(predictions, axis=0, weights=weights)
    s = scorer(y, means)  
    if cfg['scoring_higher_better']: s *= -1
    return s

  starting_values = [0.5]*len(predictions)
  cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
  bounds = [(0,1)]*len(predictions)
  res = scipy.optimize.minimize(scorer_func, starting_values, 
      method='Nelder-Mead', bounds=bounds, constraints=cons)
  dbg('Ensamble Score: {best_score}'.format(best_score=res['fun']))
  dbg('Best Weights: {weights}'.format(weights=res['x']))

def calibrate(y_train, y_true, y_test=None, method='platt'):      
  if method == 'platt':    
    clf = sklearn.linear_model.LogisticRegression()
    if y_test is None:
      return pd.DataFrame({'train': y_train, 'const': np.ones(len(y_train))}).self_predict_proba(clf, y_true)
    else:
      return pd.DataFrame(y_train).predict_proba(clf, y_true, y_test)      
  elif method == 'isotonic':    
    clf = sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')    
    if len(y_train.shape) == 2 and y_train.shape[1] > 1:            
      all_preds = []
      for target in range(y_train.shape[1]):
        y_train_target = pd.DataFrame(y_train[:,target])        
        y_true_target = (y_true == target).astype(int)
        if y_test is None:
          preds = y_train_target.self_transform(clf, y_true_target)
        else:
          y_test_target = y_test[:,target]
          preds = y_train_target.transform(clf, y_true_target, y_test_target)
        all_preds.append(preds)
      return np.asarray(all_preds).T
    else:      
      if y_test is None:
        res = pd.DataFrame(y_train).self_transform(clf, y_true).T[0]
      else:
        res = pd.DataFrame(y_train).transform(clf, y_true, y_test)
      return np.nan_to_num(res)

def xgb_picker(clf, X, y):
  clf = sklearn.base.clone(clf)
  def do(prop, vals):    
    target = clf.base_classifier if hasattr(clf, 'base_classifier') else clf
    v = score_classifier_vals(prop, vals, clf, X, y, 5)[0]['v']  
    setattr(target, prop, v)
  do('max_depth', range(3, 10))
  do('learning_rate', [.001, .01, .025, .1, .2, .5])
  do('n_estimators', [50, 75, 100, 150, 200, 250, 300, 350])
  do('min_child_weight', [1, 2, 5, 10])
  do('subsample', [.5, .6, .8, .9, .95, 1.])
  do('colsample_bytree', [.5, .6, .8, .9, .95, 1.])
  return clf


def self_predict(clf, X, y, cv=5):    
  return self_predict_impl(clf, X, y, cv, 'predict')

def self_predict_proba(clf, X, y, cv=5):    
  return self_predict_impl(clf, X, y, cv, 'predict_proba')

def self_transform(clf, X, y, cv=5):    
  return self_predict_impl(clf, X, y, cv, 'transform')

def self_predict_impl(clf, X, y, cv, method):    
  if type(y) is not pd.Series: y = pd.Series(y)
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
    
    this_clf = sklearn.base.clone(clf)
    this_clf.fit(X, y)  
    new_predictions = getattr(this_clf, method)(X2)
    if new_predictions.shape[0] == 1:      
      new_predictions = new_predictions.reshape(-1, 1)
    return new_predictions    
  
  predictions = self_chunked_op(X, y, op, cv)
  stop('self_predict completed')  
  return predictions.values

def self_chunked_op(X, y, op, cv=5):    
  if y is not None and hasattr(y, 'values'): y = y.values
  if cv is None: cv = 5
  if type(cv) is int: cv = sklearn.cross_validation.StratifiedKFold(y, cv, shuffle=True, random_state=cfg['sys_seed'])
  indexes=None
  chunks=None
  for train_index, test_index in cv:
    X_train = X.iloc[train_index] if hasattr(X, 'iloc') else X[train_index]
    y_train = y[train_index]
    X_test = X.iloc[test_index] if hasattr(X, 'iloc') else X[test_index]
    predictions = op(X_train, y_train, X_test)
    indexes = test_index if indexes is None else np.concatenate((indexes, test_index))
    chunks = predictions if chunks is None else np.concatenate((chunks, predictions))
  df = pd.DataFrame(data=chunks, index=indexes)
  return df.sort()

def dbg(*args): 
  if cfg['debug']: print(*args)

cfg = {
  'sys_seed':0,
  'debug':True,
  'scoring': None,
  'scoring_higher_better': True,
  'indent': 0,
  'cv_n_jobs': -1,
  'custom_cv': None
}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', filename='output.log', filemode='w')
log = logging.getLogger(__name__)
reseed(None)