from __future__ import print_function
import sys, gzip, time, datetime, random, os, logging, gc, \
    scipy, sklearn, sklearn.cross_validation, sklearn.grid_search,\
    sklearn.utils, sklearn.externals.joblib
import numpy as np, pandas as pd
from lib.xgb import XGBClassifier, XGBRegressor

def debug(msg): 
  if not cfg['debug']: return
  log.info(msg)

_message_timers = {}
def start(msg, id=None): 
  if not cfg['debug']: return
  id = id if id is not None else 'global'
  _message_timers[id] = time.time()
  log.info(msg)

def stop(msg, id=None): 
  if not cfg['debug']: return
  id = id if id is not None else 'global'
  took = datetime.timedelta(seconds=time.time() - _message_timers[id]) \
    if id in _message_timers else 'unknown'
  log.info(msg + (', took (h:m:s): %s' % took))
  if id in _message_timers: del _message_timers[id]

def reseed(clf):
  if clf is not None: clf.random_state = cfg['sys_seed']
  random.seed(cfg['sys_seed'])
  np.random.seed(cfg['sys_seed']) 
  return clf

def seed(seed):
  cfg['sys_seed'] = seed
  reseed(None)

def do_cv(clf, X, y, n_samples=None, n_iter=3, test_size=None, quiet=False, 
      scoring=None, stratified=False, n_jobs=-1, fit_params=None):
  start('starting cv', 'cv')
  reseed(clf)
  
  if n_samples is None: n_samples = len(X)
  elif type(n_samples) is float: n_samples = int(n_samples)
  if scoring is None: scoring = cfg['scoring']
  if test_size is None: test_size = 1./n_iter
  
  try:
    if (n_samples > X.shape[0]): n_samples = X.shape[0]
  except: pass
  cv = sklearn.cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=cfg['sys_seed']) \
    if not(stratified) else sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter, train_size=n_samples, test_size=test_size, random_state=cfg['sys_seed'])
  if n_jobs == -1 and cfg['cv_n_jobs'] > 0: n_jobs = cfg['cv_n_jobs']

  test_scores = sklearn.cross_validation.cross_val_score(
      clf, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, 
      fit_params=fit_params)
  score_desc = ("{0:.5f} (+/-{1:.5f})").format(np.mean(test_scores), scipy.stats.sem(test_scores))
  stop('done CV %s' % score_desc, 'cv')
  return (np.mean(test_scores), scipy.stats.sem(test_scores))

def score_classifier_vals(prop, vals, clf, X, y):
  results = []
  for v in vals:      
    target_clf = clf.base_classifier if hasattr(clf, 'base_classifier') else clf
    target_clf = base.clone(target_clf)
    setattr(target_clf, prop, v)    
    score = do_cv(clf, X, y)
    results.append({'prop': prop, 'v':v, 'score': score})  
  sorted_results = sorted(results, key=lambda r: r['score'][0], reverse=True)
  best = {'prop': prop, 'value': sorted_results[0]['v'], 'score': sorted_results[0]['score']}
  dbg('\n\n\n\n', best)
  return sorted_results

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

def dump(file, data):  
  if not os.path.isdir('data/pickles'): os.makedirs('data/pickles')
  if not '.' in file: file += '.pickle'
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
    else:
      
      compression = 'gzip' if file.endswith('.gz') else None
      nrows = None if nrows == None else int(nrows)  
      df = pd.read_csv(file, compression=compression, nrows=nrows, sep=sep);
  stop('done reading dataframe')
  return df

def to_csv_gz(data_dict, file, columns=None):
  if file.endswith('.gz'): file = gzip.open(file, "wb")
  df = data_dict
  if type(df) is not pd.DataFrame: df = pd.DataFrame(df)
  df.to_csv(file, index=False, columns=columns)  

def optimise(predictions, y, scorer):
  def scorer_func(weights):
    means = np.average(predictions, axis=0, weights=weights)
    return -scorer(y, means)  

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

def dbg(*args): 
  if cfg['debug']: print(*args)

cfg = {
  'sys_seed':0,
  'debug':True,
  'scoring': None,
  'indent': 0,
  'cv_n_jobs': -1
}
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
reseed(None)