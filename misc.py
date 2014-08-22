import numpy as np
import pandas as pd
import scipy as scipy
import cPickle as pickle
from collections import Counter
import gzip, time, math, datetime, random
from sklearn import preprocessing, grid_search, utils, metrics, cross_validation
from scipy.stats import sem 
from scipy.stats.mstats import mode

sys_seed = 0
random.seed(sys_seed)
np.random.seed(sys_seed) 
NA = 99999.0

def _reseed(clf):
  clf.random_state = sys_seed
  random.seed(sys_seed)
  np.random.seed(sys_seed) 

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
  return ("{0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def scale(X, min_max=None):  
  pp = preprocessing
  scaler = pp.MinMaxScaler(min_max) if min_max else pp.StandardScaler()
  return scaler.fit_transform(X)

def fillnas(X, categoricals=[], categorical_fill='mode', numerical_fill='mean', inplace=False):
  if not (inplace): X = X.copy()
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
  _reseed(clf)

  scores = []
  sems = []
  for n_samples in n_samples_arr:
    cv = do_cv(clf, X, y, n_samples, quiet=True)
    print "n_samples:", n_samples, "cv:", cv
    scores.append(cv[0])
    sems.append(cv[1])
  max_score_idx = scores.index(max(scores))
  min_sem_idx = sems.index(min(sems))
  print "best score n_samples:", n_samples_arr[max_score_idx], "score:", scores[max_score_idx]
  print "best sem n_samples:", n_samples_arr[min_sem_idx], "sem:", sems[min_sem_idx]
  return (scores, sems)


def do_cv(clf, X, y, n_samples=1000, n_iter=3, test_size=0.1, quiet=False, scoring=None, stratified=False, fit_params=None, reseed=True):
  t0 = time.time()
  if reseed: _reseed(clf)
  if (n_samples > X.shape[0]): n_samples = X.shape[0]
  cv = cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=sys_seed) \
    if not(stratified) else cross_validation.StratifiedShuffleSplit(y, n_iter, train_size=n_samples, test_size=test_size, random_state=sys_seed)

  test_scores = cross_validation.cross_val_score(clf, X, y, cv=cv, scoring=scoring, fit_params=fit_params)
  if (not(quiet)): 
    print '%s took: %.2fm' % (mean_score(test_scores), (time.time() - t0)/60)
  return (np.mean(test_scores), sem(test_scores))

def split(X, y, test_split=0.1):
  X, y = utils.shuffle(X, y, random_state=sys_seed)  
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
  _reseed(clf)
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
    print 'auc: ', auc
    return auc

  if (utils.multiclass.type_of_target(y_true) == 'continuous' and
      utils.multiclass.type_of_target(y_pred) == 'continuous'):
    r2 = metrics.r2_score(y_true, y_pred)
    print 'r2: ', r2
    return r2

  accuracy = metrics.accuracy_score(y_true, y_pred)
  matrix = metrics.confusion_matrix(y_true, y_pred)
  report = metrics.classification_report(y_true, y_pred)
  print 'Accuracy: ', accuracy, '\n\nMatrix:\n', matrix, '\n\nReport\n', report
  return accuracy

def do_gs(clf, X, y, params, n_samples=1000, cv=3, n_jobs=-1, scoring=None, fit_params=None):
  _reseed(clf)
  gs = grid_search.GridSearchCV(clf, params, cv=cv, n_jobs=n_jobs, verbose=2, scoring=scoring, fit_params=fit_params)
  X2, y2 = utils.shuffle(X, y, random_state=sys_seed)  
  gs.fit(X2[:n_samples], y2[:n_samples])
  print(gs.best_params_, gs.best_score_)
  return gs

def save_data(file, data):
  if (file.endswith('z')):
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

def to_csv_gz(data_dict, file):
  in_name = file + '.uncompressed'
  pd.DataFrame(data_dict).to_csv(in_name, index=False)  
  f_in = open(in_name, 'rb')
  f_out = gzip.open(file, 'wb')
  f_out.writelines(f_in)
  f_out.close()
  f_in.close()
  os.remove(in_name) 

def to_index(df, columns, drop_originals=False):
  to_drop = []
  for col in columns:
    if type(col) is int: col = df.columns[col]
    labels = pd.Categorical.from_array(df[col]).labels
    df[col + '_indexes'] = pd.Series(labels)
    to_drop.append(col)
  return df.drop(to_drop, 1) if drop_originals else df
