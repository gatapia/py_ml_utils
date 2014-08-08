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

def reseed_(clf):
  clf.random_state = sys_seed
  random.seed(sys_seed)
  np.random.seed(sys_seed) 

def mean_score(scores):
  return ("{0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def scale(X, min_max=None):  
  pp = preprocessing
  scaler = pp.MinMaxScaler(min_max) if min_max else pp.StandardScaler()
  return scaler.fit_transform(X)

# Does a search through n_samples_arr to test what n_samples is acceptable
#   for cross validation.  No use using very high n_samples if not required
def do_n_sample_search(clf, X, y, n_samples_arr):
  reseed_(clf)

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


def do_cv(clf, X, y, n_samples=1000, n_iter=3, test_size=0.1, quiet=False, scoring=None, stratified=False):
  t0 = time.time()
  reseed_(clf)
  if (n_samples > X.shape[0]): n_samples = X.shape[0]
  cv = cross_validation.ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=sys_seed) \
    if not(stratified) else cross_validation.StratifiedShuffleSplit(y, n_iter, train_size=n_samples, test_size=test_size, random_state=sys_seed)

  test_scores = cross_validation.cross_val_score(clf, X, y, cv=cv, scoring=scoring)
  if (not(quiet)): 
    print '%s took: %.1f' % (mean_score(test_scores), time.time() - t0)
  return (np.mean(test_scores), sem(test_scores))

def split(X, y, test_split=0.1):
  X, y = utils.shuffle(X, y, random_state=sys_seed)  
  num_split = math.floor(X.shape[0] * test_split) if type(test_split) is float else test_split
  test_X, test_y = X[:num_split], y[:num_split]
  X, y = X[num_split:], y[num_split:]
  return X, y, test_X, test_y

def score(clf, X, y, test_split=0.1):
  X, y, test_X, test_y = split(X, y, test_split)
  reseed_(clf)
  clf.fit(X, y)
  score_(test_y, clf.predict(test_X))

def score_(y_true, y_pred):  
  accuracy = metrics.accuracy_score(y_true, y_pred)
  matrix = metrics.confusion_matrix(y_true, y_pred)
  report = metrics.classification_report(y_true, y_pred)
  print 'Accuracy: ', accuracy, '\n\nMatrix:\n', matrix, '\n\nReport\n', report

def do_gs(clf, X, y, params, n_samples=1000, cv=3, n_jobs=-1, scoring=None):
  reseed_(clf)
  gs = grid_search.GridSearchCV(clf, params, cv=cv, n_jobs=n_jobs, verbose=2, scoring=scoring)
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

def to_index(df, columns, drop_original=True):
  for col in columns:
    labels = pd.Categorical.from_array(df[col]).labels
    df[col + '_indexes'] = pd.Series(labels)
  if (drop_original): df = df.drop(columns, 1)
  return df
