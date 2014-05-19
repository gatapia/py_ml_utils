import sklearn as sk
import numpy as np
import pandas as pd
import scipy as scipy
import cPickle as pickle
import gzip
import math
import datetime
import random

from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem 
from scipy.stats.mstats import mode
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

sys_seed = 0
random.seed(sys_seed)
np.random.seed(sys_seed) 
NA = 99999.0

def mean_score(scores):
  return ("Mean: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def scale(X):  
  return StandardScaler().fit_transform(X)

def reseed_(clf):
  clf.random_state = sys_seed
  random.seed(sys_seed)
  np.random.seed(sys_seed) 

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
  print "Best Score n_samples:", n_samples_arr[max_score_idx], "Score:", scores[max_score_idx]
  print "Best Sem n_samples:", n_samples_arr[min_sem_idx], "Sem:", sems[min_sem_idx]
  return (scores, sems)


def do_cv(clf, X_train, y_train, n_samples=1000, n_iter=3, test_size=0.1, quiet=False):
  reseed_(clf)
  if (n_samples > len(X_train)): n_samples = len(X_train)
  cv = ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=sys_seed)
  test_scores = cross_val_score(clf, X_train, y_train, cv=cv)
  if (not(quiet)): 
    print(mean_score(test_scores))  
  return (np.mean(test_scores), sem(test_scores))

def do_gs(clf, X_train, y_train, params, n_samples=1000, cv=3, n_jobs=-1):
  reseed_(clf)
  gs = GridSearchCV(clf, params, cv=cv, n_jobs=n_jobs, verbose=2) 
  gs.fit(X_train[:n_samples], y_train[:n_samples])
  print(gs.best_params_, gs.best_score_)
  return gs

def load_train_test_and_y(file):
  data = read_data(file)
  X_train = data['train_munged']
  X_test = data['test_munged']
  y_train = data['y']
  print "Loaded train[" + `len(X_train)` + "]"
  return (X_train, X_test, y_train)

def save_train_test_and_y(file, xtrain, xtest, y):
  save_data(file, {'train_munged': xtrain, 'test_munged': xtest, 'y': y})

def save_data(file, data):
  if (file.endswith('z')):
    f = gzip.GzipFile("../data/" + file, 'wb')
    f.write(pickle.dumps(data, 0))
    f.close()
  else:
    f = open("../data/" + file, "wb" )
    pickle.dump(data, f)
    f.close()

def read_data(file):
  if (file.endswith('z')):
    f = gzip.GzipFile("../data/" + file, 'rb')
    buffer = ""
    while True:
      data = f.read()
      if data == "": break
      buffer += data
    object = pickle.loads(buffer)
    f.close()
    return object
  else:
    f = open("../data/" + file, "rb" )
    data = pickle.load(f)
    f.close()
    return data

def make_and_save_predictions(clf, train, y, test):
  clf.fit(train, y)
  predictions = clf.predict(test)
  make_and_save_predictions_impl(predictions)  

def make_and_save_predictions_impl(predictions, file_suffix='submission'):
  test_file = pd.read_csv('../data/test_v2.csv', dtype=object, usecols=[0])
  test_cust_ids = np.unique(test_file.customer_ID)  
  id_pred = zip(test_cust_ids, predictions)
  content = ['customer_ID,plan']
  for i, ip in enumerate(id_pred): content.append(ip[0] + ',' + ip[1])
  f = open('../data/submissions/' + str(datetime.date.today()) + file_suffix + '.csv', 'w')
  f.write('\n'.join(content))
  f.close()

# For Data Munging
def idx(list, val):
  if (type(val) is pd.core.series.Series and val.isnull().iget(0)): return NA
  if (type(val) is pd.core.series.Series): val = val.iget(0)  
  idx = list[val]
  if (idx < 0): raise Exception('Error')
  return idx

def allsame(lst):
  return lst[1:] == lst[:-1]
