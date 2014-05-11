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
from scipy.stats import sem 
from scipy.stats.mstats import mode
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

seed = 0
random.seed(seed)
np.random.seed(seed) # Should be used by all sklearn algorithms also
NA = 99999.0
selections = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def mean_score(scores):
  return ("Mean: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

def do_cv(model, X_train, y_train, n_samples=1000, n_iter=3, test_size=0.1, quiet=False):
  cv = ShuffleSplit(n_samples, n_iter=n_iter, test_size=test_size, random_state=seed)
  test_scores = cross_val_score(model, X_train, y_train, cv=cv)
  if (not(quiet)): print(mean_score(test_scores))  
  return (np.mean(test_scores), sem(test_scores))

def do_gs(model, X_train, y_train, params, n_samples=1000, cv=3, n_jobs=-1):
  gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=2) 
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

def make_and_save_predictions(model, train, y, test):
  model.fit(train, y)
  predictions = model.predict(test)
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

def fval(val):
  return NA if val.isnull().iget(0) else float(val.iget(0))

def get_policy(row):
  return row.A.iget(0) + row.B.iget(0) + row.C.iget(0) + row.D.iget(0) + row.E.iget(0) + row.F.iget(0) + row.G.iget(0)

def allsame(lst):
  return lst[1:] == lst[:-1]


all_policies_ = None
def policies_to_idx(data):
  global all_policies_
  if (all_policies_ == None): all_policies_ = read_data("all_policies_count.pz")
  for r in data:
    for i, v in enumerate(r):
      if (type(v) is str and len(v) == 7):
        r[i] = all_policies[v]['idx']

def policies_arr_to_idx(arr):
  global all_policies_
  if (all_policies_ == None): all_policies_ = read_data("all_policies_count.pz")
  for i, v in enumerate(arr):
    if (type(v) is str and len(v) == 7):
      arr[i] = all_policies[v]['idx']

idx_to_pol_ = None
def policy_idxs_to_strings(arr):
  global all_policies_
  global idx_to_pol_
  if (all_policies_ == None): all_policies_ = read_data("all_policies_count.pz")
  if (idx_to_pol_ == None):
    idx_to_pol_ = {}
    for key in all_policies: 
      idx_to_pol_[all_policies[key]['idx']] = key

  return map(lambda i: idx_to_pol_[i], arr)
