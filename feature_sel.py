from misc import *
import time
import operator
from sklearn.externals import joblib

 # TODO 1): We should take into account the SEM of a new feature.  There is no
 #    point selecting a new feature if it improves score by 0.001% if the SEM
 #    is +/-0.01 (i.e. 10x greater than the improvement).
 # TODO 2): After each iteration we should do a test of all selected features
 #    to ensure they are all still required.  May be that the last feature
 #    added made another redundant.
def feature_select(clf, X, y, n_samples=3500, n_iter=3, tol=0.00001, scoring=None, mandatory_columns=[], n_jobs=1):
  if hasattr(clf, 'max_features') and clf.max_features: clf.max_features = None

  selected = map(lambda f: {'feature': f, 'score': 0, 'sem': 0}, mandatory_columns)
    
  print "starting feature selected, features: ", X.shape[1], "n_jobs:", n_jobs

  last_best = {'score': -1e6}
  while True:
    t0 = time.time()    
    iter_results = find_next_best_(selected, clf, X, y, n_samples, n_iter, scoring, n_jobs)
    this_best = max(iter_results, key=lambda s: s['score'])
    improvement = this_best['score'] - last_best['score']
    last_best = this_best
    if improvement <= 0: break
    selected.append(this_best)
    print "iteration %d took: %.2fs - features: %s" % (len(selected), (time.time() - t0)/60, selected)

    if improvement <= tol: 
      print "improvement of %.2f is less than tol: %.2f, exiting..." % (improvement, tol)
      break

  print "feature selection completed: ", selected
  return selected
    
def find_next_best_(selected, clf, X, y, n_samples, n_iter, scoring, n_jobs):
  selected_features = map(lambda s: s['feature'], selected)
  to_test = filter(lambda f: f not in selected_features, range(X.shape[1]))
  if n_jobs > 1:
    return joblib.Parallel(n_jobs=6, max_nbytes=1e6, mmap_mode='r')(
      joblib.delayed(get_feat_score_)(
        selected_features, f, clf, X, y, n_samples, n_iter, scoring) for f in to_test)      
  else:
    return map(lambda f: get_feat_score_(selected_features, f, clf, X, y, n_samples, n_iter, scoring), to_test)

def get_feat_score_(selected_features, f, clf, X, y, n_samples, n_iter, scoring):  
  feats = list(selected_features) + [f]
  Xt = X[:, feats]
  score, sem = do_cv(clf, Xt, y, n_samples=n_samples, n_iter=n_iter, scoring=scoring, quiet=True)
  return {'feature': f, 'score': score, 'sem': sem}
