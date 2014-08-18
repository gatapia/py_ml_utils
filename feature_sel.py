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
def feature_select(clf, X, y, n_samples=3500, n_iter=3, tol=0.0001, 
    scoring=None, mandatory_columns=[], n_jobs=1, least_sem_of_top_x=0):
  if hasattr(clf, 'max_features') and clf.max_features: clf.max_features = None

  column_names = X.columns.tolist() if hasattr(X, 'columns') else None
  X = X.values if hasattr(X, 'values') else X
  selected = map(lambda f: {'feature': f, 'score': 0, 'sem': 0}, mandatory_columns)
    
  print 'starting feature selected, features: ', X.shape[1], 'n_jobs:', n_jobs
  t_whole = time.time()    

  last_best = {'score': -1e6}
  while True:
    t_iter = time.time()    
    iter_results = find_next_best_(selected, clf, X, y, n_samples, 
        n_iter, scoring, n_jobs)
    iter_results = sorted(iter_results, reverse=True, key=lambda s: s['score'])
    this_best = iter_results[0]    
    
    if least_sem_of_top_x > 0:
      best_sems = iter_results[:least_sem_of_top_x]
      best_sems = filter(lambda s: s['score'] > last_best['score'], best_sems)
      if best_sems:
        best_sems = sorted(best_sems, key=lambda s: s['sem'] / s['score'])
        this_best = best_sems[0]

    improvement = this_best['score'] - last_best['score']
    last_best = this_best
    if improvement <= 0: break
    selected.append(this_best)

    feats = map(lambda s: column_names[s['feature']], selected) if column_names else selected
    print 'iteration %d took: %.2fm - features: %s' % (len(selected), (time.time() - t_iter)/60, feats)

    if improvement <= tol: 
      print 'improvement of %.3f is less than tol: %.3f, exiting...' % (improvement, tol)
      break

  feats = map(lambda s: column_names[s['feature']], selected) if column_names else selected
  print 'feature selection took: %.2fm - features: %s' % ((time.time() - t_whole)/60, feats)
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
  score, sem = do_cv(clf, Xt, y, n_samples=n_samples, n_iter=n_iter, scoring=scoring, quiet=True, reseed=False)
  return {'feature': f, 'score': score, 'sem': sem}
