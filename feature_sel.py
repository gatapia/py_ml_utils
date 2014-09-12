from misc import *
import time
import operator
from sklearn.externals import joblib

class FeatSel():
  def __init__(self, X, y, n_samples=3500, n_iter=3, tol=0.0001, 
    scoring=None, mandatory_columns=[], n_jobs=1, 
    least_sem_of_top_x=0, higher_better=True, validate_Xy=None):
    self.column_names = X.columns.tolist() if hasattr(X, 'columns') else range(0, X.shape[1])
    self.X = X.values if hasattr(X, 'values') else X
    self.y = y
    self.n_samples = n_samples
    self.n_iter = n_iter
    self.tol = tol
    self.scoring = scoring or cfg['scoring']
    self.mandatory_columns = mandatory_columns
    self.n_jobs = n_jobs
    self.least_sem_of_top_x = least_sem_of_top_x
    self.higher_better = higher_better
    self.validate_Xy = validate_Xy
    if validate_Xy is not None and hasattr(validate_Xy[0], 'values'): 
      self.validate_Xy = (validate_Xy[0].values, validate_Xy[1])

def run(clf):
  if hasattr(clf, 'max_features') and clf.max_features: clf.max_features = None
  selected = map(lambda f: {'feature': f, 'score': 0, 'sem': 0}, self.mandatory_columns)
  ohe_cache = {}
  print 'starting feature selected, features: ', X.shape[1], 'n_jobs:', n_jobs
  t_whole = time.time()    

  last_best = {'score': -1e6 if higher_better else 1e6}
  while True:
    t_iter = time.time()    
    iter_results = find_next_best_(selected, clf, X, y, n_samples, 
        n_iter, scoring, n_jobs)
    iter_results = sorted(iter_results, reverse=higher_better, key=lambda s: s['score'])
    this_best = iter_results[0]    
    
    if least_sem_of_top_x > 0:
      best_sems = iter_results[:least_sem_of_top_x]
      best_sems = filter(lambda s: s['score'] > last_best['score'], best_sems)
      if best_sems:
        best_sems = sorted(best_sems, key=lambda s: s['sem'] / s['score'])
        this_best = best_sems[0]

    if validate_Xy:
      check_top = 10
      for i, r in enumerate(iter_results[:check_top]):
        selected_features = map(lambda s: s['feature'], selected)
        s = get_feat_score_(selected_features, r['feature'], clf, 
          validate_Xy[0], validate_Xy[1], n_samples, n_iter, scoring)
        s_improvement = s['score'] - last_best['score'] if higher_better \
            else last_best['score'] - s['score']
        
        if s_improvement > 0: 
          if i > 0: 
            print 'changed selection as it did not perform' +\
              ' in the validation set. Old Best: ' + `this_best` + \
              ' New best:' + `r`
          this_best = r          
          break
        if i == check_top - 1: 
          print 'Could not find any feature that performed well ' +\
              'on validation set, stopping selecton.'
          this_best = last_best

    improvement = this_best['score'] - last_best['score'] if higher_better \
      else last_best['score'] - this_best['score']

    last_best = this_best
    if improvement <= 0: break
    selected.append(this_best)

    feats = map(lambda s: column_names[s['feature']], selected) if column_names else selected
    print 'iteration %d took: %.2fm - [%.4f] features: %s' % (len(selected), (time.time() - t_iter)/60, this_best['score'], feats)

    if improvement <= tol: 
      print 'improvement of %.3f is less than tol: %.3f, exiting...' % (improvement, tol)
      break

  feats = map(lambda s: column_names[s['feature']], selected) if column_names else selected
  print 'feature selection took: %.2fm - [%.4f] features: %s' % ((time.time() - t_whole)/60, last_best['score'], feats)
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

def get_feat_score_(selected_features, f, clf, X, y, n_samples, n_iter, scoring, ohe_cache=None):  
  feats = list(selected_features) + [f]
  Xt = X[:, feats]
  cfg['sys_seed'] = len(feats)
  score, sem = do_cv(clf, Xt, y, n_samples=n_samples, 
    n_iter=n_iter, scoring=scoring or cfg['scoring'], quiet=True)
  return {'feature': f, 'score': score, 'sem': sem}
