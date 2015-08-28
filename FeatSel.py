from misc import *
import time
import operator
from sklearn.externals import joblib

class FeatSel():
  def __init__(self, X, y, n_samples=3500, n_iter=3, tol=0.0001, 
    scoring=None, mandatory_columns=[], n_jobs=1, 
    least_sem_of_top_x=0, higher_better=True, validate_Xy=None,
    replace_columns=None, epochs=None):
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
    self.replace_columns = replace_columns
    self.epochs = epochs
    if validate_Xy is not None and hasattr(validate_Xy[0], 'values'): 
      self.validate_Xy = (validate_Xy[0].values, validate_Xy[1])

  def run(self, clf):
    if hasattr(clf, 'max_features') and clf.max_features: clf.max_features = None
    selected = map(lambda f: {'feature': f, 'score': 0, 'sem': 0}, self.mandatory_columns)
    print 'starting feature selected, features: ', self.X.shape[1], 'n_jobs:', self.n_jobs
    t_whole = time.time()    

    last_best = {'score': -1e6 if self.higher_better else 1e6}
    epoch = 0
    while True:
      epoch += 1      
      t_iter = time.time()    
      iter_results = self.find_next_best_(selected, clf)
      iter_results = sorted(iter_results, reverse=self.higher_better, key=lambda s: s['score'])
      this_best = iter_results[0]    
      
      if self.least_sem_of_top_x > 0:
        best_sems = iter_results[:self.least_sem_of_top_x]
        best_sems = filter(lambda s: s['score'] > last_best['score'], best_sems)
        if best_sems:
          best_sems = sorted(best_sems, key=lambda s: s['sem'] / s['score'])
          this_best = best_sems[0]

      if self.validate_Xy:
        check_top = 10
        for i, r in enumerate(iter_results[:check_top]):
          selected_features = map(lambda s: s['feature'], selected)
          s = self.get_feat_score_(selected_features, r['feature'], clf, 
            self.validate_Xy[0], self.validate_Xy[1])
          s_improvement = s['score'] - last_best['score'] if self.higher_better \
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

      improvement = this_best['score'] - last_best['score'] if self.higher_better \
        else last_best['score'] - this_best['score']

      last_best = this_best
      if improvement <= 0: break      
      selected.append(this_best)

      feats = map(lambda s: self.column_names[s['feature']], selected) 
      print 'iteration %d took: %.2fm - [%.4f] features: %s' % (len(selected), (time.time() - t_iter)/60, this_best['score'], feats)

      if improvement <= self.tol: 
        print 'improvement of %.3f is less than tol: %.3f, exiting...' % (improvement, self.tol)
        break
      
      print 'self.epochs:', self.epochs, 'epoch:', epoch
      if self.epochs is not None and epoch >= self.epochs: 
        print 'max epochs reached, exiting...'
        break

    feats = map(lambda s: self.column_names[s['feature']], selected) 
    print 'feature selection took: %.2fm - [%.4f] features: %s' % ((time.time() - t_whole)/60, last_best['score'], feats)
    return selected
      
  def find_next_best_(self, selected, clf):
    selected_features = map(lambda s: s['feature'], selected)
    to_test = filter(lambda f: f not in selected_features, range(self.X.shape[1]))
    if self.n_jobs > 1:
      return joblib.Parallel(n_jobs=self.n_jobs, max_nbytes=1e6, mmap_mode='r')(
        joblib.delayed(self.get_feat_score_)\
          (selected_features, f, clf, self.X, self.y) for f in to_test)      
    else:
      return map(lambda f: self.get_feat_score_(selected_features, f, clf, self.X, self.y), to_test)

  def get_feat_score_(self, selected_features, f, clf, X, y):
    feats = list(selected_features) + [f]        
    if not self.replace_columns is None:
      feat_names = map(lambda f: self.column_names[f], feats)
      replacement_cols = self.replace_columns.keys()
      feats_without_replacements = filter(lambda f, i: f not in self.replacement_cols, zip(feat_names, feats))  
      feats_without_replacements = map(lambda t: t[1], feats_without_replacements)
      to_replace = feats_without_replacements = filter(lambda f: f in self.replacement_cols, feat_names)  

      Xt = X[:, feats_without_replacements]
      for tr in to_replace:
        Xt = np.hstack((Xt, self.replace_columns[tr]))
    else:
      Xt = X[:, feats]
    
    cfg['sys_seed'] = len(feats)
    score, sem = do_cv(clf, Xt, y, n_samples=self.n_samples, 
      n_iter=self.n_iter, scoring=self.scoring, quiet=True)
    return {'feature': f, 'score': score, 'sem': sem}
