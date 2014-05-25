from misc import *
import time
import operator

def feature_select(X_train, y_train, model, n_samples=3500, n_iter=3, 
    tol=0.00001, column_names=[]):
  score_hist = []
  good_features = set([])
  last_score = 0
  n_features = X_train.shape[1]
  column_names = column_names if len(column_names) == n_features else range(n_features)
  # FEATURE SELECTION
  while len(score_hist) < 2 or (score_hist[-1][0] - score_hist[-2][0] > tol):
    scores = []  
    this_best = last_score
    t0 = time.clock()
    for f in range(n_features):      
      if f % 100 == 0: print "Testing Feature %d of %d - %d%%" % (f, n_features, f * 100 / n_features)
      if f not in good_features:      
        feats = list(good_features) + [f]
        Xt = X_train[:, feats]
        model.max_features = min(len(feats), 15)      
        score, sem = do_cv(model, Xt, y_train, n_samples=n_samples, n_iter=n_iter, quiet=True)
        scores.append((score, f))            
        if (score - last_score > 0.1): 
          print "> 10 percent improvement with feature, auto adding. %s -> %s" % (last_score, score)
          break
        if (score > this_best): 
          this_best = score
          print "Feature: %i Mean: %f (+/-%.2f) Diff: %.5f" % (f, score, sem, score - last_score)
    
    best = sorted(scores)[-1]  
    print "Last: %f Best: %s Feature: %s" % (last_score, best[0], best[1])
    last_score = best[0]
    good_features.add(best[1])
    score_hist.append(best)
    sorted_good = sorted(list(good_features))
    print "Iteration Took: %.2fm - Current Features: %s" % \
      ((time.clock() - t0)/60, operator.itemgetter(sorted_good)(column_names))

  # Remove last added feature from good_features
  good_features.remove(score_hist[-1][1])
  good_features = sorted(list(good_features))
  print "Selected features %s - names %s" % (good_features, operator.itemgetter(good_features)(column_names))
  return good_features