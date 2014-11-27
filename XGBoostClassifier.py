from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import sys
sys.path.append('lib')
import xgboost as xgb

class XGBoostClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, silent=True,
      use_buffer=True, num_round=10, ntree_limit=0,
      nthread=None, booster='gbtree', 
      eta=0.3, gamma=0.01, 
      max_depth=6, min_child_weight=1, subsample=1, 
      colsample_bytree=1,      
      l=0, alpha=0, lambda_bias=0, objective='reg:linear',
      eval_metric=None, seed=0
      ):    
    assert booster in ['gbtree', 'gblinear']
    assert objective in ['reg:linear', 'reg:logistic', 
      'binary:logistic', 'binary:logitraw', 'multi:softmax',
      'rank:pairwise']
    assert eval_metric in [None, 'rmse', 'logloss', 'error', 'merror',  
      'auc', 'ndcg', 'map', 'ndcg@n', 'map@n']

    self.silent = silent
    self.use_buffer = use_buffer
    self.num_round = num_round
    self.ntree_limit = ntree_limit
    self.nthread = nthread 
    self.booster = booster
    # Parameter for Tree Booster
    self.eta=eta
    self.gamma=gamma
    self.max_depth=max_depth
    self.min_child_weight=min_child_weight
    self.subsample=subsample
    self.colsample_bytree=colsample_bytree
    # Parameter for Linear Booster
    self.l=l
    self.alpha=alpha
    self.lambda_bias=lambda_bias
    # Misc
    self.objective=objective
    self.eval_metric=eval_metric
    self.seed=seed


  def fit(self, X, y):    
    if hasattr(y, 'values'): y = y.values
    X = X if hasattr(X, 'handle') else xgb.DMatrix(X, y)
    param = {
      'silent':0 if self.silent else 1, 
      'use_buffer': int(self.use_buffer),
      'num_round': self.num_round,
      'ntree_limit': self.ntree_limit,
      'nthread': self.nthread,
      'booster': self.booster,
      'eta': self.eta,
      'gamma': self.gamma,
      'max_depth': self.max_depth,
      'min_child_weight': self.min_child_weight,
      'subsample': self.subsample,
      'colsample_bytree': self.colsample_bytree,
      'l': self.l,
      'alpha': self.alpha,
      'lambda_bias': self.lambda_bias,
      'objective': self.objective,
      'eval_metric': self.eval_metric,
      'seed': self.seed
    }
    watchlist  = [(X,'train')]    
    self.bst = xgb.train(param, X, self.num_round, watchlist)

    return self

  def predict(self, X): 
    X = X if hasattr(X, 'handle') else xgb.DMatrix(X)
    return self.bst.predict(X)
  
  def predict_proba(self, X): 
    X = X if hasattr(X, 'handle') else xgb.DMatrix(X)
    return self.bst.predict(X)