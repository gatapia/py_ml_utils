from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import sys
sys.path.append('lib')
import xgboost as xgb
import numpy as np

class XGBoostClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, silent=True,
      use_buffer=True, num_round=10, ntree_limit=0,
      nthread=None, booster='gbtree', 
      eta=0.3, gamma=0.01, 
      max_depth=6, min_child_weight=1, subsample=1, 
      colsample_bytree=1,      
      l=0, alpha=0, lambda_bias=0, objective='reg:linear',
      eval_metric=None, seed=0, num_class=None,
      max_delta_step=0
      ):    
    assert booster in ['gbtree', 'gblinear']
    assert objective in ['reg:linear', 'reg:logistic', 
      'binary:logistic', 'binary:logitraw', 'multi:softmax',
      'multi:softprob', 'rank:pairwise']
    assert eval_metric in [None, 'rmse', 'mlogloss', 'logloss', 'error', 
      'merror',  'auc', 'ndcg', 'map', 'ndcg@n', 'map@n']

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
    self.max_delta_step=max_delta_step
    # Parameter for Linear Booster
    self.l=l
    self.alpha=alpha
    self.lambda_bias=lambda_bias
    # Misc
    self.objective=objective
    self.eval_metric=eval_metric
    self.seed=seed
    self.num_class = num_class

  def build_matrix(self, X, opt_y=None):
    if hasattr(X, 'values'): X = X.values
    if opt_y is not None and hasattr(opt_y, 'values'): opt_y = opt_y.values
    return X if hasattr(X, 'handle') else xgb.DMatrix(X, opt_y, missing=np.nan)

  def cv(self, X, y): 
    X = self.build_matrix(X, y)
    param = {
      'silent': 1 if self.silent else 0, 
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
      'max_delta_step': self.max_delta_step,
      'l': self.l,
      'alpha': self.alpha,
      'lambda_bias': self.lambda_bias,
      'objective': self.objective,
      'eval_metric': self.eval_metric,
      'seed': self.seed,
      'num_class': self.num_class,
    }    
    results = xgb.cv(param, X, self.num_round, 3)
    return results

  def fit(self, X, y):    
    X = self.build_matrix(X, y)
    param = {
      'silent': 1 if self.silent else 0, 
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
      'max_delta_step': self.max_delta_step,
      'l': self.l,
      'alpha': self.alpha,
      'lambda_bias': self.lambda_bias,
      'objective': self.objective,
      'eval_metric': self.eval_metric,
      'seed': self.seed          
    }
    if self.num_class is not None:
      param['num_class']= self.num_class

    watchlist  = [(X,'train')]    
    self.bst = xgb.train(param, X, self.num_round, watchlist)

    return self

  def predict(self, X): 
    X = self.build_matrix(X)
    return self.bst.predict(X)
  
  def predict_proba(self, X): 
    X = self.build_matrix(X)
    predictions = self.bst.predict(X)
    if self.objective == 'multi:softprob': return predictions
    return np.vstack([1 - predictions, predictions]).T