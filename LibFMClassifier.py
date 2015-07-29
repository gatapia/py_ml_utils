from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base
import numpy as np
import pandas as pd
import math
from pandas_extensions import * 
from ExeEstimator import *

class _LibFM(ExeEstimator):
  def __init__(self,
         executable='utils/lib/libfm',
         dim='1,1,8',
         init_stdev=0.1,
         iter=100,
         learn_rate=0.1,
         method='mcmc',
         regular=None,         
         task='c', 
         columns=None):
    ExeEstimator.__init__(self)
    assert method in ['sgd', 'sgda', 'als', 'mcmc']
    assert task in ['r', 'c']

    self.executable = executable
    self.dim = dim
    self.init_stdev = init_stdev
    self.iter = iter
    self.learn_rate = learn_rate
    self.method = method    
    self.regular = regular
    self.task = task
    self.columns = columns
    if hasattr(self.columns, 'tolist'): self.columns = self.columns.tolist()


  def fit(self, X, y=None):    
    if type(X) is str: self.train_file = X
    else: 
      if not hasattr(X, 'values'): X = pd.DataFrame(X, columns=self.columns)
      self.train_file = self.save_reusable('_libfm_train', 'to_libfm', X, y)
    return self

  def predict(self, X):    
    if type(X) is str: test_file = X
    else: 
      if not hasattr(X, 'values'): X = pd.DataFrame(X, columns=self.columns)
      test_file = self.save_reusable('_libfm_test', 'to_libfm', X)

    self.start_predicting(self.train_file, test_file)
    self.close_process(self.libfm_process)
    
    raw_preds = self.read_predictions(self.prediction_file)
    return np.asarray(list(raw_preds))    

  def predict_proba(self, X):   
    predictions = self.predict(X)
    return np.vstack([1 - predictions, predictions]).T

  def get_command(self, train_file, test_file, predictions_file):
    assert train_file and os.path.isfile(train_file)
    assert test_file and os.path.isfile(test_file)
    assert predictions_file
    args = [self.executable]
    if self.dim is not None: args.append('-dim ' + self.dim)
    if self.init_stdev is not None: args.append('-init_stdev ' + `self.init_stdev`)
    if self.iter is not None: args.append('-iter ' + `self.iter`)
    if self.learn_rate is not None: args.append('--learn_rate ' + `self.learn_rate`)
    if self.method is not None: args.append('-method ' + self.method)
    if self.regular is not None: args.append('-regular ' + `self.regular`)
    if self.task is not None: args.append('-task ' + self.task)
    args.append('-train ' + train_file)
    args.append('-test ' + test_file)
    args.append('-out ' + predictions_file)
    return ' '.join(args)


  def start_predicting(self, training_file, testing_file):
    self.prediction_file = self.tmpfile('libfm.prediction')    

    command = self.get_command(training_file, testing_file, self.prediction_file)
    self.libfm_process = self.make_subprocess(command)      


class LibFMRegressor(sklearn.base.RegressorMixin, _LibFM):
  pass

class LibFMClassifier(sklearn.base.ClassifierMixin, _LibFM):
  pass
