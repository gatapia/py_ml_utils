
from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base, math
import numpy as np
import pandas as pd
from pandas_extensions import * 
from ExeEstimator import *

_default_path = 'utils/lib/'

class LibFFMClassifier(ExeEstimator, sklearn.base.ClassifierMixin):
  '''
    options:
    -l <lambda>: set regularization parameter (default 0)
    -k <factor>: set number of latent factors (default 4)
    -t <iteration>: set number of iterations (default 15)
    -r <eta>: set learning rate (default 0.1)
    -s <nr_threads>: set number of threads (default 1)
    -p <path>: set path to the validation set
    --quiet: quiet model (no output)
    --norm: do instance-wise normalization
    --no-rand: disable random update

    `--norm' helps you to do instance-wise normalization. When it is enabled,
    you can simply assign `1' to `value' in the data.
  '''
  def __init__(self, lambda_v=0, factor=4, iteration=15, eta=0.1, 
      nr_threads=1, quiet=False, normalize=None, no_rand=None):
    ExeEstimator.__init__(self)
    
    self.lambda_v = lambda_v
    self.factor = factor
    self.iteration = iteration
    self.eta = eta
    self.nr_threads = nr_threads
    self.quiet = quiet
    self.normalize = normalize
    self.no_rand = no_rand
    self.model_file = None    

  def fit(self, X, y=None):
    if type(X) is str:
        train_file = X
    else:
        train_file = self.tmpfile('_libffm_train')    
        X.to_libffm(train_file, y)    
        
    self.model_file = self.save_tmp_file(X, '_libffm_model', True)

    command = _default_path + 'ffm-train.exe' + ' -l ' + `self.lambda_v` + \
      ' -k ' + `self.factor` + ' -t ' + `self.iteration` + ' -r ' + `self.eta` + \
      ' -s ' + `self.nr_threads`
    if self.quiet: command += ' --quiet'
    if self.normalize: command += ' --norm'
    if self.no_rand: command += ' --no-rand'    
    command += ' ' + train_file
    command += ' ' + self.model_file
    running_process = self.make_subprocess(command)
    self.close_process(running_process)
    return self

  def predict(self, X):    
    test_file = self.tmpfile('_libffm_test')
    output_file = self.tmpfile('_libffm_predictions')
    X.to_libffm(test_file)    

    command = _default_path + 'ffm-predict.exe ' + test_file + ' ' + self.model_file + ' ' + output_file
    running_process = self.make_subprocess(command)
    self.close_process(running_process)
    preds = list(self.read_predictions(output_file))
    return preds

  def predict_proba(self, X):            
    predictions = np.asarray(map(lambda p: 1 / (1 + math.exp(-p)), self.predict(X)))
    return np.vstack([1 - predictions, predictions]).T
