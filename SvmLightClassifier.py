
from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base, math
import numpy as np
import pandas as pd
from pandas_extensions import * 
from ExeEstimator import *

_default_path = 'utils/lib/'

class SvmLightClassifier(ExeEstimator, sklearn.base.ClassifierMixin):
  def __init__(self):
    ExeEstimator.__init__(self)

  def fit(self, X, y=None):
    if type(X) is str:
        train_file = X
    else:
        train_file = self.tmpfile('_svmlight_train')    
        X.to_svmlight(train_file, y)    
        
    self.model_file = self.save_tmp_file(X, '_svmlight_model', True)

    command = _default_path + 'svm_learn.exe'
    command += ' ' + train_file
    command += ' ' + self.model_file
    running_process = self.make_subprocess(command)
    self.close_process(running_process)
    return self

  def predict(self, X):    
    
    output_file = self.tmpfile('_svmlight_predictions')
    if type(X) is str:
        test_file = X
    else:
        test_file = self.tmpfile('_svmlight_test')
        X.to_svmlight(test_file)    

    command = _default_path + 'svm_classify.exe ' + test_file + ' ' + self.model_file + ' ' + output_file
    running_process = self.make_subprocess(command)
    self.close_process(running_process)
    preds = list(self.read_predictions(output_file))
    return preds

  def predict_proba(self, X):            
    predictions = np.asarray(map(lambda p: 1 / (1 + math.exp(-p)), self.predict(X)))
    return np.vstack([1 - predictions, predictions]).T
