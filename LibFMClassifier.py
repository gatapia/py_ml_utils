from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base
import numpy as np
import pandas as pd
import math
from pandas_extensions import * 

_libfm_default_path = 'utils/lib/libfm'

class _LibFM(sklearn.base.BaseEstimator):
  def __init__(self,
         executable=_libfm_default_path,
         dim='1,1,8',
         init_stdev=0.1,
         iter=100,
         learn_rate=0.1,
         method='mcmc',
         regular=None,         
         task='c', 
         columns=None):
    assert method in ['sgd', 'sgda', 'als', 'mcmc']
    assert task in ['r', 'c']

    self.tmpdir = 'tmpfiles'
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
    if type(X) is np.ndarray: 
      if self.columns is None: raise Exception('LibFM requires columns be set')      
      X = pd.DataFrame(X, columns=self.columns)
    if type(X) is pd.DataFrame: X = X.to_libfm(y)    

    self.training_instances = X
    return self

  def predict(self, X):    
    if type(X) is np.ndarray: 
      if self.columns is None: raise Exception('LibFM requires columns be set')      
      X = pd.DataFrame(X, columns=self.columns)
    if type(X) is pd.DataFrame: X = X.to_libfm()    
    return self.predict_proba(X)

  def predict_proba(self, X):   
    if type(X) is np.ndarray: 
      if self.columns is None: raise Exception('LibFM requires columns be set')      
      X = pd.DataFrame(X, columns=self.columns)
    if type(X) is pd.DataFrame: X = X.to_libfm() 

    train_file = self.save_tmp_file(self.training_instances, True)
    test_file = self.save_tmp_file(X, False)
    self.start_predicting(train_file, test_file)
    self.close_process()
    os.remove(train_file)
    os.remove(test_file)
    
    predictions = np.asarray(list(self.read_predictions()))    
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


  def save_tmp_file(self, instances, training=True):    
    f = self.tmpfile('_tmp_' + ('training' if training else 'testing') + '_file.libfm')
    with open(f, 'wb') as fs: fs.write('\n'.join(instances))    
    return f

  def tmpfile(self, suffix):
    _, f = tempfile.mkstemp(dir=self.tmpdir, suffix=suffix)
    os.close(_)
    return self.tmpdir + '/' + f.split('\\')[-1]

  def close_process(self):
    assert self.libfm_process

    if self.libfm_process.wait() != 0:
      raise Exception("libfm_process %d (%s) exited abnormally with return code %d" % \
        (self.libfm_process.pid, self.libfm_process.command, self.libfm_process.returncode))

  def start_predicting(self, training_file, testing_file):
    self.prediction_file = self.tmpfile('libfm.prediction')    

    command = self.get_command(training_file, testing_file, self.prediction_file)
    self.libfm_process = self.make_subprocess(command)      

  def read_predictions(self):
    lines = []
    with open(self.prediction_file) as f:
        lines=map(float, map(lambda l: l.split('#')[0], 
            f.readlines()))
    os.remove(self.prediction_file)
    return lines

  def make_subprocess(self, command):    
    stdout = open('nul', 'w')
    stderr = sys.stderr

    print 'running command: "%s"' % str(command)
    commands = shlex.split(str(command))
    result = subprocess.Popen(commands, 
        stdout=stdout, stderr=stderr, 
        close_fds=sys.platform != "win32", 
        universal_newlines=True, cwd='.')
    result.command = command
    return result



class LibFMRegressor(sklearn.base.RegressorMixin, _LibFM):
  pass

class LibFMClassifier(sklearn.base.ClassifierMixin, _LibFM):
  pass
