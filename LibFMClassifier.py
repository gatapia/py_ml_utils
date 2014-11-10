import sys
sys.path.append('lib/pyfm')
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from pylibfm import FM
import numpy as np

class LibFMClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, num_iter=100, k0=True, k1=True, init_stdev=0.01, 
      validation_size=0.01, learning_rate_schedule='optimal', 
      initial_learning_rate=0.01, power_t=0.5, t0=0.001, task='classification'):        
    self.num_iter = num_iter
    self.k0 = k0
    self.k1 = k1
    self.init_stdev = init_stdev
    self.validation_size = validation_size
    self.learning_rate_schedule = learning_rate_schedule
    self.initial_learning_rate = initial_learning_rate
    self.power_t = power_t
    self.t0 = t0
    self.task = task
    self._fm = FM(num_iter, k0, k1, init_stdev, validation_size, 
      learning_rate_schedule, initial_learning_rate, power_t, t0, task)
    pass

  def fit(self, X, y):    
    self._fm.num_iter = self.num_iter
    self._fm.k0 = self.k0
    self._fm.k1 = self.k1
    self._fm.init_stdev = self.init_stdev
    self._fm.validation_size = self.validation_size
    self._fm.learning_rate_schedule = self.learning_rate_schedule
    self._fm.initial_learning_rate = self.initial_learning_rate
    self._fm.power_t = self.power_t
    self._fm.t0 = self.t0
    self._fm.task = self.task

    self._fm.fit(X, y)
    return self

  def predict(self, X): 
    return self._fm.predict(X)

  def predict_proba(self, X): 
    return self._fm.predict(X)
