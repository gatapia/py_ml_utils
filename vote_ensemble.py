
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import numpy as np

class VotingEnsemble(BaseEstimator, ClassifierMixin):
  def __init__(self, models, voter='majority'):    
    self.models = models
    self.voter = voter

  def fit(self, X, y):
    for m in self.models: m.fit(X, y)

    return self

  def predict(self, X):
    all_preds = [m.predict(X) for m in self.models]
    predictions = np.empty(len(X), dtype=type(all_preds[0][0]));
    for i in range(len(X)):      
      i_preds = [ps[i] for ps in all_preds]      

      if (self.voter == 'majority'):                
        maj = stats.mode(i_preds)[0]
        predictions[i] = int(maj)
      else: raise Error(self.voter + ' is not implemented')
    return predictions