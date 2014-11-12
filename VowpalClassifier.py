from VowpalWrapper import VW_Classifier

class VowpalClassifier(VW_Classifier):

  def fit(self, X, y=None):    
    self.vw = 'utils/lib/vw'
    super(VW_Classifier, self).fit(X, None)
    return self

  def predict(self, X): 
    return super(VW_Classifier, self).predict(X)

  def predict_proba(self, X): 
    return super(VW_Classifier, self).predict(X)
