from sklearn import preprocessing, grid_search, utils, metrics, cross_validation

def visualise(X, y):
  for c in X.columns:
    if utils.multiclass.type_of_target(y) == 'binary' and
        utils.multiclass.type_of_target(X[c]) == 'multiclass':
      