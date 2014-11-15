import os
from ../misc import *
from DataLoader import *

class Engine():
  def __init__(self, data_loader):
    self.data_loader = data_loader

  def do_cv(self, clf, clf_args, dataset_name, cache_dataset=True):
    X, y = self.data_loader.get_dataset(dataset_name, cache_dataset)
    if clf_args is None: clf_args = {}
    for arg_name in clf_args:
      setattr(clf, arg_name, clf_args[arg_name])
    
    r = {'error': '', 'clf': clf, 'dataset': dataset, 'score': -1, 'sem': -1}
    try: r['score'], r['sem'] = do_cv(clf, X, y, len(y))
    except Exception, ex: r['error'] = str(ex)
    print r

    return r

