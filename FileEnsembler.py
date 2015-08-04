from sklearn.base import BaseEstimator, TransformerMixin
import sys
sys.path.append('utils')
from misc import *

class FileEnsembler(BaseEstimator, TransformerMixin):
  def _get_files(self, files):
    if hasattr(files, '__call__'): return files()
    first = files[0]
    if type(first) is str: 
      loaded = [load(f) for f in files]
      if loaded[0] is None: raise Exception('could not load files')
      print 'loaded:', len(loaded)
      return loaded
    return files
