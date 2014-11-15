import os
from ../misc import *

class DataLoader():
  def __init__(self, data_dir, loaders, in_mem_cache=False):
    assert type(data_dir) is str
    assert type(loaders) is dict
    assert type(in_mem_cache) is bool

    self.in_mem_cache = in_mem_cache
    self.cache = {} if in_mem_cache else None
    self.data_dir = data_dir
    self.loaders = loaders
    
    if not os.path.isdir(self.data_dir): raise Exception(data_dir + ' could not be found')

  def get_dataset(self, name, cache_results=True):
    assert type(name) is str
    assert type(cache_results) is bool

    if self.in_mem_cache and name in self.cache: return self.cache[name]    
    pickle_file = self.data_dir + 'dataset.' + name + '.pickle'
    
    if os.path.isfile(pickle_file): 
      try: return load(pickle_file)
      except: pass

    dataset = self.loaders[name](self, name)
    if not dataset: raise Exception('Could not load dataset: ' + name)
    if cache_results: 
      dump(pickle_file, dataset)
      if self.in_mem_cache: self.cache[name] = dataset        

    return dataset

