import pandas as pd, numpy as np
import itertools, ast_parser, scipy
from .. import misc

def _df_engineer(self, name, columns=None, quiet=False):  
  '''
  name(Array|string): Can list-like of names.  ';' split list of names 
  also supported

  TODO: Tons of duplicated code here, fix
  '''
  if type(name) is str and ';' in name: name = name.split(';')
  if type(name) is list or type(name) is tuple: 
    for n in name: self.engineer(n)
    return self

  def func_to_string(c):
    func = c.func
    args = c.args
    return func + '(' + ','.join(map(lambda a: 
      func_to_string(a) if hasattr(a, 'func') else a, args)) + ')'
  
  def get_new_col_name(c):
    prefix = 'c_' if c.func == 'concat' else 'n_'    
    suffix = func_to_string(c)
    return suffix if suffix.startswith(prefix) else prefix + suffix
  
  c = ast_parser.explain(name)[0]
  func = c.func if not type(c) is str else None
  args = c.args if not type(c) is str else None

  new_name = get_new_col_name(c) if not type(c) is str else c
  if new_name in self.columns: return self # already created column  

  # Evaluate any embedded expressions in the 'name' expression
  for i, a in enumerate(args): 
    if hasattr(a, 'func'): 
      args[i] = get_new_col_name(a)
      self.engineer(func_to_string(a))

  if not quiet: misc.debug('engineering feature: ' + name + ' new column: ' + new_name)
  if len(args) == 0 and (func == 'avg' or func == 'mult' or func == 'add' or func == 'concat'):    
    combs = list(itertools.combinations(columns, 2)) if columns is not None \
      else self.combinations(categoricals=func=='concat', indexes=func=='concat', numericals=func in ['mult', 'avg', 'add'])    
    for c1, c2 in combs: self.engineer(func + '(' + c1 + ',' + c2 + ')', quiet=True)
    return self
  if len(args) == 0 and (func == 'div' or func == 'subtract'):
    combs = list(itertools.combinations(columns, 2, permutations=True)) if columns is not None \
      else self.combinations(numericals=True, permutations=True)    
    for c1, c2 in combs: self.engineer(func + '(' + c1 + ',' + c2 + ')', quiet=True)
    return self
  elif func == 'concat': 
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = self[args[0]].astype(str) + self[args[1]].astype(str)
    if len(args) == 3: 
      self[new_name] = self[args[0]].astype(str) + self[args[1]].astype(str) + self[args[2]].astype(str)
  elif func  == 'mult' or func  == 'add':     
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    s1, s2 = self[args[0]], self[args[1]]
    if len(args) == 2: 
      self[new_name] = s1 * s2 if func == 'mult' else s1 + s2
    if len(args) == 3: 
      s3 = self[args[2]]
      self[new_name] = s1 * s2 * s3 if func == 'mult' else s1 + s2 + s3
  elif func  == 'div' or func  == 'subtract':     
    if len(args) != 2: raise Exception(name + ' only supports 2 columns')
    s1, s2 = self[args[0]].astype(float), self[args[1]].astype(float)
    self[new_name] = s1 / s2 if func == 'div' else s1 - s2    
  elif func  == 'avg':     
    if len(args) < 2 or len(args) > 3: raise Exception(name + ' only supports 2 or 3 columns')
    if len(args) == 2: 
      self[new_name] = (self[args[0]] + self[args[1]]) / 2
    if len(args) == 3: 
      self[new_name] = (self[args[0]] + self[args[1]] + self[args[2]]) / 3
  elif len(args) == 1 and func == 'pow':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('pow(' + n + ', ' + args[0] + ')', quiet=True)
    return self
  elif len(args) == 1 and func == 'round':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('round(' + n + ', ' + args[0] + ')', quiet=True)
    return self
  elif len(args) == 0 and func == 'lg':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('lg(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'safe_lg':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('safe_lg(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'boxcox':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('boxcox(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'safe_boxcox':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('safe_boxcox(' + n + ')', quiet=True)    
    return self
  elif len(args) == 0 and func == 'sqrt':
    cols = columns if columns is not None else self.numericals()
    for n in cols: self.engineer('sqrt(' + n + ')', quiet=True)    
    return self
  elif func == 'pow': 
    self[new_name] = np.power(self[args[0]], int(args[1]))
  elif func == 'round': 
    self[new_name] = self[args[0]].round(int(args[1]))
  elif func == 'lg': 
    self[new_name] = np.log(self[args[0]])
  elif func == 'safe_lg': 
    self[new_name] = np.log(self[args[0]] + 1 - self[args[0]].min())
  elif func == 'boxcox': 
    self[new_name] = scipy.stats.boxcox(self[args[0]])[0]
  elif func == 'safe_boxcox': 
    self[new_name] = scipy.stats.boxcox(self[args[0]] + 1 - self[args[0]].min())[0]
  elif func == 'sqrt': 
    self[new_name] = np.sqrt(self[args[0]])
  elif func.startswith('rolling_'):
    if len(args) == 1:
      cols = columns if columns is not None else self.numericals()
      for n in cols: self.engineer(func + '(' + n + ', ' + args[0] + ')', quiet=True)
      return self
    else:      
      self[new_name] = getattr(pd, func)(self[args[0]], int(args[1]))
  else: raise Exception(name + ' is not supported')

  # Absolutely no idea why this is required but if removed 
  #   pandas_extensions_engineer_tests.py T.test_long_method_chains
  #   fails.  Its like this locks the results in before the next
  #   method is called and the next method appears to change the
  #   scale of the array? Who knows...
  self[new_name]

  return self
  