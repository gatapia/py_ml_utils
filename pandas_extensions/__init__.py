'''
Naming Conventions for Features:
c_ = categorical
i_ = categoricals as indexes
n_ = numerical
b_ = binary
d_ = date
'''

import series, dataframe, dataframe_engineer, dataframe_format_convert
import pandas as pd
from .. import misc

def _extend_df(name, function):
  df = pd.DataFrame({})
  if not 'pd_extensions' in misc.cfg and hasattr(df, name): raise Exception ('DataFrame already has a ' + name + ' method')
  setattr(pd.DataFrame, name, function)

def _extend_s(name, function):
  s = pd.Series([])
  if not 'pd_extensions' in misc.cfg and hasattr(s, name): raise Exception ('Series already has a ' + name + ' method')
  setattr(pd.Series, name, function)

'''
inject all _s_ and _df_ methods into pd.DataFrame and 
    pd.Series objects
'''
def inject(prefix, module):
  for method in [m for m in dir(module) if m.startswith(prefix)]: 
    extender = _extend_s if prefix == '_s_' else _extend_df
    extender(method.replace(prefix, ''), getattr(module, method))

inject('_s_', series)
inject('_df_', dataframe)
inject('_df_', dataframe_engineer)
inject('_df_', dataframe_format_convert)

# aliases for usefull methods
_extend_s('ohe', series._s_one_hot_encode)
_extend_s('toidxs', series._s_to_indexes)

_extend_df('ohe', dataframe._df_one_hot_encode)
_extend_df('toidxs', dataframe._df_to_indexes)
_extend_df('rm', dataframe._df_remove)
_extend_df('rmnas', dataframe._df_remove_nas)
_extend_df('nas', dataframe._df_missing)
_extend_df('eng', dataframe_engineer._df_engineer)


if not 'pd_extensions' in misc.cfg: misc.cfg['pd_extensions'] = True
