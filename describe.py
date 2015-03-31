from IPython.nbformat import current as nbf
import sklearn
from misc import *
from pandas_extensions import *

class Describe():
  def __init__(self, X, opt_y=None, columns=None):
    self.columns = X.columns if columns is None else columns
    self.cells = []
    self.nb = nbf.new_notebook()
    self.original_rows = X.shape[0]
    Xy = X.subsample(opt_y, 1e6)  
    self.X = Xy if opt_y is None else Xy[0]
    self.y = None if opt_y is None else Xy[1]
    self.y_type = sklearn.utils.multiclass.type_of_target(self.y) if self.y is not None else 'na'

    dump('_desc_data', (self.X, self.y, self.columns)) 

  def get_cells(self):
    if len(self.cells) > 0: return list(self.cells)

    self._do_global_imports()
    self._do_header_markdown()
    self._do_target_description()
    '''
    TODO: 
    - if len()
    '''
    for col in self.columns: self._do_column_description(col)
    return list(self.cells)

  def _txt(self, markdown): self.cells.append(nbf.new_text_cell('markdown', markdown))    

  def _code(self, code): self.cells.append(nbf.new_code_cell(code))    

  def _do_global_imports(self):
    imports=[
      'import numpy as np',
      'import pandas as pd',
      'import matplotlib.pyplot as plt',
      'from sys import path',
      'path.append("utils")',
      'from misc import *',
      '\npd.set_option("display.notebook_repr_html", False)',
      'pd.set_option("display.max_columns", 20)',
      'pd.set_option("display.max_rows", 25)',
      'pylab.rcParams[\'figure.figsize\'] = (16.0, 10.0)',
      'X, y, columns = load("_desc_data")',
      'X["_tmpy"] = y',      
    ]
    self._code('\n'.join(imports))

  def _do_header_markdown(self):
    header = {}      
    header['Number of columns'] = len(self.columns)  
    header['Number of rows'] = self.X.shape[0]
    if self.X.shape[0] > 1e6: header['Using subsample of'] = '1,000,000 rows'

    md = '# description of data'
    for key in header:
      md += '\n### ' + key
      md += '  ' + str(header[key])

    self._txt(md + '\n\n#Columns\n\n')

  def _get_column_specified_type(self, col):
    if col.startswith('c_'): return 'categorical'
    if col.startswith('i_'): return 'categorical index'
    if col.startswith('n_'): return 'continous'
    if col.startswith('b_'): return 'binary (0, 1)'
    if col.startswith('d_'): return 'date'
    return 'unknown'

  def _do_target_description(self):
    if self.y is None or len(self.y) == 0: return '## No target variable specified\n\n\n'

    uniques = len(pd.unique(self.y))

    md = '\n\n## Target variable'
    
    md += '\n### Inferred type'
    md += '  ' + self.y_type

    md += '\n### Distinct values'
    md += '  ' + `uniques`

    self._txt(md)

    code = []  
    if self.y_type == 'continuous':
      self._continous_charts(self.y, code)
    elif self.y_type == 'binary' or self.y_type == 'multiclass':
      self._categorical_charts(self.y, code)

    self._code('\n'.join(code))

    self._txt('\n\n\n')

  def _do_column_description(self, col):
    c = self.X[col]    
    uniques = len(pd.unique(c))
    md = '\n\n## ' + col

    md += '\n### Specified type'
    md += '  ' + self._get_column_specified_type(col)

    md += '\n### Inferred type'
    md += '  ' + sklearn.utils.multiclass.type_of_target(c)

    md += '\n### Distinct values'
    md += '  ' + `uniques`
    
    self._txt(md + '\n')
    code = []
    
    '''
    TODO:    
    - If specified type does not match inferred raise issue
    - Charts for both specified and inferred type
    '''

    if col.startswith('c_') or col.startswith('b_') \
        or col.startswith('i_'): self._categorical_charts(c, code)
    elif col.startswith('n_'): self._continous_charts(c, code)

    if self.y is not None:
      self._txt('### Relationship to target variable' + '\n')
      self._relationship(c, self.y, code)

    self._code('\n'.join(code))
    self._txt('\n\n\n')

  def _categorical_charts(self, series, code_arr):
    identifier = 'X["_tmpy"]' if series is self.y else 'X["' + series.name + '"]'
    code_arr.append('fig, axs = plt.subplots(1,1)')
    code_arr.append(identifier + '.value_counts().plot(kind="barh")')        

  def _continous_charts(self, series, code_arr):
    identifier = 'X["_tmpy"]' if series is self.y else 'X["' + series.name + '"]'    
    code_arr.append('fig, axs = plt.subplots(1,2)')
    code_arr.append(identifier + '.hist(bins=20, ax=axs[0])')
    code_arr.append(identifier + '.plot(kind="box", ax=axs[1])')
    if len(pd.unique(series)) < 200: self._categorical_charts(series, code_arr)

  def _relationship(self, a, b, code_arr):
    identifier_a = 'X["' + a.name + '"]'    
    identifier_b = 'X["_tmpy"]' if b is self.y else 'X["' + b.name + '"]'
    identifier_b2 = '_tmpy' if b is self.y else b.name

    type_a = sklearn.utils.multiclass.type_of_target(a)
    type_b = sklearn.utils.multiclass.type_of_target(b)

    if type_a == 'continuous':
      if type_b == 'continuous' :
        code_arr.append('X.plot(kind="scatter", x="' + a.name + '", y="' + b.name + '")')      
        code_arr.append(identifier_a + '.cov(' + identifier_b + ', method="pearson")')      
        code_arr.append(identifier_a + '.cov(' + identifier_b + ', method="spearman")')      
      if type_b == 'binary' or type_b == 'multiclass':
        code_arr.append('fig, axs = plt.subplots(1,2)')         
        code_arr.append(identifier_a + '.plot(kind="box", by="' + identifier_b2 + '", ax=axs[0])')
        code_arr.append('X.plot(kind="scatter", x="' + identifier_b2 + '", y="' + a.name +'", ax=axs[1])')      

    if type_a == 'multiclass':
      if type_b == 'continuous':        
        code_arr.append('fig, axs = plt.subplots(1,1)') 
        code_arr.append('X.plot(kind="scatter", x="' + a.name + '", y="' + b.name +'", ax=axs[0])')      
      if type_b == 'multiclass':
        code_arr.append('fig, axs = plt.subplots(1,1)')         
        code_arr.append('X.boxplot(column="' + a.name + '", by="' + identifier_b2 + '", ax=axs)')

    if type_a == 'binary':
      if type_b == 'continuous' :
        code_arr.append('fig, axs = plt.subplots(1,1)') 
        code_arr.append(identifier_a + '.plot(kind="box", by="' + b.name + '", ax=axs)')
