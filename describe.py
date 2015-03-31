from IPython.nbformat import current as nbf
import sklearn
from misc import *
from pandas_extensions import *

class Describe():
  #############################################################################
  #       Public Interface
  #############################################################################

  def __init__(self, X, opt_y=None):
    self.columns = X.columns
    self.cells = []
    self.nb = nbf.new_notebook()
    self.original_rows = X.shape[0]
    Xy = X.subsample(opt_y, 1e6)  
    self.X = Xy if opt_y is None else Xy[0]
    self.y = None if opt_y is None else Xy[1]
    self.y_type = sklearn.utils.multiclass.type_of_target(self.y) if self.y is not None else 'na'
    self.is_regression = self.y_type == 'continous'

    self._importance_row_limit = 50000
    self._code_lines = []
    self._text_lines = []

    self.importances = self._get_column_importances()    
    if self.is_regression:
      self.f_scores = self._get_column_f_regression_scores()    
    else:
      self.chi2_scores = self._get_column_chi2_classification_scores()    
      self.f_classif_scores = self._get_column_f_classification_scores()    

    dump('_desc_data', (self.X, self.y, self.columns)) 

  def get_cells(self):
    if len(self.cells) > 0: return list(self.cells)    
    self._do_global_imports()    
    self._do_header_markdown()    
    self._do_column_summary_table()
    self._do_target_description()
    self._do_column_details()
    return list(self.cells)
  
  #############################################################################
  #       Header/Misc
  #############################################################################

  def _do_global_imports(self):
    imports = [
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
    self._code('\n'.join(imports), True)

  def _do_header_markdown(self):
    self._txt('# General Description of Data')
    self._name_value('Number of columns', len(self.columns))
    self._name_value('Number of rows', self.X.shape[0])
    if self.X.shape[0] > 1e6: self._name_value('Using subsample of', '1,000,000 rows')
    self._flush_cell()
    

  #############################################################################
  #       Target/Feature Description
  #############################################################################

  def _do_column_summary_table(self):    
    self._txt('#Features Summary Table')
    self._txt('<table>')
    self._do_column_summary_header_row()
    for idx, col in enumerate(self.columns): self._do_column_summary_row(idx, col)            
    self._txt('</table>', True)

  def _do_column_summary_header_row(self):
    cols = ['Column', 'Inferred', 'Specified','Importance', 'F Score']
    if not self.is_regression: cols.append('Chi2')
    self._txt('<tr><th>' + '</th><th>'.join(cols) + '</th></tr>')

  def _do_column_summary_row(self, idx, col):
    cols = [
      col, 
      sklearn.utils.multiclass.type_of_target(self.X[col]), 
      self._get_column_specified_type(col)
      self.importances[idx]
    ]
    if self.is_regression: cols.append(self.f_scores[idx])
    else: cols += [self.f_classif_scores[idx], self.chi2_scores[idx]]
    self._txt('<tr><td>' + '</td><td>'.join(cols) + '</td></tr>')

  def _do_column_details(self):
    self._txt('#Features Details', True)
    for idx, col in enumerate(self.columns):       
      self._do_column_details(col, importances[idx])

  def _do_target_description(self):
    if self.y is None or len(self.y) == 0: 
      self._txt('## No target variable specified\n\n\n', True)
      return

    uniques = len(pd.unique(self.y))

    self._txt('\n\n## Target variable')
    
    self._name_value('Inferred type', self.y_type)
    self._name_value('Distinct values', uniques, True)

    if self.y_type == 'continuous': self._continous_charts(self.y)
    elif self.y_type == 'binary' or self.y_type == 'multiclass': self._categorical_charts(self.y)

    self._txt('\n\n\n', True)


  def _do_column_details(self, col, importance_score):
    c = self.X[col]    
    specified_type = self._get_column_specified_type(col)
    inferred_type = sklearn.utils.multiclass.type_of_target(c)
    sel._txt('\n\n## ' + col)

    sel._name_value('Distinct values', len(pd.unique(c)))
    sel._name_value('Specified type', self._get_column_specified_type(col))
    sel._name_value('Inferred type', sklearn.utils.multiclass.type_of_target(c))

    if self.is_regression:
      sel._name_value('Inferred type', sklearn.utils.multiclass.type_of_target(c))

    if specified_type != inferred_type:
      sel._txt('####<span style="color:#8A0808">Note</span>: Check if specified type is correct as it does ' +\
          'not match inferred type'
    
    
    self._txt(md + '\n', True)
    
    if col.startswith('c_') or col.startswith('b_') \
        or col.startswith('i_'): self._categorical_charts(c)
    elif col.startswith('n_'): self._continous_charts(c)
    self._flush_cell()

    if self.y is not None:
      self._txt('### Relationship to target variable' + '\n', True)
      self._relationship(c, self.y)
      self._flush_cell()

    self._txt('\n\n\n', True)

  def _categorical_charts(self, series):
    identifier = 'X["_tmpy"]' if series is self.y else 'X["' + series.name + '"]'
    self._code('fig, axs = plt.subplots(1,1)')
    self._code(identifier + '.value_counts().plot(kind="barh")')        

  def _continous_charts(self, series):
    identifier = 'X["_tmpy"]' if series is self.y else 'X["' + series.name + '"]'    
    self._code('fig, axs = plt.subplots(1,2)')
    self._code(identifier + '.hist(bins=20, ax=axs[0])')
    self._code(identifier + '.plot(kind="box", ax=axs[1])')
    if len(pd.unique(series)) < 200: self._categorical_charts(series)

  def _relationship(self, a, b):
    identifier_a = 'X["' + a.name + '"]'    
    identifier_b = 'X["_tmpy"]' if b is self.y else 'X["' + b.name + '"]'
    identifier_b2 = '_tmpy' if b is self.y else b.name

    type_a = sklearn.utils.multiclass.type_of_target(a)
    type_b = sklearn.utils.multiclass.type_of_target(b)

    if type_a == 'continuous':
      if type_b == 'continuous' :
        self._code('X.plot(kind="scatter", x="' + a.name + '", y="' + b.name + '")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="pearson")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="spearman")')      
      if type_b == 'binary' or type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,2)')         
        self._code(identifier_a + '.plot(kind="box", by="' + identifier_b2 + '", ax=axs[0])')
        self._code('X.plot(kind="scatter", x="' + identifier_b2 + '", y="' + a.name +'", ax=axs[1])')      

    if type_a == 'multiclass':
      if type_b == 'continuous':        
        self._code('fig, axs = plt.subplots(1,1)') 
        self._code('X.plot(kind="scatter", x="' + a.name + '", y="' + b.name +'", ax=axs[0])')      
      if type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,1)')         
        self._code('X.boxplot(column="' + a.name + '", by="' + identifier_b2 + '", ax=axs)')

    if type_a == 'binary':
      if type_b == 'continuous' :
        self._code('fig, axs = plt.subplots(1,1)') 
        self._code(identifier_a + '.plot(kind="box", by="' + b.name + '", ax=axs)')


  #############################################################################
  #       Utils
  #############################################################################

  def _get_column_importances(self):
    rf = ensemble.RandomForestRegressor(50) if self.is_regression else ensemble.RandomForestClassifier(50)
    rf.fit(X[:self._importance_row_limit], y[:self._importance_row_limit])
    return feature_importances_

  
  def _get_column_f_regression_scores():
    return feature_selection.f_regression(self.X, self.y)

  def _get_column_chi2_classification_scores():
    return feature_selection.chi2(self.X, self.y)

  def _get_column_f_classification_scores():
    return feature_selection.f_classif(self.X, self.y)

  def _get_column_specified_type(self, col):
    if col.startswith('c_'): return 'categorical'
    if col.startswith('i_'): return 'categorical index'
    if col.startswith('n_'): return 'continous'
    if col.startswith('b_'): return 'binary (0, 1)'
    if col.startswith('d_'): return 'date'
    return 'unknown'

  def _name_value(self, name, value, flush=False): 
    sel._txt('\n### ' + name)
    sel._txt(value, flush)

  def _txt(self, markdown, flush=False): 
    self._text_lines.append(str(markdown))
    if flush: self._flush_cell()

  def _code(self, code, flush=False): 
    self._code_lines.append(code)
    if flush: self._flush_cell()

  def _flush_cell(self):
    if self._code_lines.length > 0 and self._text_lines.length > 0:
      raise Exception('only text or only code can be flushed')
    if self._code_lines.length == 0 and self._text_lines.length == 0:
      raise Exception('nothing to flush')

    if self._code_lines.length > 0:
      self.cells.append(nbf.new_code_cell('\n'.join(self._code_lines)))
    else:    
      self.cells.append(nbf.new_text_cell('markdown', '\n'.join(self._text_lines)))    

    self._code_lines = []
    self._text_lines = []
