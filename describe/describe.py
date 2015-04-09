import sklearn
from misc import *
from pandas_extensions import *
from sklearn import ensemble, feature_selection
from IPython.nbformat import v4 as nbf
from IPython import html
from subprocess import call


class Describe():
  #############################################################################
  #       Public Interface
  #############################################################################

  def __init__(self, X, opt_y=None):
    self.columns = X.columns
    self.cells = []
    self.original_rows = X.shape[0]
    Xy = X.subsample(opt_y, 5e5)  
    self.X = Xy if opt_y is None else Xy[0]
    self.X_no_nan = self.X.copy().missing('na', 0)
    self.y = None if opt_y is None else Xy[1]
    self.y_type = sklearn.utils.multiclass.type_of_target(self.y) if self.y is not None else 'na'
    self.is_regression = self.y_type == 'continuous'

    self._importance_row_limit = 50000
    self._code_lines = []
    self._text_lines = []    

    dump('_desc_data', (self.X, self.y)) 

  def show(self):    
    nb = nbf.new_notebook()
    nb.cells = self.get_cells()
    with open('dataset_description.ipynb', 'w') as f: 
      f.write(nbf.writes(nb)) 

    call(['ipython', 'notebook', 'dataset_description.ipynb'])


  def get_cells(self):
    if len(self.cells) > 0: return list(self.cells)    
    start('get_cells')
    self._do_global_imports()    
    self._do_header_markdown()    
    self._intialise_feature_scores()
    self._do_column_summary_table()
    self._do_column_summary_charts()
    self._do_target_description()
    self._do_all_columns_details()
    stop('done get_cells')
    return list(self.cells)
  
  #############################################################################
  #       Header/Misc
  #############################################################################

  def _do_global_imports(self):    
    imports = [
      '%matplotlib inline',
      'import numpy as np',
      'import pandas as pd',      
      'import matplotlib.pyplot as plt',
      'import pylab',
      'from sys import path',
      'path.append("utils")',
      'from misc import *',
      'from pandas_extensions import *',
      'from sklearn import *',
      '\npd.set_option("display.notebook_repr_html", False)',
      'pd.set_option("display.max_columns", 20)',
      'pd.set_option("display.max_rows", 25)',
      'pylab.rcParams[\'figure.figsize\'] = (10.0, 6.0)',
      'X, y = load("_desc_data")',
      'X["_tmpy"] = y'
    ]
    self._code('\n'.join(imports), True)

  def _do_header_markdown(self):
    self._txt('<hr/>\n# General Description of Data')
    self._name_value('Number of columns', len(self.columns))
    self._name_value('Number of rows', self.X.shape[0])
    if self.X.shape[0] > 5e5: self._name_value('Using subsample of', '50,000 rows')
    self._flush_cell()

  def _intialise_feature_scores(self):
    self.importances = self._get_column_importances()    
    if self.is_regression:
      self.f_scores = self._get_column_f_regression_scores()    
      self.col_details = zip(self.columns, self.importances, self.f_scores)
    else:
      self.f_classif_scores = self._get_column_f_classification_scores()    
      self.chi2_scores = self._get_column_chi2_classification_scores()          
      self.col_details = zip(self.columns, self.importances, self.f_classif_scores, self.chi2_scores)

    self.col_details = sorted(self.col_details, key=lambda d: d[1], reverse=True)
      
  #############################################################################
  #       Target Variabe
  #############################################################################

  def _do_target_description(self):
    if self.y is None or len(self.y) == 0: 
      self._txt('<hr/>\n## No target variable specified\n\n\n', True)
      return

    uniques = pd.unique(self.y)

    self._txt('\n\n<hr/>\n## Target variable')
    
    self._name_value('Inferred type', self.y_type)
    self._name_value('Distinct values', len(uniques), True)

    if self.y_type == 'continuous': self._continuous_charts(self.y)
    elif self.y_type == 'binary' or self.y_type == 'multiclass': 
      self._categorical_charts(self.y)
      
      self._code([
        'X2 = X.copy().missing("na", 0)',
        'X2 = pd.DataFrame(decomposition.PCA(2).fit_transform(X2), columns=["A", "B"])',
        'X2["y"] = y',
        'markers=[".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]',
        'colours=["Blue","Green","Red","Cyan","Magenta","Yellow","Black","White"]',
        'ax = None',
        'for idx, v in enumerate(pd.unique(y)):',
        '  col=colours[idx % len(colours)]',
        '  mkr=markers[idx % len(markers)]',
        '  X3 = X2[X2.y == v]'
        '  if ax is None: ax = X3.plot(kind="scatter", x="A", y="B", label=v, alpha=0.2, marker=mkr, color=col)',
        '  else: X3.plot(kind="scatter", x="A", y="B", label=v, ax=ax, alpha=0.2, marker=mkr, color=col)',
        '_ = ax.set_title("2D PCA - Target Variable Distribution")'
      ])

    self._flush_cell()
    self._txt('\n\n\n', True)

  #############################################################################
  #       Columns Summary
  #############################################################################

  def _do_column_summary_table(self):    
    self._txt('<hr/>\n#Features Summary Table')
    self._txt('<table>')
    self._do_column_summary_header_row()
    for idx, col in enumerate(self.col_details): self._do_column_summary_row(idx, col)            
    self._txt('</table>', True)

  def _do_column_summary_header_row(self):
    cols = ['Column', 'Inferred', 'Specified', 'RF Imp Rank', 'RF Importance V', 'F Score']
    if not self.is_regression: cols.append('Chi2')
    self._txt('<tr><th>' + '</th><th>'.join(cols) + '</th></tr>')

  def _do_column_summary_row(self, idx, col):
    col_name = col[0]
    inferred = sklearn.utils.multiclass.type_of_target(self.X[col_name])
    specified = self._get_column_specified_type(col_name)
    inf_same_specified = inferred == specified
    cols = [
      col_name, 
      inferred, 
      specified if inf_same_specified else self._warn(specified),
      idx + 1,
      col[1]
    ]
    if self.is_regression: cols.append(col[2])
    else: cols += [col[2], col[3]]
    self._txt('<tr><td>' + '</td><td>'.join(map(self._pretty, cols)) + '</td></tr>')

  def _do_column_summary_charts(self):
    self._txt('<hr/>\n#Top 5 Column Interaction', True)
    top_5 = map(lambda c: c[0], self.col_details[:5])
    self._code('top_5 = ' + str(top_5))
    self._code('_ = pd.tools.plotting.scatter_matrix(X[top_5], alpha=0.2, figsize=(12, 12), diagonal="kde")', True)


  #############################################################################
  #       Columns Details
  #############################################################################

  def _do_all_columns_details(self):
    self._txt('<hr/>\n#Features Details', True)
    for idx, col in enumerate(self.col_details):       
      self._do_column_details(idx, col)  


  def _do_column_details(self, idx, col):    
    col_name = col[0]
    c = self.X[col_name]    
    specified_type = self._get_column_specified_type(col_name)
    inferred_type = sklearn.utils.multiclass.type_of_target(c)

    self._txt('\n\n<hr/>\n## ' + col_name)

    self._name_value('Distinct values', len(pd.unique(c)))
    self._name_value('Specified type', self._get_column_specified_type(col_name))
    self._name_value('Inferred type', sklearn.utils.multiclass.type_of_target(c))
    self._name_value('RF Importance Rank', idx + 1)
    self._name_value('RF Importance Score', col[1])

    if self.is_regression: self._name_value('F-Score', col[2])
    else:
      self._name_value('F-Score', col[2])
      self._name_value('Chi2 Score', col[3])

    if specified_type != inferred_type:
      self._txt(self._warn('Note:') + ' Check if specified type is correct as it does not match inferred type')
    
    self._txt('\n', True)
    
    if col_name.startswith('c_') or col_name.startswith('b_') \
        or col_name.startswith('i_'): self._categorical_charts(c)
    elif col_name.startswith('n_'): self._continuous_charts(c)
    self._flush_cell()

    if self.y is not None:
      self._txt('### Relationship to target variable' + '\n', True)
      self._relationship(c, self.y)
      self._flush_cell()

    self._txt('\n\n\n', True)

  def _categorical_charts(self, series):
    identifier = '_ = X["_tmpy"]' if series is self.y else '_ = X["' + series.name + '"]'
    self._code('fig, axs = plt.subplots(1,1)')
    self._code(identifier + '.value_counts().plot(kind="barh")')        
    self._code('axs.set_ylabel("Value")')
    self._code('axs.set_xlabel("Count")')
    self._code('_ = axs.set_title("Field Values")')

  def _continuous_charts(self, series):
    identifier = '_ = X["_tmpy"]' if series is self.y else '_ = X["' + series.name + '"]'    
    self._code('fig, axs = plt.subplots(1,2)')
    self._code(identifier + '.hist(bins=20, ax=axs[0])')
    self._code(identifier + '.plot(kind="box", ax=axs[1])')

  def _relationship(self, a, b):
    identifier_a = 'X["' + a.name + '"]'    
    identifier_b = 'X["_tmpy"]' if b is self.y else 'X["' + b.name + '"]'
    identifier_b2 = '_tmpy' if b is self.y else b.name

    type_a = sklearn.utils.multiclass.type_of_target(a)
    type_b = sklearn.utils.multiclass.type_of_target(b)

    if type_a == 'continuous':
      if type_b == 'continuous' :
        self._code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name + '")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="pearson")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="spearman")')      
      
      if type_b == 'binary' or type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,2)')         
        self._code('_ = X[["' + a.name + '", "' + identifier_b2 + '"]].boxplot(by="' + identifier_b2 + '", ax=axs[0])')
        self._code('_ = X.plot(kind="scatter", x="' + identifier_b2 + '", y="' + a.name +'", ax=axs[1])')      
        self._code('_ = X.hist(column="' + a.name + '", by="_tmpy", figsize=(12, 12))')


    if type_a == 'multiclass':
      if type_b == 'continuous':        
        self._code('fig, axs = plt.subplots(1,1)') 
        self._code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name +'", ax=axs[0])')      
      
      if type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,1)')         
        self._code('X.boxplot(column="' + a.name + '", by="' + identifier_b2 + '", ax=axs)')

    if type_a == 'binary':
      if type_b == 'continuous' :
        self._code('fig, axs = plt.subplots(1,1)') 
        self._code('_ = ' + identifier_a + '.plot(kind="box", by="' + b.name + '", ax=axs)')


  #############################################################################
  #       Utils
  #############################################################################

  def _bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,        color='gray', linestyle='--')
    plt.axhline(md + 2*sd, color='gray', linestyle='--')
    plt.axhline(md - 2*sd, color='gray', linestyle='--')

  def _get_column_importances(self):
    start('_get_column_importances')
    rf = ensemble.RandomForestRegressor(50) if self.is_regression else ensemble.RandomForestClassifier(50)    
    rf.fit(self.X_no_nan[:self._importance_row_limit], self.y[:self._importance_row_limit])
    stop('done _get_column_importances, num feats: ' + `len(rf.feature_importances_)`)
    return rf.feature_importances_
  
  def _get_column_f_regression_scores(self):
    start('_get_column_f_regression_scores')
    scores = feature_selection.f_regression(self.X_no_nan, self.y)[0]
    stop('_get_column_f_regression_scores')
    return scores

  def _get_column_chi2_classification_scores(self):
    start('_get_column_chi2_classification_scores')
    scores = feature_selection.chi2(self.X_no_nan, self.y)[0]
    stop('_get_column_chi2_classification_scores')
    return scores

  def _get_column_f_classification_scores(self):
    start('_get_column_f_classification_scores')
    scores = feature_selection.f_classif(self.X_no_nan, self.y)[0]
    stop('_get_column_f_classification_scores')
    return scores

  def _get_column_specified_type(self, col_name):
    if col_name.startswith('c_'): return 'categorical'
    if col_name.startswith('i_'): return 'categorical index'
    if col_name.startswith('n_'): return 'continuous'
    if col_name.startswith('b_'): return 'binary (0, 1)'
    if col_name.startswith('d_'): return 'date'
    return 'unknown'

  def _name_value(self, name, value, flush=False): 
    self._txt('\n<b>' + name + '</b>: ' + self._pretty(value), flush)

  def _txt(self, text, flush=False): 
    if type(text) is list: self._text_lines += text
    else: self._text_lines.append(self._pretty(text))
    if flush: self._flush_cell()

  def _code(self, code, flush=False): 
    if type(code) is list: self._code_lines += code
    else: self._code_lines.append(code)
    if flush: self._flush_cell()

  def _pretty(self, value):
    if type(value) is float or \
      type(value) is np.float or \
      type(value) is np.float32 or \
      type(value) is np.float64: return '%.4f' % value
    else: return str(value)

  def _warn(self, value):
    return '<span style="color:#8A0808"><b>' + self._pretty(value) + '</b></span>'

  def _flush_cell(self):
    if len(self._code_lines) > 0 and len(self._text_lines) > 0:
      raise Exception('only text or only code can be flushed')
    if len(self._code_lines) == 0 and len(self._text_lines) == 0:
      raise Exception('nothing to flush')

    if len(self._code_lines) > 0:
      self.cells.append(nbf.new_code_cell('\n'.join(self._code_lines)))
    else:    
      self.cells.append(nbf.new_markdown_cell('\n'.join(self._text_lines)))    

    self._code_lines = []
    self._text_lines = []
