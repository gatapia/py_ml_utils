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

  def __init__(self):    
    self._importance_row_limit = 50000
    self._code_lines = []
    self._text_lines = []      
    self.cells = []  

  def show_classifier_performance(self, clf, X, y, use_proba=False):    
    predictions = X.self_predict_proba(clf, y) if use_proba else \
        X.self_predict(clf, y)
    self.show_prediction_comparison(predictions, y)        

  def show_prediction_comparison(self, y_true, y_pred):    
    if y_true.shape != y_pred.shape:
      raise Exception('y_true and y_pred are not compatible shapes (must be same rows and same columns)')
    
    self._create_notebook(self.get_prediction_comparison_cells(y_true, y_pred), False)    

  def get_prediction_comparison_cells(self, y_true, y_pred):    
    self.cells = []
    start('get_prediction_comparison_cells')
    dataset_name = '_get_prediction_comparison_cells_data'
    dump(dataset_name, (y_true, y_pred)) 

    self.y_true = y_true
    self.y_pred = y_pred
    self._do_global_imports(dataset_name, ['y_true', 'y_pred'])
    self._do_target_description(self.y_true, False)
    self._do_target_description(self.y_pred, False)
    self._do_target_comparison_charts()    
    stop('get_prediction_comparison_cells')    


  def show_dataset(self, X, opt_y=None):    
    '''
    TODO: If we are evaluating results over years or location then
      we should plot data by this same fold. Maybe ensure also that
      each year for each attribute correlates nicely
    '''        
    self.original_rows = X.shape[0]
    Xy = X.subsample(opt_y, 5e5)  
    self.X = Xy if opt_y is None else Xy[0]
    self.X_no_nan = self.X.copy().missing('na', 0)
    self.y = None if opt_y is None else Xy[1]    
    self.is_regression = self._type(self.y) == 'continuous'

    self._create_notebook(self.get_dataset_cells(), True)    


  def get_dataset_cells(self):
    self.cells = []
    start('get_dataset_cells')
    dataset_name = '_get_dataset_cells_data'
    data_x = self.X.copy()
    data_x['y'] = self.y
    dump(dataset_name, (data_x, self.y)) 

    self._do_global_imports(dataset_name, ['X', 'y'])    
    self._do_header_markdown()    
    self._intialise_feature_scores()
    self._do_column_summary_table()
    # Do table of all columns x column with correlation values for
    #  all relationships
    self._do_column_summary_charts()
    if self.y is not None: self._do_target_description('y', True)
    self._do_all_columns_details()
    stop('done get_dataset_cells')
    return list(self.cells)
  
  #############################################################################
  #       Header/Misc
  #############################################################################

  def _do_global_imports(self, dataset_name, variables):    
    imports = [
      '%matplotlib inline',
      'import numpy as np',
      'import pandas as pd',      
      'import matplotlib.pyplot as plt',
      'import pylab',
      'from sys import path',
      'path.append("utils")',
      'from misc import *',
      'from graphs import *',
      'from pandas_extensions import *',
      'from sklearn import *',
      '\npd.set_option("display.notebook_repr_html", False)',
      'pd.set_option("display.max_columns", 20)',
      'pd.set_option("display.max_rows", 25)',
      'pylab.rcParams[\'figure.figsize\'] = (10.0, 6.0)',
      ','.join(variables) + ' = load("' + dataset_name + '")',
      'MARKERS=[".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]',
      'COLOURS=["Blue","Green","Red","Cyan","Magenta","Yellow","Black","White"]',
    ]
    self._code(imports, True)

  def _do_header_markdown(self):
    self._txt('<hr/>\n# General Description of Data')
    self._name_value('Number of columns', len(self.X.columns))
    self._name_value('Number of rows', self.X.shape[0])
    if self.X.shape[0] > 5e5: self._name_value('Using subsample of', '50,000 rows')
    self._flush_cell()

  def _intialise_feature_scores(self):
    self.importances = self._get_column_importances()    
    self.variances = self.X.var().values
    if self.is_regression:
      self.f_scores = self._get_column_f_regression_scores()    
      self.col_details = zip(self.X.columns, self.importances, self.f_scores, self.variances)
    else:
      self.f_classif_scores = self._get_column_f_classification_scores()    
      self.chi2_scores = self._get_column_chi2_classification_scores()          
      self.col_details = zip(self.X.columns, self.importances, self.f_classif_scores, self.chi2_scores, self.variances)

    self.col_details = sorted(self.col_details, key=lambda d: d[1], reverse=True)
      
  #############################################################################
  #       Target Variable
  #############################################################################

  def _do_target_description(self, target_var, show_pca_graphs):
    target_type = self._type(target_var)
    uniques = pd.unique(target_var)

    self._txt('\n\n<hr/>\n## Target variable')
    
    self._name_value('Inferred type', target_type)
    self._name_value('Distinct values', len(uniques), True)

    if target_type == 'continuous': self._continuous_charts(target_var)
    elif target_type == 'binary' or target_type == 'multiclass': 
      self._categorical_charts(target_var)
      if show_pca_graphs:
        self._code([
          'X2 = X.copy().missing("na", 0)',        
          'X2 = pd.DataFrame(decomposition.PCA(2).fit_transform(X2), columns=["A", "B"])',        
          'X2["y"] = y',
          'ax = None',
          'for idx, v in enumerate(pd.unique(y)):',
          '  col=COLOURS[idx % len(COLOURS)]',
          '  mkr=MARKERS[idx % len(MARKERS)]',
          '  X3 = X2[X2.y == v]',
          '  if ax is None: ax = X3.plot(kind="scatter", x="A", y="B", label=v, alpha=0.2, marker=mkr, color=col)',
          '  else: X3.plot(kind="scatter", x="A", y="B", label=v, ax=ax, alpha=0.2, marker=mkr, color=col)',
          '_ = ax.set_title("2D PCA - Target Variable Distribution")'
        ])

    self._flush_cell()
    self._txt('\n\n\n', True)


  def _do_target_comparison_charts(self):
    y_true_type = self._type(self.y_true)
    y_pred_type = self._type(self.y_pred)
    if y_true_type == 'continuous' and y_pred_type == 'continuous':
      self._name_value('Pearson Cov:', self.y_true.cov(y_pred, method="pearson"))      
      self._name_value('Spearman Cov:', self.y_true.cov(y_pred, method="spearman"), True)      
      self._code('plt.scatter(y_true, y_pred, c=COLOURS, alpha=0.5).set_title("y_true vs y_pred")')
      self._code('bland_altman_plot(y_true, y_pred)')
    if y_true_type == 'multiclass' and y_pred_type == 'continuous': 
      # using predict proba      
      self._code('matrix = metrics.confusion_matrix(y_true, y_pred > 0.5)')
      self._code('plot_confusion_matrix(matrix)')
      self._code('reliability_curve(y_true, y_pred)')
    elif y_true_type == 'multiclass' and y_pred_type == 'multiclass':
      self._code('matrix = metrics.confusion_matrix(y_true, y_pred)')
      self._code('plot_confusion_matrix(matrix)')
      self._code('reliability_curve(y_true, y_pred)')

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
    cols.append('Var')
    self._txt('<tr><th>' + '</th><th>'.join(cols) + '</th></tr>')

  def _do_column_summary_row(self, idx, col_details):
    col_name = col_details[0]
    inferred = self._type(self.X[col_name])
    specified = self.  _get_column_specified_type(col_name)
    inf_same_specified = inferred == specified
    cols = [
      col_name, 
      inferred, 
      specified if inf_same_specified else self._warn(specified),
      idx + 1,
      col_details[1]      
    ]
    if self.is_regression: cols.append(col_details[2])
    else: cols += [col_details[2], col_details[3]]
    cols.append(col_details[-1])
    self._txt('<tr><td>' + '</td><td>'.join(map(self._pretty, cols)) + '</td></tr>')

  def _do_column_summary_charts(self):
    self._txt('<hr/>\n#Top 5 Column Interaction', True)
    top_5 = map(lambda c: c[0], self.col_details[:5])
    self._code('top_5 = ' + str(top_5))
    self._code('_ = pd.tools.plotting.scatter_matrix(X[top_5], alpha=0.2, figsize=(12, 12), diagonal="kde")', True)

    self._txt('<hr/>\n#PCA Explained Variable Ratios', True)
    self._code(['X2 = X.copy().missing("na", 0)',
        'ratios = decomposition.PCA(X2.shape[1]).fit(X2).explained_variance_ratio_',
        'df = pd.DataFrame({"variance": ratios})',
        '_ = df.plot(kind="bar", figsize=(12, 10))'], True)


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
    specifiedself._type = self.  _get_column_specified_type(col_name)
    inferredself._type = self._type(c)

    self._txt('\n\n<hr/>\n## ' + col_name)

    self._name_value('Distinct values', len(pd.unique(c)))
    self._name_value('Specified type', self.  _get_column_specified_type(col_name))
    self._name_value('Inferred type', self._type(c))
    self._name_value('RF Importance Rank', idx + 1)
    self._name_value('RF Importance Score', col[1])

    if self.is_regression: self._name_value('F-Score', col[2])
    else:
      self._name_value('F-Score', col[2])
      self._name_value('Chi2 Score', col[3])
    self._name_value('Variance', col[-1])

    if specifiedself._type != inferredself._type:
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
    identifier = '_ = X["y"]' if series is self.y else '_ = X["' + series.name + '"]'
    self._code('fig, axs = plt.subplots(1,1)')
    self._code(identifier + '.value_counts().plot(kind="barh")')        
    self._code('axs.set_ylabel("Value")')
    self._code('axs.set_xlabel("Count")')
    self._code('_ = axs.set_title("Field Values")')

  def _continuous_charts(self, series):
    identifier = 'X["y"]' if series is self.y else 'X["' + series.name + '"]'
    self._code([
      'fig, axs = plt.subplots(1,3, figsize=(12, 8))',
      identifier + '.hist(bins=30, ax=axs[0]).set_title("Distribution")',
      'bc = scipy.stats.boxcox(' + identifier +')',
      'df = pd.DataFrame(bc[0])',
      'df.hist(bins=30, ax=axs[1])',
      'axs[1].set_title("Boxcox Distribution l=%.3f" % bc[1])',
      '_ = ' + identifier + '.plot(kind="box", ax=axs[2]).set_title("Boxplot")'
    ])

  def _relationship(self, a, b):
    identifier_a = 'X["' + a.name + '"]'    
    identifier_b = 'X["y"]' if b is self.y else 'X["' + b.name + '"]'
    identifier_b2 = 'y' if b is self.y else b.name

    type_a = self._type(a)
    type_b = self._type(b)

    if type_a == 'continuous':
      if type_b == 'continuous' :
        self._code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name + '")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="pearson")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="spearman")')      
      
      if type_b == 'binary' or type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,2)')         
        self._code('_ = X[["' + a.name + '", "' + identifier_b2 + '"]].boxplot(by="' + identifier_b2 + '", ax=axs[0])')
        self._code('_ = X.plot(kind="scatter", x="' + identifier_b2 + '", y="' + a.name +'", ax=axs[1])')      
        self._code('_ = X.hist(column="' + a.name + '", by="y", figsize=(12, 12))')


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

  def   _get_column_specified_type(self, col_name):
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

  def _type(self, data):
    if data is None: return None
    return sklearn.utils.multiclass.type_of_target(data)

  def _create_notebook(cells, start_notebook):
    nb = nbf.new_notebook()
    nb.cells = cells
    with open('dataset_description.ipynb', 'w') as f: 
      f.write(nbf.writes(nb)) 

    if start_notebook:call(['ipython', 'notebook', 'dataset_description.ipynb'])
