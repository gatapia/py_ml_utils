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
    self._min_variance_threshold = 0.01
    self._code_lines = []
    self._text_lines = []      
    self.cells = []  

  def show_classifier_performance(self, clf, X, y, use_proba=False):    
    if type(X) is not pd.DataFrame: raise Exception('X must be a pandas.DataFrame')
    if type(y) is not pd.Series: raise Exception('y must be a pandas.Series')
    predictions = X.self_predict_proba(clf, y) if use_proba else \
        X.self_predict(clf, y)
    self.show_prediction_comparison(predictions, y)        

  def show_prediction_comparison(self, y_true, y_pred):        
    if type(y_true) is not pd.Series: raise Exception('y_true must be a pandas.Series')
    if type(y_pred) is not pd.Series and type(y_pred) is not pd.DataFrame: raise Exception('y_pred must be a pandas.Series or pandas.DataFrame')
    self._create_notebook(self.get_prediction_comparison_cells(y_true, y_pred), True)    

  def get_prediction_comparison_cells(self, y_true, y_pred): 
    if len(y_true) != len(y_pred): raise Exception('y_true and y_pred are not compatible shapes (must be same rows and same columns)')    

    self.cells = []
    start('get_prediction_comparison_cells')
    dataset_name = '_get_prediction_comparison_cells_data'
    dump(dataset_name, (y_true, y_pred)) 

    self.y_true = y_true.copy()
    self.y_true.name = 'y_true'
    self.y_pred = y_pred.copy()
    self.y_pred.name = 'y_pred'
    self._do_global_imports(dataset_name, ['y_true', 'y_pred'])
    self._do_target_description(self.y_true, False, 'y_true')
    self._do_target_description(self.y_pred, False, 'y_pred')
    self._do_target_comparison_charts()    
    stop('get_prediction_comparison_cells')    
    return list(self.cells)


  def show_dataset(self, X, opt_y=None):    
    if type(X) is not pd.DataFrame: raise Exception('X must be a pandas.DataFrame')
    if opt_y is not None and type(opt_y) is not pd.Series: raise Exception('opt_y must be a pandas.Series')
    '''
    TODO: If we are evaluating results over years or location then
      we should plot data by this same fold. Maybe ensure also that
      each year for each attribute correlates nicely
    '''        
    self.original_rows = X.shape[0]
    Xy = X.subsample(opt_y, 5e5)  
    self.X = Xy.copy() if opt_y is None else Xy[0].copy()
    self.X_no_nan = self.X.copy().missing('na', 0)
    if opt_y is not None:
      self.y = Xy[1].copy()
      self.y.name = 'y'
    else: self.y = None
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
    # TODO: Do table of all columns x column with correlation 
    #   values for all relationships.
    # TODO: Correlation matrix of all variables
    self._do_column_summary_charts()
    if self.y is not None: self._do_target_description(self.y, True)
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

  def _do_target_description(self, target_var, show_pca_graphs, target_var_name='Target Variable'):
    if len(target_var.shape) == 2 and target_var.shape[1] > 1:
      for col_idx in range(target_var.shape[1]):
        self._do_target_description_impl(target_var.ix[:, col_idx], show_pca_graphs, target_var_name + ' - ' + `col_idx`, target_var.name + '.ix[:,' + `col_idx` + ']')  
    else: 
      self._do_target_description_impl(target_var, show_pca_graphs, target_var_name, target_var.name)

  def _do_target_description_impl(self, target_var, show_pca_graphs, target_var_name, target_var_identifier):
    target_type = self._type(target_var)
    uniques = pd.unique(target_var)

    self._txt('\n\n<hr/>\n## ' + target_var_name)
    
    self._name_value('Inferred type', target_type)
    self._name_value('Distinct values', len(uniques), True)

    if target_type == 'continuous': self._continuous_charts(target_var_identifier)
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
    self._txt('<hr/>\n# Prediction Comparison', True)
    y_true_type = self._type(self.y_true)
    y_pred_type = self._type(self.y_pred)
    y_true_classification = y_true_type == 'multiclass' or y_true_type == 'binary'
    y_pred_classification = y_pred_type == 'multiclass' or y_pred_type == 'binary'
    if y_true_type == 'continuous' and y_pred_type == 'continuous':
      self._name_value('Pearson Cov:', self.y_true.cov(y_pred, method="pearson"))      
      self._name_value('Spearman Cov:', self.y_true.cov(y_pred, method="spearman"), True)      
      self._code('plt.scatter(y_true, y_pred, c=COLOURS, alpha=0.5).set_title("y_true vs y_pred")')
      self._code('bland_altman_plot(y_true, y_pred)')
    elif y_true_classification and y_pred_type == 'continuous-multioutput': 
      self._txt('##Confusion Matrices<br/>', True)
      for col_idx in range(self.y_pred.shape[1]):
        self._code(
'''
matrix = metrics.confusion_matrix(y_true == %d, y_pred.ix[:,%d] > 0.5)
print 'Confusion Matrix Target Variable - %d'\n
print_confusion_matrix(matrix, ['True', 'False'])
plot_confusion_matrix(matrix)
plt.show()
''' % (col_idx, col_idx, col_idx))
      self._flush_cell()
      self._txt('##Reliability Diagrams<br/>', True)
      self._code('f, axs = plt.subplots(%d, 2, figsize=(12, %d))\naxs = axs.reshape(-1)' % 
        (int(math.ceil(self.y_pred.shape[1] / 2.0)), self.y_pred.shape[1]*4))
      for col_idx in range(self.y_pred.shape[1]):        
        self._code(
'''
y_score_bin_mean, empirical_prob_pos = reliability_curve((y_true==%d).values, y_pred.ix[:,%d].values, 200)
axs[%d].scatter(y_score_bin_mean, empirical_prob_pos)
axs[%d].set_title('Reliability Diagram - %d')
axs[%d].set_xlabel('Predicted Probability')
_ = axs[%d].set_ylabel('Empirical Probability')
''' % (col_idx, col_idx, col_idx, col_idx, col_idx, col_idx, col_idx))

    elif y_true_classification and y_pred_type == 'continuous': 
      # using predict proba      
      self._code(
'''
fig, axs = plt.subplots(1,2, figsize=(12, 8))
matrix = metrics.confusion_matrix(y_true, y_pred > 0.5)
_ = plot_confusion_matrix(matrix)
y_score_bin_mean, empirical_prob_pos = reliability_curve(y_true.values, y_pred.values, 200)
axs[0].scatter(y_score_bin_mean, empirical_prob_pos)
axs[0].set_title('Reliability Diagram')
axs[0].set_xlabel('Predicted Probability')
_ = axs[0].set_ylabel('Empirical Probability')
print_confusion_matrix(matrix, ['True', 'False'])
''')
    elif y_true_classification and y_pred_classification:
      self._code(
'''
fig, axs = plt.subplots(1,2, figsize=(12, 8))
matrix = metrics.confusion_matrix(y_true, y_pred)
_ = plot_confusion_matrix(matrix)
y_score_bin_mean, empirical_prob_pos = reliability_curve(y_true.values, y_pred.values, 200)
axs[0].scatter(y_score_bin_mean, empirical_prob_pos)
axs[0].set_title('Reliability Diagram')
axs[0].set_xlabel('Predicted Probability')
_ = axs[0].set_ylabel('Empirical Probability')
print_confusion_matrix(matrix, ['True', 'False'])
''')
    else: raise Exception('Type combination not supported: y_true_type: ' + y_true_type + ' y_pred_type: ' + y_pred_type)
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
    variance = col_details[-1]
    if variance < self._min_variance_threshold: cols.append('<b>' + self._pretty(col_details[-1]) + '</b>')
    else: cols.append(col_details[-1])
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
    elif col_name.startswith('n_'): self._continuous_charts('X[' + `c.name` + ']')
    self._flush_cell()

    if self.y is not None:
      self._txt('### Relationship to target variable' + '\n', True)
      self._relationship(c, self.y)
      self._flush_cell()

    self._txt('\n\n\n', True)

  def _categorical_charts(self, s):
    identifier = 'X["' + s.name + '"]' if 'X' in globals() and s.name in X.columns else s.name
    self._code('fig, axs = plt.subplots(1,1)')
    self._code('_ = ' + identifier + '.value_counts().plot(kind="barh")')        
    self._code('axs.set_ylabel("Value")')
    self._code('axs.set_xlabel("Count")')
    self._code('_ = axs.set_title("Field Values")')

  def _continuous_charts(self, identifier):
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
    identifier_a = 'X["' + a.name + '"]' if 'X' in globals() and a.name in X.columns else a.name
    identifier_b = 'X["' + b.name + '"]' if 'X' in globals() and b.name in X.columns else b.name

    type_a = self._type(a)
    type_b = self._type(b)

    if type_a == 'continuous':
      if type_b == 'continuous' :
        self._code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name + '")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="pearson")')      
        self._code(identifier_a + '.cov(' + identifier_b + ', method="spearman")')      
      
      if type_b == 'binary' or type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,2)')         
        self._code('_ = X[["' + a.name + '", "' + b.name + '"]].boxplot(by="' + b.name + '", ax=axs[0])')
        self._code('_ = X.plot(kind="scatter", x="' + b.name + '", y="' + a.name +'", ax=axs[1])')      
        self._code('_ = X.hist(column="' + a.name + '", by="y", figsize=(12, 12))')


    if type_a == 'multiclass':
      if type_b == 'continuous':        
        self._code('fig, axs = plt.subplots(1,1)') 
        self._code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name +'", ax=axs[0])')      
      
      if type_b == 'multiclass':
        self._code('fig, axs = plt.subplots(1,1)')         
        self._code('X.boxplot(column="' + a.name + '", by="' + b.name + '", ax=axs)')

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

  def _create_notebook(self, cells, start_notebook):
    nb = nbf.new_notebook()
    nb.cells = cells
    with open('dataset_description.ipynb', 'w') as f: 
      f.write(nbf.writes(nb)) 

    if start_notebook:call(['ipython', 'notebook', 'dataset_description.ipynb'])
