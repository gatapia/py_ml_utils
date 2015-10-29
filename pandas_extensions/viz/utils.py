import sklearn
from .. import misc
from IPython.nbformat import v4 as nbf
from subprocess import call
import pandas as pd, numpy as np

class Utils(object):

  def __init__(self):
    self._code_lines = []
    self._text_lines = []      
    self.cells = []      

  def do_global_imports(self, dataset_name, variables):    
    imports = [
      '%matplotlib inline',
      'import qgrid, numpy as np, pandas as pd, matplotlib.pyplot as plt, pylab',
      'from sys import path',
      'path.append("ml")',
      'from misc import *; from graphs import *; from ml.pandas_extensions import *;from sklearn import *',
      '\npd.set_option("display.notebook_repr_html", False)',
      'pd.set_option("display.max_columns", 20)',
      'pd.set_option("display.max_rows", 25)',
      'qgrid.nbinstall(overwrite=True)',
      'qgrid.set_defaults(remote_js=True, precision=4)',
      'pylab.rcParams[\'figure.figsize\'] = (10.0, 6.0)',      
      ','.join(variables) + ' = load("' + dataset_name + '")',
      'MARKERS=[".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]',
      'COLOURS=["Blue","Green","Red","Cyan","Magenta","Yellow","Black","White"]',
    ]
    self.code(imports, True)

  #############################################################################
  #       Target Variable
  #############################################################################

  def do_target_description(self, target_var, show_pca_graphs, target_var_name='Target Variable'):
    if len(target_var.shape) == 2 and target_var.shape[1] > 1:
      for col_idx in range(target_var.shape[1]):
        self.do_target_description_impl(target_var.ix[:, col_idx], show_pca_graphs, target_var_name + ' - ' + `col_idx`, target_var.name + '.ix[:,' + `col_idx` + ']')  
    else: 
      self.do_target_description_impl(target_var, show_pca_graphs, target_var_name, target_var.name)

  def do_target_description_impl(self, target_var, show_pca_graphs, target_var_name, target_var_identifier):
    target_type = self.type(target_var)
    uniques = pd.unique(target_var)

    self.txt('\n\n<hr/>\n## ' + target_var_name)
    
    self.name_value('Inferred type', target_type)
    self.name_value('Distinct values', len(uniques), True)

    if target_type == 'continuous': self.continuous_charts(target_var_identifier)
    elif target_type == 'binary' or target_type == 'multiclass': 
      self.categorical_charts('X["y"]')
      if show_pca_graphs:
        self.code([
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

    self.flush_cell()
    self.txt('\n\n\n', True)


  def do_target_comparison_charts(self, y1, y2):
    self.txt('<hr/>\n# Prediction Comparison', True)
    y_true_type = self.type(y1)
    y_pred_type = self.type(y2)
    y_true_classification = y_true_type == 'multiclass' or y_true_type == 'binary'
    y_pred_classification = y_pred_type == 'multiclass' or y_pred_type == 'binary'
    if y_true_type == 'continuous' and y_pred_type == 'continuous':
      self.name_value('Covariance:', y1.cov(y2))      
      self.code('plt.scatter(y_true, y_pred, c=COLOURS, alpha=0.5).set_title("y_true vs y_pred")')
      self.code('bland_altman_plot(y_true, y_pred)')
    elif y_true_classification and y_pred_type == 'continuous-multioutput': 
      self.txt('##Confusion Matrices<br/>', True)
      for col_idx in range(y2.shape[1]):
        self.code(
'''
matrix = metrics.confusion_matrix(y_true == %d, y_pred.ix[:,%d] > 0.5)
print 'Confusion Matrix Target Variable - %d'\n
print_confusion_matrix(matrix, ['True', 'False'])
plot_confusion_matrix(matrix)
plt.show()
''' % (col_idx, col_idx, col_idx))
      self.flush_cell()
      self.txt('##Reliability Diagrams<br/>', True)
      self.code('f, axs = plt.subplots(%d, 2, figsize=(12, %d))\naxs = axs.reshape(-1)' % 
        (int(math.ceil(self.y_pred.shape[1] / 2.0)), self.y_pred.shape[1]*4))
      for col_idx in range(self.y_pred.shape[1]):        
        self.code(
'''
y_score_bin_mean, empirical_prob_pos = reliability_curve((y_true==%d).values, y_pred.ix[:,%d].values, 200)
axs[%d].scatter(y_score_bin_mean, empirical_prob_pos)
axs[%d].set_title('Reliability Diagram - %d')
axs[%d].set_xlabel('Predicted Probability')
_ = axs[%d].set_ylabel('Empirical Probability')
''' % (col_idx, col_idx, col_idx, col_idx, col_idx, col_idx, col_idx))

    elif y_true_classification and y_pred_type == 'continuous': 
      # using predict proba      
      self.code(
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
      self.code(
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
    self.flush_cell()
    self.txt('\n\n\n', True)

  #############################################################################
  #       COLUMN TYPE CHARTS
  #############################################################################

  def categorical_charts(self, s):
    identifier = 'X["' + s.name + '"]' if 'X' in globals() and s.name in X.columns else s.name
    self.code('fig, axs = plt.subplots(1,1)')
    self.code('_ = ' + identifier + '.value_counts().plot(kind="barh")')        
    self.code('axs.set_ylabel("Value")')
    self.code('axs.set_xlabel("Count")')
    self.code('_ = axs.set_title("Field Values")')

  def continuous_charts(self, identifier):
    self.code([
      'fig, axs = plt.subplots(1,3, figsize=(12, 8))',
      identifier + '.hist(bins=30, ax=axs[0]).set_title("Distribution")',
      'bc = scipy.stats.boxcox(' + identifier +')',
      'df = pd.DataFrame(bc[0])',
      'df.hist(bins=30, ax=axs[1])',
      'axs[1].set_title("Boxcox Distribution l=%.3f" % bc[1])',
      '_ = ' + identifier + '.plot(kind="box", ax=axs[2]).set_title("Boxplot")'
    ])

  def relationship(self, a, b):
    identifier_a = 'X["' + a.name + '"]' if 'X' in globals() and a.name in X.columns else a.name
    identifier_b = 'X["' + b.name + '"]' if 'X' in globals() and b.name in X.columns else b.name

    type_a = self.type(a)
    type_b = self.type(b)

    if type_a == 'continuous':
      if type_b == 'continuous' :
        self.code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name + '")')      
        self.code(identifier_a + '.cov(' + identifier_b + ', method="pearson")')      
        self.code(identifier_a + '.cov(' + identifier_b + ', method="spearman")')      
      
      if type_b == 'binary' or type_b == 'multiclass':
        self.code('fig, axs = plt.subplots(1,2)')         
        self.code('_ = X[["' + a.name + '", "' + b.name + '"]].boxplot(by="' + b.name + '", ax=axs[0])')
        self.code('_ = X.plot(kind="scatter", x="' + b.name + '", y="' + a.name +'", ax=axs[1])')      
        self.code('_ = X.hist(column="' + a.name + '", by="y", figsize=(12, 12))')


    if type_a == 'multiclass':
      if type_b == 'continuous':        
        self.code('fig, axs = plt.subplots(1,1)') 
        self.code('_ = X.plot(kind="scatter", x="' + a.name + '", y="' + b.name +'", ax=axs[0])')      
      
      if type_b == 'multiclass':
        self.code('fig, axs = plt.subplots(1,1)')         
        self.code('X.boxplot(column="' + a.name + '", by="' + b.name + '", ax=axs)')

    if type_a == 'binary':
      if type_b == 'continuous' :
        self.code('fig, axs = plt.subplots(1,1)') 
        self.code('_ = ' + identifier_a + '.plot(kind="box", by="' + b.name + '", ax=axs)')


  def name_value(self, name, value, flush=False): 
    self.txt('\n<b>' + name + '</b>: ' + self.pretty(value), flush)

  def txt(self, text, flush=False): 
    if type(text) is list: self._text_lines += text
    else: self._text_lines.append(self.pretty(text))
    if flush: self.flush_cell()

  def code(self, code, flush=False): 
    if type(code) is list: self._code_lines += code
    else: self._code_lines.append(code)
    if flush: self.flush_cell()

  def pretty(self, value):
    if type(value) is float or \
      type(value) is np.float or \
      type(value) is np.float32 or \
      type(value) is np.float64: return '%.4f' % value
    else: return str(value)

  def warn(self, value):
    return '<span style="color:#8A0808"><b>' + self.pretty(value) + '</b></span>'

  def flush_cell(self):
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

  def type(self, data):
    if data is None: return None
    return sklearn.utils.multiclass.type_of_target(data)

  def create_notebook(self, cells, filename, start_notebook):
    nb = nbf.new_notebook()
    nb.cells = cells
    with open(filename, 'w') as f: f.write(nbf.writes(nb)) 
    call(['ipython', 'trust', filename])
    if start_notebook: call(['ipython', 'notebook', filename])
