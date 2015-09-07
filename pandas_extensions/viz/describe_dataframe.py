import sklearn
from .. import misc
from . import *
from sklearn import ensemble, feature_selection
from IPython.nbformat import v4 as nbf
from IPython import html
from subprocess import call
from .utils import Utils

class DescribeDataFrame(Utils):
  #############################################################################
  #       Public Interface
  #############################################################################

  def __init__(self):    
    super(DescribeDataFrame, self).__init__()
    self._importance_row_limit = 50000
    self._min_variance_threshold = 0.01    

  def show_dataset(self, X, opt_y=None, start_notebook=True, notebook_file='dataset_description.ipynb'):    
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
    self.is_regression = self.type(self.y) == 'continuous'

    self.create_notebook(
        self.get_dataset_cells(), 
        notebook_file, 
        start_notebook)    


  def get_dataset_cells(self):
    self.cells = []
    misc.start('get_dataset_cells')
    dataset_name = '_get_dataset_cells_data'
    data_x = self.X.copy()
    data_x['y'] = self.y
    misc.dump(dataset_name, (data_x, self.y)) 

    self.do_global_imports(dataset_name, ['X', 'y'])    
    self._do_header_markdown()      
    self._intialise_feature_basic_details()
    self._do_column_summary_table()
    # TODO: Do table of all columns x column with correlation 
    #   values for all relationships.
    # TODO: Correlation matrix of all variables
    self._do_column_summary_charts()
    if self.y is not None: self.do_target_description(self.y, True)
    self._do_all_columns_details()
    misc.stop('done get_dataset_cells')
    return list(self.cells)
  
  #############################################################################
  #       Header/Misc
  #############################################################################


  def _do_header_markdown(self):
    self.txt('<hr/>\n# General Description of Data')
    self.name_value('Number of columns', len(self.X.columns))
    self.name_value('Number of rows', self.X.shape[0])
    if self.X.shape[0] > 5e5: self.name_value('Using subsample of', '50,000 rows')
    self.flush_cell()

  def _intialise_feature_basic_details(self):
    self.importances = self.__get_column_importances()    
    self.variances = self.X.var().values
    if self.y is None:
      self.col_details = zip(self.X.columns, self.importances, self.variances)
    elif self.is_regression:
      self.f_scores = self.__get_column_f_regression_scores()    
      self.col_details = zip(self.X.columns, self.importances, self.f_scores, self.variances)
    else:
      self.f_classif_scores = self.__get_column_f_classification_scores()    
      self.chi2_scores = self.__get_column_chi2_classification_scores()          
      self.col_details = zip(self.X.columns, self.importances, self.f_classif_scores, self.chi2_scores, self.variances)

    self.col_details = sorted(self.col_details, key=lambda d: d[1], reverse=True)
      
  #############################################################################
  #       COLUMN HELPERS
  #############################################################################

  def _get_column_importances(self):
    if self.y is None: return np.ones(self.X.shape[0])
    misc.start('_get_column_importances')    
    rf = ensemble.RandomForestRegressor(50) if self.is_regression else ensemble.RandomForestClassifier(50)    
    rf.fit(self.X_no_nan[:self._importance_row_limit], self.y[:self._importance_row_limit])
    misc.stop('done _get_column_importances, num feats: ' + `len(rf.feature_importances_)`)
    return rf.feature_importances_
  
  def _get_column_f_regression_scores(self):
    misc.start('_get_column_f_regression_scores')
    scores = feature_selection.f_regression(self.X_no_nan, self.y)[0]
    misc.stop('_get_column_f_regression_scores')
    return scores

  def _get_column_chi2_classification_scores(self):
    misc.start('_get_column_chi2_classification_scores')
    scores = feature_selection.chi2(self.X_no_nan, self.y)[0]
    misc.stop('_get_column_chi2_classification_scores')
    return scores

  def _get_column_f_classification_scores(self):
    misc.start('_get_column_f_classification_scores')
    scores = feature_selection.f_classif(self.X_no_nan, self.y)[0]
    misc.stop('_get_column_f_classification_scores')
    return scores

  def _get_column_specified_type(self, col_name):
    if col_name.startswith('c_'): return 'categorical'
    if col_name.startswith('i_'): return 'categorical index'
    if col_name.startswith('n_'): return 'continuous'
    if col_name.startswith('b_'): return 'binary (0, 1)'
    if col_name.startswith('d_'): return 'date'
    return 'unknown'

  #############################################################################
  #       Columns Summary
  #############################################################################

  def _do_column_summary_table(self):    
    self.txt('<hr/>\n#Features Summary Table')
    self.txt('<table>')
    self._do_column_summary_header_row()
    for idx, col in enumerate(self.col_details): self._do_column_summary_row(idx, col)            
    self.txt('</table>', True)

  def _do_column_summary_header_row(self):
    cols = ['Column', 'Inferred', 'Specified', 'RF Imp Rank', 'RF Importance V', 'F Score']
    if not self.is_regression: cols.append('Chi2')
    cols.append('Var')
    self.txt('<tr><th>' + '</th><th>'.join(cols) + '</th></tr>')

  def _do_column_summary_row(self, idx, col_details):
    col_name = col_details[0]
    inferred = self.type(self.X[col_name])
    specified = self.  __get_column_specified_type(col_name)
    inf_same_specified = inferred == specified
    cols = [
      col_name, 
      inferred, 
      specified if inf_same_specified else self.warn(specified),
      idx + 1,
      col_details[1]      
    ]
    if self.is_regression: cols.append(col_details[2])
    else: cols += [col_details[2], col_details[3]]
    variance = col_details[-1]
    if variance < self._min_variance_threshold: cols.append('<b>' + self.pretty(col_details[-1]) + '</b>')
    else: cols.append(col_details[-1])
    self.txt('<tr><td>' + '</td><td>'.join(map(self.pretty, cols)) + '</td></tr>')

  def _do_column_summary_charts(self):
    numericals = self.X.numericals()
    self.txt('<hr/>\n#Top ' + `max(5, len(numericals))` + ' Column Interaction', True)
    valid_col_details = filter(lambda c: c[0] in numericals, self.col_details)
    top_X = map(lambda c: c[0], valid_col_details[:5])
    print 'numericals:', numericals,'top_X:', top_X, 'valid_col_details:', valid_col_details
    self.code([
      'top_X = ' + str(top_X),      
      '_ = pd.tools.plotting.scatter_matrix(X[top_X], alpha=0.2, figsize=(12, 12), diagonal="kde")'
    ], True)    

    self.txt('<hr/>\n#PCA Explained Variable Ratios', True)
    self.code(['X2 = X.copy().missing("na", 0)',
        'X2 = X2[X2.numericals()]'
        'ratios = decomposition.PCA(X2.shape[1]).fit(X2).explained_variance_ratio_',
        'df = pd.DataFrame({"variance": ratios})',
        '_ = df.plot(kind="bar", figsize=(12, 10))'], True)


  #############################################################################
  #       Columns Details
  #############################################################################

  def _do_all_columns_details(self):
    self.txt('<hr/>\n#Features Details', True)
    for idx, col in enumerate(self.col_details):       
      self._do_column_details(idx, col)  


  def _do_column_details(self, idx, col):    
    col_name = col[0]
    c = self.X[col_name]    
    specifiedself.type = self.  __get_column_specified_type(col_name)
    inferredself.type = self.type(c)

    self.txt('\n\n<hr/>\n## ' + col_name)

    self.name_value('Distinct values', len(pd.unique(c)))
    self.name_value('Specified type', self.  __get_column_specified_type(col_name))
    self.name_value('Inferred type', self.type(c))
    self.name_value('RF Importance Rank', idx + 1)
    self.name_value('RF Importance Score', col[1])

    if self.is_regression: self.name_value('F-Score', col[2])
    else:
      self.name_value('F-Score', col[2])
      self.name_value('Chi2 Score', col[3])
    self.name_value('Variance', col[-1])

    if specifiedself.type != inferredself.type:
      self.txt(self.warn('Note:') + ' Check if specified type is correct as it does not match inferred type')
    
    self.txt('\n', True)
    
    if col_name.startswith('c_') or col_name.startswith('b_') \
        or col_name.startswith('i_'): self.categorical_charts(c)
    elif col_name.startswith('n_'): self.continuous_charts('X[' + `c.name` + ']')
    self.flush_cell()

    if self.y is not None:
      self.txt('### Relationship to target variable' + '\n', True)
      self.relationship(c, self.y)
      self.flush_cell()

    self.txt('\n\n\n', True)

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