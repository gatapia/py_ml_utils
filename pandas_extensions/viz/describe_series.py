import pandas as pd, numpy as np, matplotlib.pyplot
from .. import misc
from .utils import Utils

class DescribeSeries(Utils):
  #############################################################################
  #       Public Interface
  #############################################################################

  def __init__(self, s):    
    super(DescribeSeries, self).__init__()
    self.s = s

  def compare_distribution(self, other):
    if type(other) is pd.DataFrame: other = other[self.s.name]
    if self.s.is_date():
      df = pd.DataFrame({
        'column 1': self.s.groupby(self.s.dt.month).count(), 
        'column 2': other.groupby(other.dt.month).count()
        })
      plt = df.plot(kind='bar', alpha=0.5)
    else:
      raise Exception('TODO')
    matplotlib.pyplot.show(block=True)

  def compare_predictions(self, y_pred, start_notebook=True, notebook_file='dataset_description.ipynb'):        
    if type(y_pred) is not pd.Series: y_pred = pd.Series(y_pred)
    self.create_notebook(      
        self._get_prediction_comparison_cells(self.s, y_pred), 
        notebook_file,
        start_notebook)    

  def _get_prediction_comparison_cells(self, y_true, y_pred): 
    if len(y_true) != len(y_pred): raise Exception('y_true and y_pred are not compatible shapes (must be same rows and same columns)')    

    self.cells = []
    misc.start('get_prediction_comparison_cells')
    dataset_name = '_get_prediction_comparison_cells_data'
    misc.dump(dataset_name, (y_true, y_pred)) 

    y_true = y_true.copy()
    y_true.name = 'y_true'
    y_pred = y_pred.copy()
    y_pred.name = 'y_pred'
    self.do_global_imports(dataset_name, ['y_true', 'y_pred'])
    self.do_target_description(y_true, False, 'y_true')
    self.do_target_description(y_pred, False, 'y_pred')
    self.do_target_comparison_charts(y_true, y_pred)    
    misc.stop('get_prediction_comparison_cells')    
    return list(self.cells)
