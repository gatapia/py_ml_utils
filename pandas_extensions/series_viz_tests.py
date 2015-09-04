import unittest, sklearn, datetime
import pandas as pd, numpy as np
from . import *
from . import base_pandas_extensions_tester

start_notebook = True

class T(base_pandas_extensions_tester.BasePandasExtensionsTester):  

  def test_viz_returns_viz_object(self):
    v = pd.Series([1, 2, 3]).viz()
    self.assertIsNotNone(v)

  def test_compare_predictions(self):
    v = pd.Series(np.random.random(1000)).viz()
    # v.compare_predictions(np.random.random(1000), start_notebook=start_notebook)