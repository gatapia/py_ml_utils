__version__ = '0.1.0'

from .misc import *
from .pandas_extensions_init import *

import pandas as pd
import numpy as np
import scipy, sklearn
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def S(arr, *args, **kargs):
    if hasattr(arr, 'data'): arr = arr.data
    if hasattr(arr, 'cpu'): arr = arr.cpu()
    if hasattr(arr, 'numpy'): arr = arr.numpy()
    return pd.Series(arr.reshape(-1), *args, **kargs)

def DF(arr, *args, **kargs):
    if hasattr(arr, 'data'): arr = arr.data
    if hasattr(arr, 'cpu'): arr = arr.cpu()
    if hasattr(arr, 'numpy'): arr = arr.numpy()
    return pd.DataFrame(arr, *args, **kargs)