__version__ = '0.1.0'

from contextlib import contextmanager
from .misc import *
from .pandas_extensions_init import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import builtins, time

plt.style.use('dark_background')

@contextmanager
def time_op(title):
    t0 = time.time()
    yield
    print("{} - took {:.0f}s".format(title, time.time() - t0))

def S(arr, *args, **kargs): return pd.Series(NP(arr).reshape(-1), *args, **kargs)

def DF(arr, *args, **kargs): return pd.DataFrame(NP(arr), *args, **kargs)

def NP(x, *args, **kargs):
    if hasattr(x, 'data'): x = x.data
    if hasattr(x, 'cpu'): x = x.cpu()
    if hasattr(x, 'numpy'): x = x.numpy()
    if hasattr(x, 'values'): x = x.values
    if isinstance(x, (memoryview, list, tuple)): x = np.asarray(x, *args, **kargs)
    return x

builtins.NP = NP
builtins.S = S
builtins.DF = DF
builtins.time_op = time_op
builtins.pd = pd
builtins.np = np
builtins.plt = plt