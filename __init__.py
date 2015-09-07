'''
Collection of machine learning utilities for PicNet and Predict Bench
=====================================================================
PicNet and Predict Bench provide predictive analytics services and 
products like Centazio.  These products and services are supported
by this library that combines best in breed libraries, implementations,
algorithms and utilities that help us provice machine learning services
at speed.

See http://www.predictbench.com for more details

Instructions:
- Check out a submodule to this lib name it ml
- Create a <project_name>_utils.py file with project wide utilties
- In <project_name>_utils.py add "from ml import *"

This will inject all the required libraries into your environment 
including:
- pandas (as pd)
- numpy (as np)
- scipy
- sklearn
- all utiltiy functions in misc.py
- all pandas extensions defined in pandas_extensions

License: MIT
Author: Guido Tapia - guido.tapia@picnet.com.au
'''
__version__ = '0.1.0'

from .misc import *
from .pandas_extensions import *

import pandas as pd
import numpy as np
import scipy, sklearn
