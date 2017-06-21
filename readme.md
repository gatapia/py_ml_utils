Collection of machine learning utilities for PicNet and Predict Bench
=====================================================================
PicNet and Predict Bench provide predictive analytics services and 
products like Centazio.  These products and services are supported
by this library that combines best in breed libraries, implementations,
algorithms and utilities that help us provice machine learning services
at speed.

See http://www.picnet.com.au for more details

Instructions:
- Python 2:
  - Check out a submodule to this lib name it ml
  - Create a <project_name>_utils.py file with project wide utilties
  - In <project_name>_utils.py add "from ml import *"
- Python 3:
  - Expectes a folder structure as follows:
    - src
      - utils.py (with `from ml import *`)
      - script01.py (with `import src.utils`)
    - ml [git submodule to this lib]
  - To run a script use `python -m src.script01`
  - Or in ipython `import src.utils` to get going
- Jupyter Notebook
  - ml will need to live in the src directory
  - "from ml import *"

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