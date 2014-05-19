
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

pylab.rcParams['figure.figsize'] = (15., 14.)

pd.set_option('display.mpl_style', 'default')
pd.set_option('display.width', 4000)
pd.set_option('display.max_columns', 100)

markers = ['o', 'v', '^', '<', '>', '*', 'h', 'H', '1', '2', '3', 
    '4', '8', 's', 'p', '+', 'x', 'D', 'd', '|', '-', '.', ',']
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def get_marker(i):
  return colours[i % len(colours)] + markers[i % len(markers)]

def plot_cats(X, y, n_samples=2000, n_cols=3):
  lbls = y.unique()
  for i, col in enumerate(X.columns):
    if (i > 11): break # No more than 12 figures please

    plt.subplot(len(X_s.columns)/n_cols, n_cols, i+1)
    plt.title(col)
    for i, lbl in enumerate(lbls):
      lbl_mask = [v == lbl for v in y]
      X_lbl = X[lbl_mask][:n_samples]
      plt.plot(X_lbl[col], get_marker(i))

def plot_scores(values, scores, sems):
  plt.ylabel('Scores Plot')
  plt.xlabel('Values')
  plt.title('Scores')
  plt.plot(values, scores, get_marker(0))
  if (sems):
    plt.plot(values, sems, get_marker(1))


def plot_probabilities_of(names, importances, title='Variable Importance', relative=True):  
  importances = 100.0 * (importances / importances.max()) if relative else 100.8 * importances
  sorted_idx = np.argsort(importances)
  pos = np.arange(sorted_idx.shape[0]) + .5
  plt.subplot(1, 2, 2)
  plt.barh(pos, importances[sorted_idx], align='center')
  plt.yticks(pos, names[sorted_idx])
  plt.xlabel('Relative Importance')
  plt.title(title)
