import numpy as np
import matplotlib.pyplot as plt

def bland_altman_plot(data1, data2, *args, **kwargs):
  data1   = np.asarray(data1)
  data2   = np.asarray(data2)
  mean    = np.mean([data1, data2], axis=0)
  diff    = data1 - data2            # Difference between data1 and data2
  md    = np.mean(diff)              # Mean of the difference
  sd    = np.std(diff, axis=0)       # Standard deviation of the difference

  plt.scatter(mean, diff, *args, **kwargs)
  plt.axhline(md,    color='gray', linestyle='--')
  plt.axhline(md + 2*sd, color='gray', linestyle='--')
  plt.axhline(md - 2*sd, color='gray', linestyle='--')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=None):
  if labels is None: labels = range(cm.shape[0])
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(cm.shape[0])
  plt.xticks(tick_marks, labels, rotation=45)
  plt.yticks(tick_marks, labels)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')  

def print_confusion_matrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
  """pretty print for confusion matrixes"""
  columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
  empty_cell = " " * columnwidth
  # Print header
  print "    " + empty_cell,
  for label in labels: 
    print "%{0}s".format(columnwidth) % label,
  print
  # Print rows
  for i, label1 in enumerate(labels):
    print "    %{0}s".format(columnwidth) % label1,
    for j in range(len(labels)): 
      cell = "%{0}.1f".format(columnwidth) % cm[i, j]
      if hide_zeroes:
        cell = cell if float(cm[i, j]) != 0 else empty_cell
      if hide_diagonal:
        cell = cell if i != j else empty_cell
      if hide_threshold:
        cell = cell if cm[i, j] > hide_threshold else empty_cell
      print cell,
    print

def reliability_curve(y_true, y_score, bins=10, normalize=False):
  """Compute reliability curve

  Reliability curves allow checking if the predicted probabilities of a
  binary classifier are well calibrated. This function returns two arrays
  which encode a mapping from predicted probability to empirical probability.
  For this, the predicted probabilities are partitioned into equally sized
  bins and the mean predicted probability and the mean empirical probabilties
  in the bins are computed. For perfectly calibrated predictions, both
  quantities whould be approximately equal (for sufficiently many test
  samples).

  Note: this implementation is restricted to binary classification.

  Parameters
  ----------

  y_true : array, shape = [n_samples]
      True binary labels (0 or 1).

  y_score : array, shape = [n_samples]
      Target scores, can either be probability estimates of the positive
      class or confidence values. If normalize is False, y_score must be in
      the interval [0, 1]

  bins : int, optional, default=10
      The number of bins into which the y_scores are partitioned.
      Note: n_samples should be considerably larger than bins such that
            there is sufficient data in each bin to get a reliable estimate
            of the reliability

  normalize : bool, optional, default=False
      Whether y_score needs to be normalized into the bin [0, 1]. If True,
      the smallest value in y_score is mapped onto 0 and the largest one
      onto 1.


  Returns
  -------
  y_score_bin_mean : array, shape = [bins]
      The mean predicted y_score in the respective bins.

  empirical_prob_pos : array, shape = [bins]
      The empirical probability (frequency) of the positive class (+1) in the
      respective bins.


  References
  ----------
  .. [1] `Predicting Good Probabilities with Supervised Learning
          <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf>`_

  """
  if normalize:  # Normalize scores into bin [0, 1]
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

  bin_width = 1.0 / bins
  bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

  y_score_bin_mean = np.empty(bins)
  empirical_prob_pos = np.empty(bins)
  for i, threshold in enumerate(bin_centers):
    # determine all samples where y_score falls into the i-th bin
    bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                             y_score <= threshold + bin_width / 2)
    # Store mean y_score and mean empirical probability of positive class
    y_score_bin_mean[i] = y_score[bin_idx].mean()
    empirical_prob_pos[i] = y_true[bin_idx].mean()
  return y_score_bin_mean, empirical_prob_pos
