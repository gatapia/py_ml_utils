from __future__ import print_function, absolute_import

import matplotlib as plt
import itertools, img_utils

def viz_history_loss(history, filename='viz\\validation_curve.png'):
  losses = {'loss': history['loss']}
  if 'val_loss' in history: losses.add('val_loss', history['val_loss'])

  viz_losses(losses, filename)

def viz_losses(losses, filename='viz\\validation_curve.png'):
  if '.' not in filename: filename += '.png'

  x = history['epoch']
  legend = losses.keys

  for v in losses.values: plt.plot(np.arange(len(v)) + 1, v, marker='.')

  plt.title('Loss over epochs')
  plt.xlabel('Epochs')
  plt.xticks(history['epoch'], history['epoch'])
  plt.legend(legend, loc = 'upper right')
  plt.savefig(filename)

def vis_convolutions(cnn, filename='viz\\convolutions.png'):
  convs = [l for l in cnn.layers if isinstance(l, Convolution2D)]
  for i, c in enumerate(convs): vis_convolution(c, str(i+1) + '_' + filename)

def vis_convolution(layer, filename):
  if '.' not in filename: filename += '.png'
  w = layer.get_weights()[0][:2500].reshape((32, 3, 3))
  w -= w.min()
  w /= w.max()
  w *= 255
  img_utils.save_imgs(np.ceil(w), filename)

def plot_confusion_matrix(cm, classes,
    normalize=False, title='Confusion matrix',
    cmap=plt.cm.Blues, filename='viz\\confusion_matrix.png'):
  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(filename)
