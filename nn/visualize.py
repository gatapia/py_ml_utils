from __future__ import print_function, absolute_import

from PIL import Image
from img_utils import *
import matplotlib as plt
import itertools

def viz_history_loss(filename, history):
  if '.' not in filename: filename += '.png'

  losses = {'loss': history['loss']}
  if 'val_loss' in history: losses.add('val_loss', history['val_loss'])

  save_losses(filename, losses)
  
def viz_losses(filename, losses):
  if '.' not in filename: filename += '.png'

  x = history['epoch']
  legend = losses.keys

  for v in losses.values: plt.plot(np.arange(len(v)) + 1, v, marker='.')

  plt.title('Loss over epochs')
  plt.xlabel('Epochs')
  plt.xticks(history['epoch'], history['epoch'])
  plt.legend(legend, loc = 'upper right')
  plt.savefig(filename)

def vis_convolutions(filename, cnn):
  convs = [l for l in cnn.layers if instanceof(l, Convolution2D)]
  for i, c in enumerate(convs): vis_convolution(filename + '_' + str(i+1), c)

def vis_convolution(filename, layer):
  if '.' not in filename: filename += '.png'
  w = layer.get_weights()[0][:2500].reshape((32, 3, 3))
  w -= w.min()  
  w /= w.max()
  w *= 255
  save_imgs(filename, np.ceil(w)) 

def plot_confusion_matrix(cm, classes, 
    normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):  
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  (This function is copied from the scikit docs.)
  """  

  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print(cm)
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig('confusion_matrix.png')
