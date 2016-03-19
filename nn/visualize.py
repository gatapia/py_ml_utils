import cv2
from PIL import Image
from img_utils import *
import matplotlib as plt

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
