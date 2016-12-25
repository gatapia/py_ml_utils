from __future__ import print_function, absolute_import

import os, math
import numpy as np
from PIL import Image, ImageChops
from keras import backend as K

def prepare_imgs(X):
  '''
  turns monochrome image into a single channeled 4D tensor suitable for keras
  '''
  if (len(X.shape) != 3): 
    print ('images are not monochrome 2 dimension images, not touching them')
    return X
    
  if K.image_dim_ordering() == 'th': X =  X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
  else: X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
  X = X.astype('float32')
  X /= X.max()
  return X

def dataset_shape(X):
  if (len(X.shape) != 4): return None
  else: return X.shape[1:]

def load_imgs(path, files=None, grayscale=False):
  if files is None: 
    def isimg(f):
      ext = f.split('.')[-1]
      return ext in ['jpg', 'gif', 'png', 'bmp']      
    files = [f for f in os.listdir(path) if isimg(f)]
  files = [os.path.join(path, f) for f in files]
  return [Image.open(f).convert("L" if grayscale else "RGB") for f in files]

def save_imgs_sample(
    imgs, 
    filename='sample.png', 
    sample_size=100, 
    image_size=800,
    shuffle=True):
  sample = np.random.permutation(imgs)[:sample_size] \
      if shuffle else imgs
  save_imgs(sample, filename, image_size)

def save_imgs(
    imgs, 
    filename='images.png', 
    image_size=800):
  mode, bgcol = 'RGB', (255, 255, 255)
  if (imgs.shape[-1] == 1): 
    mode, bgcol = 'L', 255
    imgs = imgs.reshape(-1, imgs.shape[1], imgs.shape[2])
  
  if imgs.max() <= 1: imgs = imgs * 255
  imgs = imgs.astype('uint8')

  if '.' not in filename: filename += '.png'
  
  new_im = Image.new(mode, (image_size, image_size))  
  rows = cols = math.ceil(math.sqrt(len(imgs)))  
  if (rows - 1) * cols >= len(imgs): rows -= 1
  size_s = int(math.ceil(image_size / float(cols)))
  idx = 0 
  for y in range(0, image_size, size_s):
    for x in range(0, image_size, size_s):
      if idx == len(imgs): 
        continue 
      im = Image.fromarray(imgs[idx], mode)      
      w_border = Image.new(mode, (size_s, size_s), bgcol)
      new_size = size_s - 2
      if im.size[0] != new_size or im.size[1] != new_size: 
        im = im.resize((new_size, new_size))
      w_border.paste(im, (1, 1))      
      idx += 1      
      new_im.paste(w_border, (x, y))
      new_im.paste(im, (x, y))
  new_im.save(filename)

def save_img(filename, img):
  if '.' not in filename: filename += '.png'
  if type(img) is np.array: img = Image.fromarray(img)
  img.save(filename)

def resize_imgs(imgs, size):
  return [resize_img(i, size) for i in imgs]

def resize_img(img, size):
  img = img.copy()
  img.thumbnail(size, Image.ANTIALIAS)
  new_size = img.size
  img = img.crop( (0, 0, size[0], size[1]))

  offset_x = max( (size[0] - new_size[0]) / 2, 0 )
  offset_y = max( (size[1] - new_size[1]) / 2, 0 )

  return ImageChops.offset(img, offset_x, offset_y)  

def rotate_imgs(imgs, angle=20): return [rotate_img(i, angle) for i in imgs]

def rotate_img(img, angle=20): return img.rotate(_get_rng_from_min_max(angle))

def toarr_imgs(imgs, keras_style=True):
  arr = np.array([np.asarray(img) for img in imgs])
  if keras_style: arr = np.swapaxes(arr,3,1)
  return arr

def flip_imgs(imgs, horizontal=True):
  return [flip_img(img, horizontal) for img in imgs]

def flip_img(img, horizontal=True):
  return img.transpose(Image.FLIP_LEFT_RIGHT if horizontal else Image.FLIP_TOP_BOTTOM)

def zoom_imgs(imgs, factor=10):
  return np.array([zoom_img(i, factor) for i in imgs])

def zoom_img(img, factor=10):
  rows, cols = img.shape[:2]  
  if isinstance(factor, (list, tuple)): factor = np.random.uniform(factor[0], factor[1])
  else: factor = np.random.uniform(factor/2., factor)
  pts1 = np.float32([[factor,factor],[cols-factor,factor],[factor,rows-factor],[cols-factor, rows-factor]])
  pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
  M = cv2.getPerspectiveTransform(pts1, pts2)
  return cv2.warpPerspective(img,M,(rows,cols))

def save_history_loss(filename, history):
  if '.' not in filename: filename += '.png'

  losses = {'loss': history['loss']}
  if 'val_loss' in history: losses.add('val_loss', history['val_loss'])
  
  save_losses(filename, losses)
  
def save_losses(filename, losses):
  if '.' not in filename: filename += '.png'

  x = history['epoch']
  legend = losses.keys

  for v in losses.values: plt.plot(np.arange(len(v)) + 1, v, marker='.')

  plt.title('Loss over epochs')
  plt.xlabel('Epochs')
  plt.xticks(history['epoch'], history['epoch'])
  plt.legend(legend, loc = 'upper right')
  plt.savefig(filename)

def _get_rng_from_min_max(num):
  if isinstance(num, (list, tuple)): return np.random.uniform(num[0], num[1])
  else: return np.random.uniform(-num, num)