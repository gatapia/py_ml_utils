from __future__ import print_function, absolute_import

import math
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras import backend as K

def prepare_monochrome_images(X, convert_to_rgb=True):
  '''
  turns monochrome image into a single channeled 4D tensor suitable for keras
  '''
  if (len(X.shape) != 3): 
    print ('images are not monochrome 2 dimension images, not touching them')
    return X
  
  X = X.astype('float32')
  X /= X.max()

  if convert_to_rgb:
    arrays = [X, X, X]
    axis = 1 if K.image_dim_ordering() == 'th' else 3
    return np.stack(arrays, axis=axis)
  else:
    if K.image_dim_ordering() == 'th': return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else: return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  

def save_imgs_sample_for_classes(
    imgs, 
    filename_prefix, 
    classes, 
    sample_size=100, 
    image_size=800, 
    shuffle=True):
  for cls in range(max(classes)):
    save_imgs_sample(imgs[classes==cls], filename_prefix + '_' + str(cls) + '.png', sample_size, image_size, shuffle)

def save_imgs_sample(
    imgs, 
    filename='viz\\sample.png', 
    sample_size=100, 
    image_size=800,
    shuffle=True):
  if type(imgs) is image.NumpyArrayIterator: 
    tmp = imgs.next()[0]
    while(len(tmp)) < sample_size: tmp = tmp + imgs.next()[0]
    imgs = tmp

  sample = np.random.permutation(imgs)[:sample_size] \
      if shuffle else imgs
  save_imgs(sample, filename, image_size)

def save_imgs(
    imgs, 
    filename='images.png', 
    image_size=800):
  mode, bgcol = 'RGB', (255, 255, 255)
  channel_axis = 1 if K.image_dim_ordering() == 'th' else 3
  if (imgs.shape[channel_axis] == 1): 
    mode, bgcol = 'L', 255
  
  if channel_axis == 1:      
    imgs = np.swapaxes(imgs, 1, 3)

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
      if idx == len(imgs): continue             
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
