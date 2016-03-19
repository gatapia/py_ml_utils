import cv2, os, math
from PIL import Image
import numpy as np

def load_imgs(path, files=None, grayscale=False):
  if files is None: files = os.listdir(path)
  return np.array([cv2.imread(os.path.join(path, f), 0) for f in files]) if grayscale \
      else np.array([cv2.imread(os.path.join(path, f)) for f in files]) 

def save_imgs(filename, imgs, size=800):
  if '.' not in filename: filename += '.png'
  new_im = Image.new('RGB', (size, size))  
  rows = cols = math.ceil(math.sqrt(len(imgs)))
  if (rows - 1) * cols >= len(imgs): rows -= 1
  size_s = int(math.ceil(size / float(cols))) 
  idx = 0 
  for y in xrange(0, size, size_s):
    for x in xrange(0, size, size_s):
      if idx == len(imgs): continue      
      im = Image.fromarray(imgs[idx])
      w_border = Image.new("RGB", (size_s, size_s), (255, 255, 255))
      im = im.resize((size_s - 2, size_s - 2))
      w_border.paste(im, (1, 1))      
      idx += 1      
      new_im.paste(w_border, (x, y))
  new_im.save(filename)

def save_img(filename, img):
  if '.' not in filename: filename += '.png'
  cv2.imwrite(filename, img)

def rotate_imgs(imgs, angle=20):  
  return np.array([rotate_img(i, angle) for i in imgs])

def rotate_img(img, angle=20):  
  rows, cols = img.shape[:2]
  M = cv2.getRotationMatrix2D((cols/2,rows/2), _get_rng_from_min_max(angle), 1)
  return cv2.warpAffine(img, M, (cols,rows))

def flip_imgs(imgs, horizontal=True):
  return np.array([flip_img(i, horizontal) for i in imgs])

def flip_img(img, horizontal=True):
  return cv2.flip(img, int(horizontal))

def shift_imgs(imgs, shift=(10, 10)):
  return np.array([shift_img(i, shift) for i in imgs])

def shift_img(img, shift=(10, 10)):
  rows, cols = img.shape[:2]
  M = np.float32([[1,0, _get_rng_from_min_max(shift[0])],[0,1,_get_rng_from_min_max(shift[1])]])
  return cv2.warpAffine(img,M,(cols,rows))

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

def to_rgb(img):
  if len(img.shape) == 4: return np.array([to_rgb(i) for i in img])
  b,g,r = cv2.split(img)
  return cv2.merge([r,g,b])

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