from keras.preprocessing.image import *
import cv2, os, math
from PIL import Image
'''
TODO:
Add convolutional visualisation:
  https://github.com/guruucsd/CNN_visualization
'''
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
  for i in xrange(0, size, size_s):
    for j in xrange(0, size, size_s):
      if idx == len(imgs): continue      
      im = Image.fromarray(imgs[idx])
      im = im.resize((size_s, size_s))
      idx += 1
      im.thumbnail((size_s, size_s))
      new_im.paste(im, (i, j))
  new_im.save(filename)

def save_img(filename, img):
  if '.' not in filename: filename += '.png'
  cv2.imwrite(filename, img)

def rotate_imgs(imgs, angle=20):  
  return np.array([rotate_img(i, angle) for i in imgs])

def rotate_img(img, angle=20):  
  rows, cols = img.shape[:2]
  angle = random.uniform(-angle, angle)
  M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
  return cv2.warpAffine(img, M, (cols,rows))

def flip_imgs(imgs, horizontal=True):
  return np.array([flip_img(i, horizontal) for i in imgs])

def flip_img(img, horizontal=True):
  return cv2.flip(img, int(horizontal))

def shift_imgs(imgs, shift=(10, 10)):
  return np.array([shift_img(i, shift) for i in imgs])

def shift_img(img, shift=(10, 10)):
  rows, cols = img.shape[:2]
  M = np.float32([[1,0,random.uniform(-shift[0], shift[0])],[0,1,random.uniform(-shift[1], shift[1])]])
  return cv2.warpAffine(img,M,(cols,rows))

def zoom_imgs(imgs, factor=10):
  return np.array([zoom_img(i, factor) for i in imgs])

def zoom_img(img, factor=10):
  rows, cols = img.shape[:2]
  factor = random.uniform(factor/2., factor)
  pts1 = np.float32([[factor,factor],[cols-factor,factor],[factor,rows-factor],[cols-factor, rows-factor]])
  pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
  M = cv2.getPerspectiveTransform(pts1, pts2)
  return cv2.warpPerspective(img,M,(rows,cols))

def to_rgb(img):
  if len(img.shape) == 4: return np.array([to_rgb(i) for i in img])
  b,g,r = cv2.split(img)
  return cv2.merge([r,g,b])