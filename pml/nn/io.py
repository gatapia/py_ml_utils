from __future__ import print_function, absolute_import

import os, math, shutil, glob, keras
import numpy as np
from PIL import Image, ImageChops
from keras.preprocessing import image
from keras import backend as K


def load_test_imgs(dir):
  tmp_class = dir.split('\\')[-1]
  parent = os.path.join(dir, '..')
  batches = get_batches(parent, batch_size=1, class_mode=None, shuffle=False, classes=[tmp_class])
  data = np.concatenate([batches.next() for i in range(batches.nb_sample)])    
  return data, [f.split('\\')[-1] for f in batches.filenames]

def split_train_directory_into_train_and_valid(source_dir,
    dest_train_dir='data\\train_subset',
    dest_valid_dir='data\\valid_subset',
    validation_perc=.15, extension='*.jpg'):  
  if (os.path.exists(dest_train_dir)): os.rmdir(dest_train_dir)
  if (os.path.exists(dest_valid_dir)): os.rmdir(dest_valid_dir)
  if (not os.path.exists(dest_train_dir)): os.makedirs(dest_train_dir)
  if (not os.path.exists(dest_valid_dir)): os.makedirs(dest_valid_dir)

  batches = get_batches(source_dir)
  classes = batches.classes
  filenames = np.array(batches.filenames)
  indexes = np.random.permutation(len(classes))
  split = int(math.floor(len(filenames) * validation_perc))
  
  valid = filenames[indexes][:split]
  train = filenames[indexes][split:] 
  valid_classes = classes[indexes][:split]
  train_classes = classes[indexes][split:]

  def _copy(images, dest_dir):
    for img in images: 
      to_file = dest_dir + '\\' + img
      img_dir = os.path.dirname(to_file)
      if not os.path.exists(img_dir): os.makedirs(img_dir)
      shutil.copy(source_dir + '\\' + img, to_file)

  _copy(valid, dest_valid_dir)
  _copy(train, dest_train_dir)

  train_batches = get_batches(dest_train_dir, batch_size=1, class_mode=None, shuffle=False)
  valid_batches = get_batches(dest_valid_dir, batch_size=1, class_mode=None, shuffle=False)
  train_data = np.concatenate([train_batches.next() for i in range(train_batches.nb_sample)])
  valid_data = np.concatenate([valid_batches.next() for i in range(valid_batches.nb_sample)])

  return (train_data, train_batches.classes, valid_data, valid_batches.classes)

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, 
      batch_size=4, class_mode='categorical', target_size=(224,224), classes=None):
  return gen.flow_from_directory(dirname, target_size=target_size,
      class_mode=class_mode, shuffle=shuffle, batch_size=batch_size, classes=classes)

def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, 
        batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.nb_sample)])
