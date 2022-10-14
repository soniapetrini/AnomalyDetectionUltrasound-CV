import tensorflow as tf
import numpy as np
import os
import random
import cv2
import pandas as pd
#from sklearn.model_selection import train_test_split


### Preparation
def read_and_resize(path, size=(256,256)):
  image   = cv2.imread(path)                
  image   = cv2.resize(image, size)
  return image

def GenCoords():
  a = random.randint(60,160)
  b = random.randint(a+40,200)
  while (b-a<=60):
    a = random.randint(60,160)
    b = random.randint(a+40,200)
  return [a, b, b, a]

def mask_image(image, mask_coords = [110, 150, 150, 110]):
  mask_color = (0, 0, 0) # black
  x, y, w, z = mask_coords
  copy_image = image.copy()
  if tf.is_tensor(image):                                   # convert to numpy for masking
    copy_image = copy_image.numpy()
  cv2.rectangle(copy_image, (x, y), (w, z), mask_color, -1) # mask
  if tf.is_tensor(image):                                   # back to tensor
    copy_image = tf.constant(copy_image)
  return copy_image

common_dir = 'AnomalyDetectionUltrasound-CV/'
classes    = ['faces','comics']
size       = (256,256)


### Pre-processing

# paths to all data
paths = {}    
for cl in classes:
  for f in os.listdir(f'{common_dir}sample_data/'+cl):
    paths[f'{common_dir}sample_data/{cl}/{f}'] = classes.index(cl)

# labelled dataframe relating paths to labels
data_df = pd.DataFrame.from_dict(paths, 'index', columns = ['label'])
data_df['coords'] = [GenCoords() for _ in range(len(data_df))]


# split
train_data_df = data_df[:200]     # only faces
test_data_df  = data_df[200:400]  # both faces and comics


# get image files
Y_train = train_data_df.index
Y_test = test_data_df.index


# read and resize images
Y_train = map(read_and_resize, Y_train)
Y_train = np.array(list(Y_train)).reshape(-1, size[0], size[1], 3)

Y_test  = map(read_and_resize, Y_test)
Y_test  = np.array(list(Y_test)).reshape(-1, size[0], size[1], 3)


# masking
X_train = list(map(mask_image, Y_train, train_data_df['coords']))
X_train = np.array(X_train).reshape(-1, size[0], size[1], 3)

X_test = list(map(mask_image, Y_test, test_data_df['coords']))
X_test = np.array(X_test).reshape(-1, size[0], size[1], 3)


# create tensor version of train data
X_train_t = tf.constant(X_train, dtype=tf.float32)
Y_train_t = tf.constant(Y_train, dtype=tf.float32)

