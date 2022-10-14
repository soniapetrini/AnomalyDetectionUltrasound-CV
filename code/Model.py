import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import numpy as np

import os
import random

import tensorflow_addons as tfa
import tensorflow_io as tfio


# Building blocks

def EncoderBlock(input, out_channels, kernel_size, stride):
  ## miniblock 1
  c1 = layers.Conv2D(out_channels, kernel_size, stride, 'same')(input)
  c1 = layers.BatchNormalization()(c1)
  c1 = layers.Activation(activations.relu)(c1)
  ## miniblock 2
  c1 = layers.Conv2D(out_channels, 3, 1, 'same')(c1)
  c1 = layers.BatchNormalization()(c1)
  c1 = layers.Activation(activations.relu)(c1)
  return c1

def DecoderBlock(up_input, down_input, out_channels):
  up_input = layers.Conv2DTranspose(out_channels, 4, 2, 'same')(up_input)
  input = layers.concatenate([up_input, down_input])

  ## miniblock 1
  c1 = layers.Conv2D(out_channels, 3, 1, 'same')(input)
  c1 = layers.BatchNormalization()(c1)
  c1 = layers.Activation(activations.relu)(c1)
  ## miniblock 2
  c1 = layers.Conv2D(out_channels, 3, 1, 'same')(c1)
  c1 = layers.BatchNormalization()(c1)
  c1 = layers.Activation(activations.relu)(c1)
  return c1



#Build the model

def AutoEncoder(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, loss_fn):
  # input layer with normalization
  inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  s = layers.Lambda(lambda x: x / 255)(inputs)

  # Encoding path
  e1 = EncoderBlock(s,   64, 3, 1)
  e2 = EncoderBlock(e1, 128, 4, 2)
  e3 = EncoderBlock(e2, 256, 4, 2)
  e4 = EncoderBlock(e3, 512, 4, 2)
  e5 = EncoderBlock(e4, 512, 4, 2)

  # Dencoding path 
  d1 = DecoderBlock(e5, e4, 512)
  d2 = DecoderBlock(d1, e3, 256)
  d3 = DecoderBlock(d2, e2, 128)
  d4 = DecoderBlock(d3, e1, 64)

  # final convolution 
  d5 = layers.Conv2D(3, 3, 1, 'same')(d4)
  #outputs = layers.Activation(activations.tanh)(d5)
  outputs = tf.keras.activations.relu(d5)

  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer='adam', loss=loss_fn)

  return model
