import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow_io as tfio
from tensorflow import keras



""" Gradient Magnitude Map """
def Grad_Mag_Map(I, show = False):
  I = tf.reduce_mean(I, axis=-1, keepdims=True)
  I = tfa.image.median_filter2d(I, filter_shape=(3, 3), padding='REFLECT')
  x = tfio.experimental.filter.prewitt(I) 
  if show:
    x = tf.squeeze(x, axis=0)
    x = x.numpy()
  return x


""" Gradient Magnitude Similarity Map"""
def GMS(I, I_r, show=False, c=0.0026):
    g_I   = Grad_Mag_Map(I)
    g_Ir  = Grad_Mag_Map(I_r)
    similarity_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    if show:
      similarity_map = tf.squeeze(similarity_map, axis=0)
      similarity_map = similarity_map.numpy()
    return similarity_map


""" Gradient Magnitude Distance Map"""
def GMS_distance(I, I_r):
    x = tf.math.reduce_mean(1 - GMS(I, I_r))
    return x



### LOSS FUNCTIONS ####

""" Define MSGMS """
def MSGMS_Loss(I, I_r):
  # normal scale loss
  tot_loss = GMS_distance(I, I_r)
  # pool 3 times and compute GMS
  for _ in range(3):
    I   = tf.nn.avg_pool2d(I, ksize=2, strides=2, padding= 'VALID')
    I_r = tf.nn.avg_pool2d(I_r, ksize=2, strides=2, padding= 'VALID')
    # sum loss
    tot_loss += GMS_distance(I, I_r)

  return tot_loss/4


""" Define SSIM loss"""
def SSIM_Loss(I, I_r):
  I = tf.cast(I, dtype=tf.double)
  I_r = tf.cast(I_r, dtype=tf.double)

  ssim = tf.image.ssim(I, I_r, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
  return tf.reduce_mean(1 - ssim)


""" Define l2 loss"""
def L2_Loss(I, I_r):
  l2_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  return l2_loss(I, I_r)


""" Define total loss"""
def LOSS(I, I_r):
  l2_loss = L2_Loss(I, I_r)
  S_loss  = SSIM_Loss(I, I_r)
  M_loss  = MSGMS_Loss(I, I_r)

  x = 1 * l2_loss + 1 * S_loss + 1 * M_loss 
  return tf.reduce_mean(x)
