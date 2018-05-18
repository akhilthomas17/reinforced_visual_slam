#!/usr/bin/env python3

import tensorflow as tf
import sys

sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')

from net.my_networks import *
from net.my_losses import *


########
# model functions used while training (needed for checkpoint based restore)
########

def model_fn_L2_loss_clean_down(features, labels, mode, params):
  """ Used in the early L2 loss clean for downsampled one. 
  Note: The idepth used here was cleaned with threshold = 0.01
  Note: model_name = NetV01_batch12_defInit_L2clean_downsampled_idepthClean
  """
  loss_function = pointwise_l2_loss_clean
  network = NetworkV01
  learning_rate_base = 0.0004
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_Netv2_LossL1CleanSigL1_down(features, labels, mode, params):
	""" Used in the net version resembling sparse invariant cnn. 
  	Note: The idepth used here was not cleaned with threshold = 0
  	Note: model_name = NetV02_L1cleanSigL1_downsampled
  	"""
	loss_function = pointwise_l1_loss_clean_sigma_l1_loss
	network = NetworkV02
	learning_rate_base = 0.0004
	return model_fn_general(features, labels, mode, params, loss_function, network, 
		learning_rate_base)

#########
# general model function
#########

def model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base):
  """The model_fn argument for creating an Estimator."""
  rgb = features['rgb']
  sparseIdepth = features['sparseInverseDepth']
  sparseIdepthVar = features['sparseInverseDepthVariance']
  inputs = [rgb, sparseIdepth, sparseIdepthVar]

  # summaries for images
  if params.get('data_format') == "channels_first":
    rgb_nhwc = tf.transpose(rgb, [0,2,3,1])
    sparseIdepth_nhwc = tf.transpose(sparseIdepth, [0,2,3,1])
    sparseIdepthVar_nhwc = tf.transpose(sparseIdepthVar, [0,2,3,1])
  else:
    rgb_nhwc = rgb
    sparseIdepth_nhwc = sparseIdepth
    sparseIdepthVar_nhwc = sparseIdepthVar
  tf.summary.image('rgb', rgb_nhwc, max_outputs=1)
  tf.summary.image('sparseIdepth', sparseIdepth_nhwc, max_outputs=1)
  tf.summary.image('sparseIdepthVar', sparseIdepthVar_nhwc, max_outputs=1)

  model = network()

  ######
  #predict
  ######
  if mode == tf.estimator.ModeKeys.PREDICT:
    depth = model(inputs, params.get('data_format'))
    predictions = {
        'depth': depth,
    }
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, 
                                  predictions=predictions, 
                                  export_outputs={'depth': tf.estimator.export.PredictOutput(predictions)})

  #########
  #train
  #########
  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.exponential_decay(learning_rate_base, tf.train.get_or_create_global_step(), 
      decay_rate=0.8, decay_steps=700000, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if params.get('multi_gpu'):
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    
    depth = model(inputs, data_format=params.get('data_format'))
    loss = loss_function(inp=depth, gt=labels, data_format=params.get('data_format'))
    if params.get('data_format') == "channels_first":
      depth_nhwc = tf.transpose(depth, [0,2,3,1])
      labels_nhwc = tf.transpose(labels, [0,2,3,1])
    else:
      depth_nhwc = depth
      labels_nhwc = labels
    tf.summary.image('depthPredicted', depth_nhwc, max_outputs=1)
    tf.summary.image('depthGt', labels_nhwc, max_outputs=1)  

    # Save scalars to Tensorboard output.
    tf.summary.scalar('learning_rate', learning_rate)
    #tf.summary.scalar('total_loss', loss)

    logging_hook = tf.train.LoggingTensorHook({"tower0_loss" : loss}, every_n_iter=10)
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, 
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()),
      training_hooks=[logging_hook])

  ######
  #eval
  ######
  if mode == tf.estimator.ModeKeys.EVAL:
    depth = model(inputs, params.get('data_format'))
    loss = loss_function(inp=labels, gt=depth, data_format=params.get('data_format'))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss)

