#!/usr/bin/env python3

import tensorflow as tf
import sys

sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')

from net.my_networks import *
from net.my_losses import *


########
# model functions used while training (needed for checkpoint based restore)
########

def model_fn_NetV04_LossL1SigL1(features, labels, mode, params):
  """
  model dir names and properties:
  model_name = NetV04_L1Sig4L1_down_tr1, lr=0.00014,
  weights=[500, 1500], sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
  """
  def loss_function(inp, gt, data_format='channels_first'):
    weights = [500, 1500]
    sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
    return pointwise_l1_loss_sig_l1_loss(inp, gt, data_format=data_format, weights=weights, sig_params_list=sig_params_list_current)
  network = NetworkV04
  learning_rate_base = 0.00014
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_NetV04Res_LossL1SigL1ExpResL1(features, labels, mode, params):
  """
    Used for network with a smaller decoder compared to NetV03
    model_dir names and properties of training instances:
    model_name = NetV04Res_L1SigL1ExpResL1_down_aug_tr1, weights=[500,500,500], res_converter_exp_conf, lr=0.00015
    model_name = NetV04Res_L1Sig4L1ExpResL1_down_tr1, weights=[500, 1500, 500], res_converter_exp_conf, lr=0.00014,
                  sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
    model_name = NetV04Res_L1Sig4L1ExpResL1_down_tr2, same as tr1 but with the new training dataset

  """
  def loss_function(inp, gt, data_format='channels_first'):
    weights = [500, 1500, 500]
    res_converter = res_converter_exp_conf
    sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
    return pointwise_l1_loss_sig_l1_loss_res_l1_loss(inp, gt, data_format=data_format, weights=weights,
      res_converter=res_converter_exp_conf, sig_params_list=sig_params_list_current)
  network = NetworkV04Res
  learning_rate_base = 0.00014
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_NetV03Res_LossL1SigL1ExpResL1(features, labels, mode, params):
  """
    Used for network with residual prediction along with depth.
    model_dir_names and properties of instances:
    model_name = NetV03Res_L1SigL1ExpResL1_down_aug_tr1, weights=[1,0.5,1], res_converter_exp_conf, lr=0.0001
    model_name = NetV03Res_L1SigL1ExpResL1_down_aug_tr1_1, weights=[1,0.5,1], res_converter_exp_conf, lr=0.001
    model_name = NetV03Res_L1SigL1ExpResL1_down_aug_tr2, weights=[1,1,1], res_converter_exp_conf, lr=0.001
    model_name = NetV03Res_L1SigL1ExpResL1_down_aug_tr2_1, weights=[1,1,1], res_converter_exp_conf, lr=0.01
    model_name = NetV03Res_L1SigL1ExpResL1_down_aug_tr3, weights=[1,0.7,0.5], res_converter_exp_conf, lr=0.001
    model_name = NetV04Res_L1SigL1ExpResL1_down_aug_tr3_1, weights=[1,0.7,0.5], res_converter_exp_conf, lr=0.01
    model_name = NetV04Res_L1SigL1ExpResL1_down_aug_tr3_2, weights=[1,0.7,0.5], res_converter_exp_conf, lr=0.0001

  """
  def loss_function(inp, gt, data_format='channels_first'):
    weights=[1,0.7,0.5]
    res_converter=res_converter_exp_conf
    return pointwise_l1_loss_sig_l1_loss_res_l1_loss(inp, gt, data_format=data_format, weights=weights, 
      res_converter=res_converter)
  network = NetworkV03Res
  learning_rate_base = 0.0001
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_Netv3_LossL1SigL1(features, labels, mode, params):
  """ Used in the net version with only kernel=3 convolutions, with equal weight for sig n l1 losses. 
    Note: The idepth used here is cleaned using sops.replace_nonfinite
    model_dir_names and associated weights: 
    model_name = NetV03_L1SigEqL1_down_aug, weights=[0.5, 0.5/3]
    model_name = NetV03_L1SigL1_down_aug_0.5, weights=[0.5, 0.5]
    model_name = NetV03_L1SigL1_down_aug_0.25, weights=[0.5, 0.25]
    """
  def loss_function(inp, gt, data_format='channels_first'):
    weights=[0.5, 0.25]
    return pointwise_l1_loss_sig_l1_loss(inp, gt, data_format=data_format, weights=weights)
  network = NetworkV03
  learning_rate_base = 0.0004
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_Netv3_LossL1SigL1_down(features, labels, mode, params):
  """ Used in the net version with only kernel=3 convolutions. 
    Note: The idepth used here is cleaned using sops.replace_nonfinite
    Note: model_name = NetV03_L1SigL1_down_aug
    """
  loss_function = pointwise_l1_loss_sig_l1_loss
  network = NetworkV03
  learning_rate_base = 0.0004
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_L2_loss_clean_down(features, labels, mode, params):
  """ Used in the early L2 loss clean for downsampled one. 
  Note: The idepth used here was cleaned with threshold = 0.01
  Note: model_name = NetV01_batch12_defInit_L2clean_downsampled_idepthClean
  """
  loss_function = pointwise_l2_loss_clean
  network = NetworkV01
  learning_rate_base = 0.0004
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

def model_fn_Netv2_LossL1SigL1_down(features, labels, mode, params):
  """
  Used in the net version resembling sparse invariant cnn!
  model dir names and properties:
  model_name = NetV02_L1Sig4L1_down_tr1, lr=0.0004, 
  weights=[500, 1500], sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
  """
  def loss_function(inp, gt, data_format='channels_first'):
    weights = [500, 1500]
    sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
    return pointwise_l1_loss_sig_l1_loss(inp, gt, data_format=data_format, weights=weights, sig_params_list=sig_params_list_current)
  network = NetworkV02
  learning_rate_base = 0.0004
  return model_fn_general(features, labels, mode, params, loss_function, network, learning_rate_base)

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
  #tf.summary.image('sparseIdepth', sparseIdepth_nhwc, max_outputs=1)
  tf.summary.image('sparseIdepthVar', sparseIdepthVar_nhwc, max_outputs=1)

  model = network()

  ######
  #predict
  ######
  if mode == tf.estimator.ModeKeys.PREDICT:
    depth = model(inputs, data_format=params.get('data_format'))
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
      labels_nhwc = sops.replace_nonfinite(tf.transpose(labels, [0,2,3,1]))
    else:
      depth_nhwc = depth
      labels_nhwc = sops.replace_nonfinite(labels)
    if depth_nhwc.shape[-1]==2:
      depths_nhwc = tf.concat([sparseIdepth_nhwc[0:1,:,:,:], depth_nhwc[0:1,:,:,0:1], 
        labels_nhwc[0:1,:,:,:]], 0)
    else:
      depths_nhwc = tf.concat([sparseIdepth_nhwc[0:1,:,:,:], depth_nhwc[0:1,:,:,:], 
        labels_nhwc[0:1:,:,:]], 0)
    tf.summary.image('depths', depths_nhwc, max_outputs=3)
    #tf.summary.image('depthGt', labels_nhwc, max_outputs=1)  

    # Save scalars to Tensorboard output.
    tf.summary.scalar('learning_rate', learning_rate)

    logging_hook = tf.train.LoggingTensorHook({"tower0_loss" : loss}, every_n_iter=10)
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, 
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()),
      training_hooks=[logging_hook])

  ######
  #eval
  ######
  if mode == tf.estimator.ModeKeys.EVAL:
    depth = model(inputs, data_format=params.get('data_format'))
    loss = loss_function(inp=depth, gt=labels, data_format=params.get('data_format'))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss)


#######################
# single Image models 
#######################
def modelfn_NetV0_LossL1SigL1(features, labels, mode, params):
  """
    Network for single image depth prediction.
    model_dir names and properties of instances:
    model_name =  NetV0_L1SigL1_tr1 weights=[500, 1500], res_converter_exp_conf, lr=0.0004, 
                  sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
    model_name =  NetV0_L1SigL1_tr2 weights=[500, 1500], res_converter_exp_conf, lr=0.00014, 
                  sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
  """ 
  def loss_function(inp, gt, data_format='channels_first'):
    weights = [500, 1500]
    sig_params_list_current = [{'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},]
    return pointwise_l1_loss_sig_l1_loss(inp, gt, data_format=data_format, weights=weights, sig_params_list=sig_params_list_current)
  learning_rate_base = 0.00014
  network = NetworkV0
  return model_fn_general_singleImage(features, labels, mode, params, loss_function, network, learning_rate_base)


def model_fn_general_singleImage(features, labels, mode, params, loss_function, network, learning_rate_base):
  """The model_fn argument for creating an Estimator."""
  inputs = features['rgb']

  # summaries for images
  if params.get('data_format') == "channels_first":
    rgb_nhwc = tf.transpose(inputs, [0,2,3,1])
  else:
    rgb_nhwc = inputs
  tf.summary.image('rgb', rgb_nhwc, max_outputs=1)

  model = network()

  ######
  #predict
  ######
  if mode == tf.estimator.ModeKeys.PREDICT:
    depth = model(inputs, data_format=params.get('data_format'))
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
      labels_nhwc = sops.replace_nonfinite(tf.transpose(labels, [0,2,3,1]))
    else:
      depth_nhwc = depth
      labels_nhwc = sops.replace_nonfinite(labels)
    if depth_nhwc.shape[-1]==2:
      depths_nhwc = tf.concat([depth_nhwc[0:1,:,:,0:1], labels_nhwc[0:1,:,:,:]], 0)
    else:
      depths_nhwc = tf.concat([depth_nhwc[0:1,:,:,:], labels_nhwc[0:1:,:,:]], 0)
    tf.summary.image('depths', depths_nhwc, max_outputs=2)
    #tf.summary.image('depthGt', labels_nhwc, max_outputs=1)  

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
    depth = model(inputs, data_format=params.get('data_format'))
    loss = loss_function(inp=depth, gt=labels, data_format=params.get('data_format'))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss)