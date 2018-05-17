#!/usr/bin/env python3
import tensorflow as tf
#import lmbspecialops as sops

def pointwise_l2_loss(inp, gt, epsilon=1e-6, data_format='channels_last'):
    """Computes the pointwise unsquared l2 loss.
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l2_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = replace_nonfinite(inp-gt_)
        if data_format == 'channels_first':
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=1)+epsilon))
        else: # NHWC
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=3)+epsilon))

def pointwise_l2_loss_clean(inp, gt, epsilon=1e-6, data_format='channels_last'):
    """Computes the pointwise unsquared l2 loss.
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l2_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = replace_nonfinite_loss(inp-gt_, gt_)
        if data_format == 'channels_first':
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=1)+epsilon))
        else: # NHWC
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=3)+epsilon))

def pointwise_l1_loss_norm(inp, gt, data_format='channels_last'):
    """Computes the pointwise unsquared l2 loss.
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l1_loss_norm'):
        gt_ = tf.stop_gradient(gt)
        mean_idepth = tf.reduce_mean(gt_)
        diff = replace_nonfinite(tf.abs(inp-gt_))
        diff_norm = tf.divide(diff, mean_idepth)
        if data_format == 'channels_first':
            return tf.reduce_mean(tf.reduce_sum(diff_norm, axis=1))
        else: # NHWC
            return tf.reduce_mean(tf.reduce_sum(diff_norm, axis=3))

def pointwise_l1_loss(inp, gt, epsilon=1e-6, data_format='channels_last'):
    """Computes the pointwise unsquared l2 loss.
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    The loss is normalized by the number of pixels.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l1_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = replace_nonfinite(tf.abs(inp-gt_))
        if data_format == 'channels_first':
            return tf.reduce_mean(tf.reduce_sum(diff, axis=1))
        else: # NHWC
            return tf.reduce_mean(tf.reduce_sum(diff, axis=3))

def replace_nonfinite(x):
    mask = tf.is_finite(x)
    x_clean = tf.where(mask, x, tf.zeros_like(x))
    return x_clean

def replace_nonfinite_loss(diff, gt):
    mask = tf.logical_and(tf.is_finite(diff), tf.greater(gt, 0.))
    x_clean = tf.where(mask, diff, tf.zeros_like(diff))
    return x_clean