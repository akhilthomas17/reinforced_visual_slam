#!/usr/bin/env python3
import tensorflow as tf
import lmbspecialops as sops

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

def mean_l1_loss(inp, gt, epsilon=1e-6):
    """ L1 loss ignoring nonfinite values.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l1_loss'):
        gt_ = tf.stop_gradient(gt)
        diff = sops.replace_nonfinite(inp-gt_)
        return tf.reduce_mean(tf.sqrt(diff**2 + epsilon))

def mean_l1_loss_nonfinite(inp, gt, epsilon=1e-6):
    """L1 loss ignoring nonfinite values.

    Returns a scalar tensor with the loss.
    The loss is the mean.
    """
    with tf.name_scope("l1_loss_nonfinite"):
        gt_ = tf.stop_gradient(gt)
        diff = sops.replace_nonfinite(inp-gt_)
        return tf.reduce_mean(tf.sqrt(diff**2 + epsilon))

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
    with tf.name_scope('pointwise_l2_loss_clean'):
        gt_ = tf.stop_gradient(gt)
        diff = replace_nonfinite_loss(inp-gt_, gt_)
        if data_format == 'channels_first':
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=1)+epsilon))
        else: # NHWC
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(diff**2, axis=3)+epsilon))

def mean_l1_loss_robust(inp, gt, epsilon=1e-6):
    """ L1 loss ignoring nonfinite values.
    
    inp: Tensor
        This is the prediction.
        
    gt: Tensor
        The ground truth with the same shape as 'inp'
        
    epsilon: float
        The epsilon value to avoid division by zero in the gradient computation
    """
    with tf.name_scope('pointwise_l1_loss_robust'):
        gt_ = tf.stop_gradient(gt)
        diff = replace_nonfinite_loss(inp-gt_, gt_)
        return tf.reduce_mean(tf.sqrt(diff**2 + epsilon))


def pointwise_l1_loss_clean_sigma_l1_loss(inp, gt, data_format='channels_first', weights=[0.4, 0.6]):
    """ Computes pointwise l1 loss and sigma loss (gradient).
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    Total loss is weighted sum of both normalized by the number of pixels.
    """
    with tf.name_scope('pointwise_l1_loss_clean_sigma_l1_loss'):
        loss_depth_l1 = mean_l1_loss_robust(inp, gt)
        tf.summary.scalar('loss_depth_l1', loss_depth_l1)
        tf.losses.add_loss(weights[0] * loss_depth_l1)
        
        # finding scale invarient gradients of prediction and gt using different params
        inp_sig = get_multi_sig_list(inp, sig_params_list)
        gt_sig = get_multi_sig_list(gt, sig_params_list)

        for ind in range(len(inp_sig)):
            loss_depth2_sig = mean_l1_loss_nonfinite(inp_sig[ind], gt_sig[ind])
            tf.summary.scalar('loss_depth2_sig'+str(ind), loss_depth2_sig)
            tf.losses.add_loss(weights[1] * loss_depth2_sig)
        return tf.losses.get_total_loss()


#################
##helpers
#################

def replace_nonfinite(x):
    mask = tf.is_finite(x)
    x_clean = tf.where(mask, x, tf.zeros_like(x))
    return x_clean

def replace_nonfinite_loss(diff, gt):
    mask = tf.logical_and(tf.is_finite(diff), tf.greater(gt, 0.))
    x_clean = tf.where(mask, diff, tf.zeros_like(diff))
    return x_clean

sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 1e-9}

sig_params_list = [{'deltas':[1,], 'weights':[1,], 'epsilon': 1e-9},
                   {'deltas':[2,], 'weights':[1,], 'epsilon': 1e-9},
                   {'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},
                   ]

def get_multi_sig_list(inp, sig_param_list):
    """ Get multi sig with the pattern in the sig_param_list, result is concat in axis=1
    
        inp: Tensor
        sig_param_list: a list of sig_param dictionary
    """
    sig_list = []
    for sig_param in sig_param_list:
        sig_list.append(sops.scale_invariant_gradient(inp,**sig_param))
    #multi_sig = tf.concat(sig_list,axis=1)
    return sig_list