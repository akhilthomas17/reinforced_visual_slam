#!/usr/bin/env python3
import tensorflow as tf
import lmbspecialops as sops
import numpy as np


#################
##helpers my_losses
#################

def nan_mean(x):
  # x will be a numpy array with the contents of the placeholder below
  return np.nanmean(np.abs(x))


def res_converter_exp_conf(res_gt, k = 1):
    ## plotting mean value to help decide the k
    mean_res = tf.py_func(nan_mean, [res_gt], tf.float32)
    tf.summary.scalar("mean_residual", mean_res)
    ## converting residual to confidence in range (0,1]
    res_gt =  tf.exp(tf.scalar_mul(-k, res_gt))
    return res_gt

def invert_finite_depth(x):
    # mask is true if input is finite and greater than 0. If condition is false, make it invalid (nan)
    mask = tf.logical_and(tf.is_finite(x), tf.greater(x, 0.))
    ix_clean = tf.where(mask, tf.reciprocal(x), tf.fill(x.shape, np.nan))
    return ix_clean

def replace_nonfinite(x):
    mask = tf.is_finite(x)
    x_clean = tf.where(mask, x, tf.zeros_like(x))
    return x_clean

def replace_nonfinite_loss(diff, gt):
    mask = tf.logical_and(tf.is_finite(diff), tf.greater(gt, 0.))
    x_clean = tf.where(mask, diff, tf.zeros_like(diff))
    return x_clean

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