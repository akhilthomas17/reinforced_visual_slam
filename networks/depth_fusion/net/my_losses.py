#!/usr/bin/env python3
import sys
sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')
from net.helpers import *

sig_params_list = [{'deltas':[1,], 'weights':[1,], 'epsilon': 1e-9},
                   {'deltas':[2,], 'weights':[1,], 'epsilon': 1e-9},
                   {'deltas':[4,], 'weights':[1,], 'epsilon': 1e-9},
                   ]

def pointwise_l1_loss_sig_l1_loss_res_l1_loss(pred, gt, data_format='channels_first', weights=[0.4, 0.6, 0.5], 
    res_converter=res_converter_exp_conf):
    """ Computes pointwise l1 loss and sig loss (gradient) of depth, and loss of residual
    The input tensors must use the format NCHW.
    This loss ignores nan values. 
    Total loss is weighted sum of both normalized by the number of pixels.
    """
    with tf.name_scope('pointwise_l1_loss_sig_l1_loss_res_l1_loss'):
        if data_format=='channels_first':
            depth = pred[:,0:1,:,:]
            res = pred[:,1:,:,:]
        else:
            depth = pred[:,:,:,0:1]
            res = pred[:,:,:,1:]
        loss_depth_l1 = mean_l1_loss_nonfinite(depth, gt)
        tf.summary.scalar('loss_depth_l1', loss_depth_l1)
        tf.losses.add_loss(weights[0] * loss_depth_l1)
        
        # finding scale invarient gradients of prediction and gt using different params
        print('depth.shape', depth.shape)
        inp_sig = get_multi_sig_list(depth, sig_params_list)
        gt_sig = get_multi_sig_list(gt, sig_params_list)
        print('inp_sig[0].shape', inp_sig[0].shape)


        for ind in range(len(inp_sig)):
            # adding the sig images tp tensorboard
            if data_format=='channels_last':
                inp_sig_nhwc = sops.replace_nonfinite(inp_sig[ind])
                gt_sig_nhwc = sops.replace_nonfinite(gt_sig[ind])
            else:
                inp_sig_nhwc = sops.replace_nonfinite(tf.transpose(inp_sig[ind], [0,2,3,1]))
                gt_sig_nhwc = sops.replace_nonfinite(tf.transpose(gt_sig[ind], [0,2,3,1]))
            sigs_nhwc = tf.concat([inp_sig_nhwc[0:1,:,:,0:1], gt_sig_nhwc[0:1,:,:,0:1], 
                inp_sig_nhwc[0:1,:,:,1:], gt_sig_nhwc[0:1,:,:,1:]], 0)
            print('sigs_nhwc.shape', sigs_nhwc.shape)
            tf.summary.image('sigs'+str(ind), sigs_nhwc, max_outputs=4)
            loss_depth2_sig = mean_l1_loss_nonfinite(inp_sig[ind], gt_sig[ind])
            tf.summary.scalar('loss_depth2_sig'+str(ind), loss_depth2_sig)
            tf.losses.add_loss(weights[1] * loss_depth2_sig)

        # finding loss for depth residual prediction
        res_gt = gt - depth
        res_gt = res_converter(res_gt)
        # writing residuals to tensorboard
        if data_format=='channels_last':
            res_gt_nhwc = sops.replace_nonfinite(res_gt)
            res_pr_nhwc = res
        else:
            res_gt_nhwc = sops.replace_nonfinite(tf.transpose(res_gt, [0,2,3,1]))
            res_pr_nhwc = tf.transpose(res, [0,2,3,1])
        res_nhwc = tf.concat([res_pr_nhwc[0:1, :, :, :], res_gt_nhwc[0:1, :, :, :]], 0)
        tf.summary.image('confidence', res_nhwc, max_outputs=2)
        # calculating loss
        loss_res = mean_l1_loss_nonfinite(res, res_gt)
        tf.summary.scalar('loss_res', loss_res)
        tf.losses.add_loss(weights[2] * loss_res)

        return tf.losses.get_total_loss()

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


def pointwise_l1_loss_sig_l1_loss(inp, gt, data_format='channels_first', weights=[0.4, 0.6]):
    """ Computes pointwise l1 loss and sig loss (gradient).
    The input tensors must use the format NCHW. 
    This loss ignores nan values. 
    Total loss is weighted sum of both normalized by the number of pixels.
    """
    with tf.name_scope('pointwise_l1_loss_sig_l1_loss'):
        loss_depth_l1 = mean_l1_loss_nonfinite(inp, gt)
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


