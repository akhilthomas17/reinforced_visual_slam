#!/usr/bin/env python3
from __future__ import division
import tensorflow as tf

######################
# Networks with residual prediction
######################

class NetworkV04Res(object):
  """ 
    Network architecture with sparse convolution bootstrapping for sparse depths.
    It is followed by concatenation with bootstrapped rgb. 
    The concatenated tensor then goes through an encoder-decoder network
    All convolutions are done with filter size  = 3.
  """
  def __init__(self):
    super(NetworkV04Res, self).__init__()
    self.initializer = initializer_none

  def bootstrapper_block_rgb(self, x, start_dim=8, num=2, name='rgb_bootstrap', data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii), kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii)+'.1', kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def bootstrapper_block_sparse(self, x, mask, start_dim=8, num=2, name='sparse_bootstrap', 
    data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii), data_format=data_format, initializer=kernel_init)
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii)+'.1', data_format=data_format, initializer=kernel_init)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        mask = tf.layers.max_pooling2d(mask, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def output_block(self, x, input_dim=8, data_format='channels_first', name='ouput_conv'):
    kernel_init = self.initializer()
    filters = input_dim // 2
    x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", name=name,
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    x = tf.layers.conv2d(x, filters=2, kernel_size=3, padding="same", name=name+'.1',
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, inputs, training=False, data_format='channels_first'):
    with tf.variable_scope('depth_fusion_model'):
      rgb, sparse_depth, sparse_depth_var = inputs
      mask = create_mask(sparse_depth)
      if data_format=='channels_last':
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], -1, name='sparse_concat')
        mask_nhwc = mask
      else:
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], 1, name='sparse_concat')
        mask_nhwc = tf.transpose(mask, [0,2,3,1])
      tf.summary.image('sparse_conv_mask', mask_nhwc, max_outputs=1)
      x1 = self.bootstrapper_block_rgb(rgb, data_format=data_format)
      x2 = self.bootstrapper_block_sparse(sparse_depth_conc, mask, data_format=data_format)
      if data_format=='channels_last':
        x = tf.concat([x1, x2], -1)
      else:
        x = tf.concat([x1, x2], 1)
      x = encoder_block(x, input_dim=32, num=2, data_format=data_format, initializer=self.initializer)
      x = tf.layers.conv2d(x, filters=128, kernel_size=3, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=self.initializer(), padding="same", data_format=data_format)
      x = decoder_block_small(x, input_dim=128, num=4, data_format=data_format, initializer=self.initializer)
      x = self.output_block(x, data_format=data_format)
      return x

class NetworkV03Res(object):
  """ 
  Network architecture with sparse convolution bootstrapping for sparse depths.
  It is followed by concatenation with bootstrapped rgb. 
  The concatenated tensor then goes through an encoder-decoder network
  All convolutions are done with filter size  = 3.
  """
  def __init__(self):
    super(NetworkV03Res, self).__init__()
    self.initializer = initializer_none

  def bootstrapper_block_rgb(self, x, start_dim=8, num=2, name='rgb_bootstrap', data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii), kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii)+'.1', kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def bootstrapper_block_sparse(self, x, mask, start_dim=8, num=2, name='sparse_bootstrap', 
    data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii), data_format=data_format, initializer=kernel_init)
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii)+'.1', data_format=data_format, initializer=kernel_init)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        mask = tf.layers.max_pooling2d(mask, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def output_block(self, x, input_dim=8, data_format='channels_first', name='ouput_conv'):
    kernel_init = self.initializer()
    filters = input_dim // 2
    x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", name=name,
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    x = tf.layers.conv2d(x, filters=2, kernel_size=3, padding="same", name=name+'.1',
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, inputs, training=False, data_format='channels_first'):
    with tf.variable_scope('depth_fusion_model'):
      rgb, sparse_depth, sparse_depth_var = inputs
      mask = create_mask(sparse_depth)
      if data_format=='channels_last':
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], -1, name='sparse_concat')
        mask_nhwc = mask
      else:
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], 1, name='sparse_concat')
        mask_nhwc = tf.transpose(mask, [0,2,3,1])
      tf.summary.image('sparse_conv_mask', mask_nhwc, max_outputs=1)
      x1 = self.bootstrapper_block_rgb(rgb, data_format=data_format)
      x2 = self.bootstrapper_block_sparse(sparse_depth_conc, mask, data_format=data_format)
      if data_format=='channels_last':
        x = tf.concat([x1, x2], -1)
      else:
        x = tf.concat([x1, x2], 1)
      x = encoder_block(x, input_dim=32, num=2, data_format=data_format, initializer=self.initializer)
      x = tf.layers.conv2d(x, filters=128, kernel_size=3, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=self.initializer(), padding="same", data_format=data_format)
      x = decoder_block(x, input_dim=128, num=4, data_format=data_format, initializer=self.initializer)
      x = self.output_block(x, data_format=data_format)
      return x



######################
# Networks without error prediction
######################

class NetworkV01(object):
  """ 
  Network architecture with sparse convolution bootstrapping for sparse depths.
  It is followed by concatenation with bootstrapped rgb. 
  The concatenated tensor then goes through an encoder-decoder network
  """
  def __init__(self):
    super(NetworkV01, self).__init__()
    self.initializer = initializer_none

  def bootstrapper_block(self, x, start_dim=4, name='rgb_bootstrap', data_format='channels_last'):
    with tf.variable_scope(name):
      filters = start_dim
      for k in [11, 9, 5]:
        kernel_init = self.initializer()
        x = tf.layers.conv2d(x, filters=filters, kernel_size=k, padding="same", activation=tf.nn.relu, 
          name=name+str(k), kernel_initializer=kernel_init, data_format=data_format)
        filters = filters*2
      return x

  def bootstrapper_block_sparse(self, x, mask, start_dim=4, name='sparse_bootstrap', 
    data_format='channels_last'):
    with tf.variable_scope(name):
      filters = start_dim
      for k in [11, 9, 5]:
        kernel_init = self.initializer()
        x, mask = sparse_conv2d(x, mask, kernel_size=k, num_filters=filters, 
          name='sparse_conv'+str(k), data_format=data_format, initializer=kernel_init)
        filters = filters*2
      return x

  def ouput_block(self, x, data_format='channels_last'):
    kernel_init = self.initializer()
    x = tf.layers.conv2d(x, filters=1, kernel_size=5, padding="same",
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x


  def __call__(self, inputs, training=False, data_format='channels_last'):
    with tf.variable_scope('depth_fusion_model'):
      rgb, sparse_depth, sparse_depth_var = inputs
      #tf.summary.scalar('batch_size', tf.shape(rgb)[0])
      mask = create_mask(sparse_depth)
      if data_format=='channels_last':
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], -1, name='sparse_concat')
        mask_nhwc = mask
      else:
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], 1, name='sparse_concat')
        mask_nhwc = tf.transpose(mask, [0,2,3,1])
      tf.summary.image('sparse conv mask', mask_nhwc, max_outputs=1)
      x1 = self.bootstrapper_block(rgb, data_format=data_format)
      x2 = self.bootstrapper_block_sparse(sparse_depth_conc, mask, data_format=data_format)
      if data_format=='channels_last':
        x = tf.concat([x1, x2], -1)
        #x = tf.keras.layers.concatenate( [x1,x2] )
      else:
        x = tf.concat([x1, x2], 1)
      x = encoder_decoder_block(x, input_dim=32, num=4, initializer=self.initializer, 
        data_format=data_format)
      if training:
        x = self.dropout(x)
      x = self.ouput_block(x, data_format=data_format)
      return x

class NetworkV02(object):
  """ 
  Network architecture with sparse convolution bootstrapping for sparse depths.
  The layers of this bootstrapping is the same as that "sparse invariant cnn" paper.
  It is followed by concatenation with bootstrapped rgb. 
  The concatenated tensor then goes through an encoder-decoder network
  """
  def __init__(self):
    super(NetworkV02, self).__init__()
    self.initializer = initializer_none

  def bootstrapper_block(self, x, start_dim=4, name='rgb_bootstrap', data_format='channels_last'):
    with tf.variable_scope(name):
      filters = start_dim
      cnt = 0
      for k in [11, 7, 5]:
        kernel_init = self.initializer()
        x = tf.layers.conv2d(x, filters=filters, kernel_size=k, padding="same", activation=tf.nn.relu, 
          name=name+str(k)+str(cnt), kernel_initializer=kernel_init, data_format=data_format)
        filters = filters*2
        cnt += 1
      return x

  def bootstrapper_block_sparse(self, x, mask, filters=16, name='sparse_bootstrap', 
    data_format='channels_last'):
    with tf.variable_scope(name):
      cnt = 0
      for k in [11, 7, 5, 3, 3]:
        kernel_init = self.initializer()
        x, mask = sparse_conv2d(x, mask, kernel_size=k, num_filters=filters, 
          name='sparse_conv'+str(k)+str(cnt), data_format=data_format, initializer=kernel_init)
        cnt += 1
      return x

  def output_block(self, x, data_format='channels_last'):
    kernel_init = self.initializer()
    x = tf.layers.conv2d(x, filters=1, kernel_size=5, padding="same",
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, inputs, training=False, data_format='channels_last'):
    with tf.variable_scope('depth_fusion_model'):
      rgb, sparse_depth, sparse_depth_var = inputs
      mask = create_mask(sparse_depth)
      if data_format=='channels_last':
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], -1, name='sparse_concat')
        mask_nhwc = mask
      else:
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], 1, name='sparse_concat')
        mask_nhwc = tf.transpose(mask, [0,2,3,1])
      tf.summary.image('sparse_conv_mask', mask_nhwc, max_outputs=1)
      x1 = self.bootstrapper_block(rgb, data_format=data_format)
      x2 = self.bootstrapper_block_sparse(sparse_depth_conc, mask, data_format=data_format)
      if data_format=='channels_last':
        x = tf.concat([x1, x2], -1)
        #x = tf.keras.layers.concatenate( [x1,x2] )
      else:
        x = tf.concat([x1, x2], 1)
      x = encoder_decoder_block(x, input_dim=32, num=4, initializer=self.initializer, 
        data_format=data_format)
      if training:
        x = self.dropout(x)
      x = self.output_block(x, data_format=data_format)
      return x

class NetworkV03(object):
  """ 
  Network architecture with sparse convolution bootstrapping for sparse depths.
  It is followed by concatenation with bootstrapped rgb. 
  The concatenated tensor then goes through an encoder-decoder network
  All convolutions are done with filter size  = 3.
  """
  def __init__(self):
    super(NetworkV03, self).__init__()
    self.initializer = initializer_none

  def bootstrapper_block_rgb(self, x, start_dim=8, num=2, name='rgb_bootstrap', data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii), kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii)+'.1', kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def bootstrapper_block_sparse(self, x, mask, start_dim=8, num=2, name='sparse_bootstrap', 
    data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii), data_format=data_format, initializer=kernel_init)
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii)+'.1', data_format=data_format, initializer=kernel_init)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        mask = tf.layers.max_pooling2d(mask, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def output_block(self, x, input_dim=8, data_format='channels_first', name='ouput_conv'):
    kernel_init = self.initializer()
    filters = input_dim // 2
    x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", name=name,
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    x = tf.layers.conv2d(x, filters=1, kernel_size=3, padding="same", name=name+'.1',
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, inputs, training=False, data_format='channels_first'):
    with tf.variable_scope('depth_fusion_model'):
      rgb, sparse_depth, sparse_depth_var = inputs
      mask = create_mask(sparse_depth)
      if data_format=='channels_last':
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], -1, name='sparse_concat')
        mask_nhwc = mask
      else:
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], 1, name='sparse_concat')
        mask_nhwc = tf.transpose(mask, [0,2,3,1])
      tf.summary.image('sparse_conv_mask', mask_nhwc, max_outputs=1)
      x1 = self.bootstrapper_block_rgb(rgb, data_format=data_format)
      x2 = self.bootstrapper_block_sparse(sparse_depth_conc, mask, data_format=data_format)
      if data_format=='channels_last':
        x = tf.concat([x1, x2], -1)
      else:
        x = tf.concat([x1, x2], 1)
      x = encoder_block(x, input_dim=32, num=2, data_format=data_format, initializer=self.initializer)
      x = tf.layers.conv2d(x, filters=128, kernel_size=3, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=self.initializer(), padding="same", data_format=data_format)
      x = decoder_block(x, input_dim=128, num=4, data_format=data_format, initializer=self.initializer)
      x = self.output_block(x, data_format=data_format)
      return x

class NetworkV04(object):
  """ 
  Network architecture with sparse convolution bootstrapping for sparse depths.
  It is followed by concatenation with bootstrapped rgb. 
  The concatenated tensor then goes through an encoder-decoder network
  All convolutions are done with filter size  = 3.
  """
  def __init__(self):
    super(NetworkV04, self).__init__()
    self.initializer = initializer_none

  def bootstrapper_block_rgb(self, x, start_dim=8, num=2, name='rgb_bootstrap', data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii), kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", activation=tf.nn.relu, 
          name=name+str(ii)+'.1', kernel_initializer=kernel_init, data_format=data_format)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def bootstrapper_block_sparse(self, x, mask, start_dim=8, num=2, name='sparse_bootstrap', 
    data_format='channels_first'):
    with tf.variable_scope(name):
      filters = start_dim
      for ii in range(num):
        kernel_init = self.initializer()
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii), data_format=data_format, initializer=kernel_init)
        x, mask = sparse_conv2d(x, mask, kernel_size=3, num_filters=filters, 
          name='sparse_conv'+str(ii)+'.1', data_format=data_format, initializer=kernel_init)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
        mask = tf.layers.max_pooling2d(mask, pool_size=2, strides=2, padding="same", data_format=data_format)
        filters = filters*2
      return x

  def output_block(self, x, input_dim=8, data_format='channels_first', name='ouput_conv'):
    kernel_init = self.initializer()
    filters = input_dim // 2
    x = tf.layers.conv2d(x, filters=filters, kernel_size=3, padding="same", name=name,
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    x = tf.layers.conv2d(x, filters=1, kernel_size=3, padding="same", name=name+'.1',
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, inputs, training=False, data_format='channels_first'):
    with tf.variable_scope('depth_fusion_model'):
      rgb, sparse_depth, sparse_depth_var = inputs
      mask = create_mask(sparse_depth)
      if data_format=='channels_last':
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], -1, name='sparse_concat')
        mask_nhwc = mask
      else:
        sparse_depth_conc = tf.concat([sparse_depth, sparse_depth_var], 1, name='sparse_concat')
        mask_nhwc = tf.transpose(mask, [0,2,3,1])
      tf.summary.image('sparse_conv_mask', mask_nhwc, max_outputs=1)
      x1 = self.bootstrapper_block_rgb(rgb, data_format=data_format)
      x2 = self.bootstrapper_block_sparse(sparse_depth_conc, mask, data_format=data_format)
      if data_format=='channels_last':
        x = tf.concat([x1, x2], -1)
      else:
        x = tf.concat([x1, x2], 1)
      x = encoder_block(x, input_dim=32, num=2, data_format=data_format, initializer=self.initializer)
      x = tf.layers.conv2d(x, filters=128, kernel_size=3, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=self.initializer(), padding="same", data_format=data_format)
      x = decoder_block(x, input_dim=128, num=4, data_format=data_format, initializer=self.initializer)
      x = self.output_block(x, data_format=data_format)
      return x

##########################
# single image depth prediction networks
##########################

class NetworkV0(object):
  """ 
  Network architecture for single image depth prediction.
  All convolutions are done with filter size  = 5.
  """
  def __init__(self):
    super(NetworkV0, self).__init__()
    self.initializer = initializer_none

  def output_block(self, x, input_dim=8, data_format='channels_first', name='ouput_conv'):
    kernel_init = self.initializer()
    filters = input_dim // 2
    x = tf.layers.conv2d(x, filters=filters, kernel_size=5, padding="same", name=name,
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    x = tf.layers.conv2d(x, filters=1, kernel_size=5, padding="same", name=name+'.1',
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, rgb, training=False, data_format='channels_first'):
    with tf.variable_scope('single_image_depth_prediction_model'):
      x = encoder_block(rgb, input_dim=4, num=4, k=5, data_format=data_format, initializer=self.initializer)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=self.initializer(), padding="same", data_format=data_format)
      x = decoder_block_small(x, input_dim=64, num=4, k=5, data_format=data_format, initializer=self.initializer)
      x = self.output_block(x, data_format=data_format)
      return x

class NetworkV0l(object):
  """ 
  Network architecture for single image depth prediction.
  All convolutions are done with filter size  = 5.
  """
  def __init__(self):
    super(NetworkV0l, self).__init__()
    self.initializer = initializer_none

  def output_block(self, x, input_dim=8, data_format='channels_first', name='ouput_conv'):
    kernel_init = self.initializer()
    filters = input_dim // 2
    x = tf.layers.conv2d(x, filters=filters, kernel_size=5, padding="same", name=name,
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    x = tf.layers.conv2d(x, filters=1, kernel_size=5, padding="same", name=name+'.1',
     activation=tf.nn.relu, kernel_initializer=kernel_init, data_format=data_format)
    return x

  def __call__(self, rgb, training=False, data_format='channels_first'):
    with tf.variable_scope('single_image_depth_prediction_model'):
      x = encoder_block(rgb, input_dim=4, num=5, k=5, data_format=data_format, initializer=self.initializer)
      x = tf.layers.conv2d(x, filters=64, kernel_size=3, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=self.initializer(), padding="same", data_format=data_format)
      x = decoder_block_small(x, input_dim=64, num=5, k=5, data_format=data_format, initializer=self.initializer)
      x = self.output_block(x, input_dim=4, data_format=data_format)
      return x


##########################
# helper functions
##########################

def initializer_normal():
  return tf.random_normal_initializer(0, 1)

def initializer_uniform():
  return tf.random_uniform_initializer(-1,1)

def initializer_truncated_normal():
  return  tf.truncated_normal_initializer(0, 1)

def initializer_none():
  return  None

def sparse_conv2d(x, mask, kernel_size, num_filters, stride=1, name='sparse_conv', 
  initializer=None, data_format='channels_last', epsilon=1e-6):
  with tf.variable_scope(name):
    dtype = x.dtype
    stride_ = [stride, stride, stride, stride]
    x_out = tf.multiply(x, mask)
    if data_format == 'channels_last':
      in_channels = x.shape.as_list()[-1]
      data_fmt = "NHWC"
    else:
      in_channels = x.shape.as_list()[1]
      data_fmt = "NCHW"
    kernel_shape = [kernel_size, kernel_size, in_channels, num_filters]
    w = tf.get_variable("weights", kernel_shape, initializer=initializer, dtype=dtype)
    # performing (x conv kernel)
    x_out = tf.nn.conv2d(x_out, w, data_format=data_fmt, padding="SAME",
      strides=stride_, name="conv_01")
    # making a sum convolution of x_mask
    sum_const = tf.ones([kernel_size, kernel_size, 1, 1], dtype)
    mask_y = tf.nn.conv2d(mask, sum_const, strides=stride_, padding="SAME", 
      data_format=data_fmt, name="conv_01")
    # normalizing x_out with the mask convolution
    mask_y =  tf.reciprocal(tf.add(mask_y, epsilon))
    x_out = tf.multiply(x_out, mask_y)
    # adding bias
    b = tf.get_variable("bias", [num_filters])
    x_out = tf.nn.bias_add(x_out, b, data_format=data_fmt)
    # Max pooling on mask
    mask_out = tf.layers.max_pooling2d(mask, kernel_size, stride, padding="same",
      data_format=data_format, name="max_pool")
    return x_out, mask_out

def create_mask(x):
  return tf.to_float(tf.greater(x, 0.))

def decoder_block_small(x, input_dim=256, num=4, k=3, basename='comb_dec_', initializer=initializer_none,
  data_format='channels_first'):
  filters = input_dim
  for ii in range(num):
    name = basename +  str(ii)
    kernel_init = initializer()
    filters = int(filters/2)
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=k, activation=tf.nn.relu, 
      name=name, kernel_initializer=kernel_init, strides=2, padding="same", data_format=data_format)
  return x

def decoder_block(x, input_dim=256, num=4, k=3, basename='comb_dec_', initializer=initializer_none,
  data_format='channels_first'):
  filters = input_dim
  for ii in range(num):
    name = basename +  str(ii)
    kernel_init = initializer()
    filters = int(filters/2)
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=k, activation=tf.nn.relu, 
      name=name, kernel_initializer=kernel_init, strides=1, padding="same", data_format=data_format)
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=k, activation=tf.nn.relu, 
      name=name+'.1', kernel_initializer=kernel_init, strides=2, padding="same", data_format=data_format)
  return x

def encoder_block(x, input_dim=16, num=2, k=3, basename='comb_enc_', initializer=initializer_none,
  data_format='channels_first'):
  filters = input_dim
  for ii in range(num):
    filters = filters*2
    name = basename +  str(ii)
    kernel_init = initializer()
    x = tf.layers.conv2d(x, filters=filters, kernel_size=k, activation=tf.nn.relu, 
      name=name, kernel_initializer=kernel_init, padding="same", data_format=data_format)
    x = tf.layers.conv2d(x, filters=filters, kernel_size=k, activation=tf.nn.relu, 
      name=name+'.1', kernel_initializer=kernel_init, padding="same", data_format=data_format)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
  return x

def encoder_decoder_block(x, input_dim=16, num=6, basename='comb', initializer=initializer_normal, 
  data_format='channels_last'):
  filters = input_dim
  k1 = 5
  k2 = 3
  # encoder networks
  for ii in range(num):
    if ii > 3:
      k1 = 3
      k2 = 1
    filters = filters*2
    name = basename + '_enc' +  str(ii)
    kernel_init = initializer()
    x = tf.layers.conv2d(x, filters=filters, kernel_size=k1, activation=tf.nn.relu, 
      name=name, kernel_initializer=kernel_init, padding="same", data_format=data_format)
    x = tf.layers.conv2d(x, filters=filters, kernel_size=k2, activation=tf.nn.relu, 
      name=name+'_2', kernel_initializer=kernel_init, padding="same", data_format=data_format)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same", data_format=data_format)
  # decoder network
  x = tf.layers.conv2d(x, filters=filters, kernel_size=1, activation=tf.nn.relu, 
      name='conv_low', kernel_initializer=kernel_init, padding="same", data_format=data_format)
  for ii in range(num):
    name = basename + '_dec' +  str(ii)
    kernel_init = initializer()
    filters = int(filters/2)
    x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=5, activation=tf.nn.relu, 
      name=name, kernel_initializer=kernel_init, strides=2, padding="same", data_format=data_format)
  return x