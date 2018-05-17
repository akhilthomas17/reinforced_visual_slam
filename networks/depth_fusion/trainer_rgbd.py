#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from absl import app as absl_app
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug

import sys
import os
sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')
from net.my_networks import *
from net.my_losses import *

feature_names = [
'rgb',
'sparseInverseDepth',
'sparseInverseDepthVariance'
]

FLAGS = {
  'multi_gpu': True,
  'batch_size':12,
  'prefetch_buffer_size': 24,
  'num_parallel_calls': 24,
  'num_epochs':50,
  'learning_rate':0.0004,
  'data_format': "channels_first"
}

def parse_load_augment(filename_line):
  filenames = tf.decode_csv(filename_line, [[''], [''], [''], ['']])
  indices = tf.data.Dataset.from_tensors([0, 1, 2, 3])
  image_files = tf.data.Dataset.zip((filenames, indices))
  images = images_files.map(map_func=load_image, num_parallel_calls=4)
  # add augmentation step here
  depth_gt = images.range(0, 1)
  train_images = images.range(1, 4)
  d = dict(zip(feature_names, train_images)), depth_gt
  return d

def invert_clean(x, threshold=0.01):
    mask =  tf.logical_and(tf.is_finite(x), tf.greater(x,threshold))
    x_clean = tf.where(mask, tf.reciprocal(x), tf.zeros_like(x))
    return x_clean

def load_image(filename, index, data_format=FLAGS['data_format'], width=320, height=240, 
  resize=True, basepath = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/train/"):
  image_string = tf.read_file(basepath + filename)
  if index == 0:
    # current image is depth groundtruth
    image_decoded = tf.image.decode_png(image_string, dtype=tf.uint16)
    image_decoded = tf.reshape(image_decoded, [480, 640, 1])
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = tf.scalar_mul(0.0002, image_decoded)
    # converting (and clean) depth to idepth and removing NaN's and Inf's
    image_decoded = invert_clean(image_decoded)

  elif index == 1:
    # current image is rgb
    image_decoded = tf.image.decode_png(image_string)
    image_decoded = tf.reshape(image_decoded, [480, 640, 3])
    image_decoded = tf.cast(image_decoded, tf.float32)
    # converting rgb to [0,1] from [0,255]
    image_decoded = tf.scalar_mul(0.003926, image_decoded)

  else:
    # for both the sparse depth and variance
    image_decoded = tf.decode_raw(image_string, tf.half)
    image_decoded = tf.reshape(image_decoded, [480, 640, 1])
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = replace_nonfinite(image_decoded)

  if resize == True:
    image_decoded = tf.image.resize_images(image_decoded, [height, width], align_corners=True)
  if data_format == 'channels_first':
    image_decoded = tf.transpose(image_decoded,[2,0,1])
  return image


def parse_n_load(filename_line, data_format=FLAGS['data_format'], width=320, height=240):
  filenames = tf.decode_csv(filename_line, [[''], [''], [''], ['']])
  images = []
  basepath = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/train/"
  for cnt in range(4):
      image_string = tf.read_file(basepath + filenames[cnt])
      if cnt < 2:
        if cnt == 0:
          image_decoded = tf.image.decode_png(image_string, dtype=tf.uint16)
          image_decoded = tf.reshape(image_decoded, [480, 640, 1])
        else:
          image_decoded = tf.image.decode_png(image_string)
          image_decoded = tf.reshape(image_decoded, [480, 640, 3])
        image_decoded = tf.cast(image_decoded, tf.float32)
      else:
        image_decoded = tf.decode_raw(image_string, tf.half)
        image_decoded = tf.reshape(image_decoded, [480, 640, 1])
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded = replace_nonfinite(image_decoded)
      image_decoded = tf.image.resize_images(image_decoded, [height, width], 
        align_corners=True)
      if data_format == 'channels_first':
        image_decoded = tf.transpose(image_decoded,[2,0,1])
      images.append(image_decoded)
          
  depth_gt = tf.scalar_mul(0.0002, images[0])
  # converting depth to idepth and removing NaN's and Inf's
  #idepth_gt = tf.reciprocal(depth_gt)
  #idepth_gt_clean = replace_nonfinite(idepth_gt)
  idepth_gt_clean = invert_clean(depth_gt)
  del images[0]
  # converting rgb to [0,1] from [0,255]
  images[0] = tf.scalar_mul(0.003926, images[0])
  #d = dict(zip(feature_names, images)), idepth_gt_clean
  d = dict(zip(feature_names, images)), depth_gt
  return d

def dataset_shuffler(training_file):
  filename_records = tf.data.TextLineDataset(training_file)
  dataset = filename_records.shuffle(buffer_size=50000)
  return dataset

def input_fn(dataset, shuffle=False):
  dataset = dataset.map(map_func=parse_n_load, num_parallel_calls=FLAGS['num_parallel_calls'])
  #dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_n_load, num_parallel_calls=FLAGS.num_parallel_calls,
   #                                           batch_size=FLAGS.batch_size))
  dataset = dataset.repeat(FLAGS['num_epochs'])
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(batch_size=FLAGS['batch_size'])
  dataset = dataset.prefetch(buffer_size=FLAGS['prefetch_buffer_size'])
  iterator = dataset.make_one_shot_iterator()
  features, label = iterator.get_next()
  return features, label

def train_input_fn():
  training_file = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/train/rgbd_lsdDepth_train.txt"
  #basepath = os.path.dirname(training_file)
  dataset = dataset_shuffler(training_file)
  #dataset = tf.data.TextLineDataset(training_file)
  features, label = input_fn(dataset, shuffle=False)
  return features, label

def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.
  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.
  Args:
    batch_size: the number of examples processed in each training batch.
  Raises:
    ValueError: if no GPUs are found, or selected batch_size is invalid.
  """
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)

def model_fn(features, labels, mode, params):
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

  model = NetworkV01()

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
    learning_rate_base = FLAGS['learning_rate']
    learning_rate = tf.train.exponential_decay(learning_rate_base, tf.train.get_or_create_global_step(), 
      decay_rate=0.8, decay_steps=700000, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if params.get('multi_gpu'):
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    
    depth = model(inputs, data_format=params.get('data_format'))
    loss = pointwise_l2_loss_clean(inp=depth, gt=labels, data_format=params.get('data_format'))
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
    tf.summary.scalar('pointwise_l2_loss_clean', loss)

    logging_hook = tf.train.LoggingTensorHook({"pointwise_l2_loss_clean" : loss}, every_n_iter=10)
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, 
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()),
      training_hooks=[logging_hook])

  ######
  #eval
  ######
  if mode == tf.estimator.ModeKeys.EVAL:
    depth = model(inputs, params.get('data_format'))
    loss = pointwise_l2_loss_clean(inp=labels, gt=depth, data_format=params.get('data_format'))
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss)



def run_trainer(model_dir_name, training_steps):
  """Run training and eval loop for lsd depth fusion network.
  """
  model_function = model_fn
  hooks = [tf_debug.LocalCLIDebugHook()]
  #hooks = [tf_debug.TensorBoardDebugHook("localhost:6064")] (tf 1.5+)

  if FLAGS['multi_gpu']:
    validate_batch_size_for_multi_gpu(FLAGS['batch_size'])

    # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    # and (2) wrap the optimizer. The first happens here, and (2) happens
    # in the model_fn itself when the optimizer is defined.
    model_function = tf.contrib.estimator.replicate_model_fn(
        model_fn, loss_reduction=tf.losses.Reduction.MEAN)

  model_base_dir = "/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion/training"
  model_dir = os.path.join(model_base_dir, model_dir_name)

  lsd_depth_fuser = tf.estimator.Estimator(
    model_fn=model_function,
    model_dir=model_dir,
    params={
    'data_format': FLAGS['data_format'],
    'multi_gpu': FLAGS['multi_gpu']
    })

  # Train and evaluate model.
  #for _ in range(FLAGS['num_epochs'] // flags_obj.epochs_between_evals):
  lsd_depth_fuser.train(input_fn=train_input_fn, steps=training_steps)
  #eval_results = lsd_depth_fuser.evaluate(input_fn=eval_input_fn)
  #print('\nEvaluation results:\n\t%s\n' % eval_results)

  #if model_helpers.past_stop_threshold(flags_obj.stop_threshold,
   #                                    eval_results['loss']):
    #break

  # Export the model
  #if flags_obj.export_dir is not None:
   # image = tf.placeholder(tf.float32, [None, 28, 28])
    #input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
     #   'image': image,
    #})
#mnist_classifier.export_savedmodel(flags_obj.export_dir, input_fn)
def dataset_len(dataset_loc):
    with open(dataset_loc) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
  parser = argparse.ArgumentParser(description=( "Train LSDDepth fusion network using the  custom estimator."))
  parser.add_argument("--model_name", type=str, required=True, help="Specify name of model (will be used as ouput dir_name)")
  parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
  args = parser.parse_args()

  training_file = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/train/rgbd_lsdDepth_train.txt"
  data_len = dataset_len(training_file)
  steps_per_epoch = data_len // FLAGS['batch_size']
  num_steps = steps_per_epoch * args.num_epochs

  FLAGS['num_epochs'] = args.num_epochs

  print("Number of training steps:", num_steps)
  
  run_trainer(args.model_name, num_steps)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
  sys.exit()
  #absl_app.run(main)