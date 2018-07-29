#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug

import sys
import os
sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')
from net.my_networks import *
from net.my_losses import *
from net.my_models import *

feature_names = [
  'rgb',
  'sparseInverseDepth',
  'sparseInverseDepthVariance'
]

FLAGS = {
  'multi_gpu': False,
  'batch_size':12,
  'prefetch_buffer_size': 12,
  'num_parallel_calls': 12,
  'num_epochs':50,
  'learning_rate':0.0004,
  'data_format': "channels_first",
  'epochs_bw_eval':1
}

augment_props = {
  'max_contrast_factor': 1.1,
  'max_brightness_factor': 0.05,
  'max_scale_factor': 1.2
}

def parse_n_load(filename_line, basepath, data_format=FLAGS['data_format'], augment=True, width=320, height=240):
  filenames = tf.decode_csv(filename_line, [[''], [''], [''], ['']])
  images = []
  for cnt in range(4):
      image_string = tf.read_file(basepath +'/'+ filenames[cnt])
      if cnt < 2:
        if cnt == 0:
          image_decoded = tf.image.decode_png(image_string, dtype=tf.uint16)
          image_decoded = tf.reshape(image_decoded, [480, 640, 1])
          image_decoded = tf.cast(image_decoded, tf.float32)
          image_decoded = tf.scalar_mul(0.0002, image_decoded)
          # converting depth to idepth but keeping NaN's
          image_decoded = invert_finite_depth(image_decoded)
        else:
          image_decoded = tf.image.decode_png(image_string)
          image_decoded = tf.reshape(image_decoded, [480, 640, 3])
          image_decoded = tf.cast(image_decoded, tf.float32)
          # converting rgb to [0,1] from [0,255]
          image_decoded = tf.divide(image_decoded, 255)
      else:
        image_decoded = tf.decode_raw(image_string, tf.half)
        image_decoded = tf.reshape(image_decoded, [480, 640, 1])
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded = replace_nonfinite(image_decoded)

      image_decoded = tf.image.resize_images(image_decoded, [height, width], 
        align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      if data_format == 'channels_first':
        image_decoded = tf.transpose(image_decoded,[2,0,1])
      images.append(image_decoded)       
  idepth_gt_clean = images[0]
  del images[0]
  if augment:
    images, idepth_gt_clean = random_augmentations(images, idepth_gt_clean)
  d = dict(zip(feature_names, images)), idepth_gt_clean
  return d

def random_augmentations(images, idepth_gt):
  rand_nums = np.random.rand(2)
  rand_bools = rand_nums < 0.3
  rand_scale = np.random.uniform(1./augment_props['max_scale_factor'],
    augment_props['max_scale_factor'])
  # rand_bools = np.random.choice([True, False], 2)
  
  # brightness and contrast transform only for rgb
  images[0] = tf.image.random_brightness(images[0], max_delta=augment_props['max_brightness_factor'])
  images[0] = tf.image.random_contrast(images[0], upper=augment_props['max_contrast_factor'], 
      lower=1./augment_props['max_contrast_factor'])

  # random scale operation for sparse depth inputs
  for ii in [1,2]:
    images[ii] = tf.scalar_mul(rand_scale, images[ii])

  # input augmentations (for all inputs)
  for ii in range(len(images)):
    # common augmentations
    if rand_bools[0]:
      images[ii] = tf.image.flip_left_right(images[ii])
    if rand_bools[1]:
      images[ii] = tf.image.flip_up_down(images[ii])
  
  # modifiying gt for mirroring
  if rand_bools[0]:
    idepth_gt = tf.image.flip_left_right(idepth_gt)
  if rand_bools[1]:
    idepth_gt = tf.image.flip_up_down(idepth_gt)
  return(images, idepth_gt)


def dataset_shuffler(training_file):
  filename_records = tf.data.TextLineDataset(training_file)
  #dataset = filename_records.shuffle(buffer_size=50000)
  dataset = filename_records.apply(tf.contrib.data.shuffle_and_repeat(50000, FLAGS['num_epochs']))
  return dataset

def input_fn(dataset, basename, augment=True, batch_size=FLAGS['batch_size']):
  #dataset = dataset.apply(tf.contrib.data.parallel_interleave(parse_n_extract_images, 
   # cycle_length=FLAGS['num_parallel_calls'], sloppy=True))
  #dataset = dataset.map(map_func=parse_n_load, num_parallel_calls=FLAGS['num_parallel_calls'])
  #dataset = dataset.batch(batch_size=FLAGS['batch_size'])
  dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=lambda filename:parse_n_load(filename, basename, augment=True), 
    num_parallel_batches=2, batch_size=batch_size))
  
  #dataset = dataset.shuffle(buffer_size=10000)

  dataset = dataset.prefetch(buffer_size=FLAGS['prefetch_buffer_size'])
  #iterator = dataset.make_one_shot_iterator()
  #features, label = iterator.get_next()
  #return features, label
  return dataset

def train_input_fn():
  training_file = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/train/rgbd_lsdDepth_train.txt"
  dataset = dataset_shuffler(training_file)
  basename = os.path.dirname(training_file)
  dataset = input_fn(dataset, basename, augment=True)
  return dataset

def test_input_fn():
  test_file = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/test/rgbd_lsdDepth_test.txt"
  dataset = tf.data.TextLineDataset(test_file)
  basename = os.path.dirname(test_file)
  dataset = dataset.map(map_func=lambda filename:parse_n_load(filename, basename, augment=False), 
    num_parallel_calls=FLAGS['num_parallel_calls'])
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(4))
  #dataset = input_fn(dataset, basename, augment=False, batch_size=4)
  return dataset

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



def run_trainer(model_fn, model_dir_name, training_steps, config):
  """Run training and eval loop for lsd depth fusion network.
  """
  model_function = model_fn
  hooks = [tf_debug.LocalCLIDebugHook()]

  if FLAGS['multi_gpu']:
    validate_batch_size_for_multi_gpu(FLAGS['batch_size'])

    # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    # and (2) wrap the optimizer. The first happens here, and (2) happens
    # in the model_fn itself when the optimizer is defined.
    model_function = tf.contrib.estimator.replicate_model_fn(
        model_fn, loss_reduction=tf.losses.Reduction.MEAN)

  model_base_dir = "/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion/training"
  model_dir = os.path.join(model_base_dir, model_dir_name)

  run_config = tf.estimator.RunConfig(session_config=config)

  lsd_depth_fuser = tf.estimator.Estimator(
    model_fn=model_function,
    model_dir=model_dir,
    params={
    'data_format': FLAGS['data_format'],
    'multi_gpu': FLAGS['multi_gpu']
    },
    config=run_config)

  # Train and evaluate model.
  for _ in range(FLAGS['num_epochs'] // FLAGS['epochs_bw_eval']):
    lsd_depth_fuser.train(input_fn=train_input_fn, steps=training_steps)
    eval_results = lsd_depth_fuser.evaluate(input_fn=test_input_fn)
    print('\nEvaluation results:\n\t%s\n' % eval_results)
    

  #if model_helpers.past_stop_threshold(flags_obj.stop_threshold,
   #                                    eval_results['loss']):
    #break
  # Export the model for later use
  rgb = tf.placeholder(tf.float32, [1, 3, 240, 320])
  idepth = tf.placeholder(tf.float32, [1, 1, 240, 320])
  idepthVar = tf.placeholder(tf.float32, [1, 1, 240, 320])
  export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
     'rgb': rgb,
     'sparseInverseDepth': idepth,
     'sparseInverseDepthVariance': idepthVar
  })
  lsd_depth_fuser.export_savedmodel(os.path.join(model_dir, "exported_model"), 
    export_input_fn)

def dataset_len(dataset_loc):
    with open(dataset_loc) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
  parser = argparse.ArgumentParser(description=( "Train LSDDepth fusion network using the  custom estimator."))
  parser.add_argument("--model_name", type=str, required=True, help="Specify name of model (will be used as ouput dir_name)")
  parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
  parser.add_argument("--limit_gpu", action='store_true', help="run program with limited GPU (dacky)")
  args = parser.parse_args()

  training_file = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/train/rgbd_lsdDepth_train.txt"
  data_len = dataset_len(training_file)
  steps_per_epoch = data_len // FLAGS['batch_size']
  #num_steps = steps_per_epoch * args.num_epochs
  num_steps = steps_per_epoch * FLAGS['epochs_bw_eval']

  FLAGS['num_epochs'] = args.num_epochs

  print("Number of training steps (until eval):", num_steps)
  ####
  model_fn = model_fn_NetV02Res_LossL1SigL1ExpResL1
  #model_fn = model_fn_NetV04_LossL1SigL1
  #model_fn = model_fn_NetV04Res_LossL1SigL1ExpResL1
  #model_fn = model_fn_NetV03Res_LossL1SigL1ExpResL1
  #model_fn = model_fn_Netv3_LossL1SigL1  
  #model_fn = model_fn_Netv2_LossL1SigL1_down
  ####

  if args.limit_gpu:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    FLAGS['multi_gpu'] = False
  else:
    config = None

  run_trainer(model_fn, args.model_name, num_steps, config)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
  sys.exit()
