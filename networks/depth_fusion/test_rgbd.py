#!/usr/bin/env python3

from reinforced_visual_slam.srv import *
import rospy

import matplotlib.pyplot as plt
from matplotlib import colors

import os
import sys
sys.path.insert(0,'/misc/software/opencv/opencv-3.2.0_cuda8_with_contrib-x86_64-gcc5.4.0/lib/python3.5/dist-packages')

import cv2
import numpy as np
from skimage.transform import resize

import tensorflow as tf

sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')
from net.my_models import *

import random

def predict_input_fn(rgb, sparse_idepth, sparse_idepth_var):
    feature_names = [
      'rgb',
      'sparseInverseDepth',
      'sparseInverseDepthVariance'
    ]
    input_tensors = [rgb, sparse_idepth[np.newaxis,:,:], sparse_idepth_var[np.newaxis,:,:]]
    inputs = dict(zip(feature_names, input_tensors))
    print(inputs['rgb'].shape)
    dataset = tf.data.Dataset.from_tensors(inputs)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    print(features['rgb'].shape, features['sparseInverseDepth'].shape, 
          features['sparseInverseDepthVariance'].shape)
    return features

def configure(model_name, model_function):
	model_base_dir = "/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion/training"
	model_dir = os.path.join(model_base_dir, model_name)
	print(model_dir)
	print(model_function)
	# loading the network
	lsd_depth_fuser = tf.estimator.Estimator(model_fn=model_function, model_dir=model_dir, 
	                                         params={'data_format':"channels_first", 'multi_gpu':False})

def test_network(model_name, model_function):
	model_base_dir = "/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion/training"
	model_dir = os.path.join(model_base_dir, model_name)
	print(model_dir)
	print(model_function)
	# loading the network
	lsd_depth_fuser = tf.estimator.Estimator(model_fn=model_function, model_dir=model_dir, 
	                                         params={'data_format':"channels_first", 'multi_gpu':False})

	test_file = '/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/test/rgbd_lsdDepth_test.txt'
	base_path = os.path.dirname(test_file) 
	with open(test_file) as file:
		lines = file.readlines()
		random.shuffle(lines)
		for line in lines:
			# prepare test input
			depth_gt_file, rgb_file, sparse_idepth_bin, sparse_idepth_var_bin = line.strip().split(',')[0:]

			print("Sparse idepth ---")
			sparse_idepth = np.fromfile(os.path.join(base_path, sparse_idepth_bin), dtype=np.float16)
			print("min half-float sparse_idepth: ", np.nanmin(sparse_idepth))
			print("max half-float sparse_idepth: ", np.nanmax(sparse_idepth))
			sparse_idepth = sparse_idepth.astype(np.float32)
			sparse_idepth = sparse_idepth.reshape((480, 640))
			#sparse_idepth = resize(sparse_idepth, output_shape=(240,320), order=0)
			sparse_idepth = cv2.resize(sparse_idepth, (320, 240), cv2.INTER_NEAREST)
			#sparse_idepth = sparse_idepth.resize()
			print("max: ", np.nanmax(sparse_idepth))
			print("shape: ",sparse_idepth.shape)
			print("dtype: ", sparse_idepth.dtype)

			print("Sparse idepth var ---")
			sparse_idepth_var = np.fromfile(os.path.join(base_path, sparse_idepth_var_bin), dtype=np.float16)
			print("min half-float sparse_idepth: ", np.nanmin(sparse_idepth_var))
			print("max half-float sparse_idepth: ", np.nanmax(sparse_idepth_var))
			sparse_idepth_var = sparse_idepth_var.astype(np.float32)
			sparse_idepth_var = sparse_idepth_var.reshape((480, 640))
			#sparse_idepth = resize(sparse_idepth, output_shape=(240,320), order=0)
			sparse_idepth_var = cv2.resize(sparse_idepth_var, (320, 240), cv2.INTER_NEAREST)
			#sparse_idepth = sparse_idepth.resize()
			print("max: ", np.nanmax(sparse_idepth_var))
			print("shape: ",sparse_idepth_var.shape)
			print("dtype: ", sparse_idepth_var.dtype)

			print("rgb---")
			rgb = cv2.imread(os.path.join(base_path, rgb_file), -1).astype(np.float32)/255
			rgb = cv2.resize(rgb, (320, 240))
			rgb = np.transpose(rgb, (2,0,1))
			print("shape: ", rgb.shape)
			print("max: ", np.nanmax(rgb))
			print("dtype: ", rgb.dtype)

			depth_gt = cv2.imread(os.path.join(base_path, depth_gt_file), -1).astype(np.float32)/5000
			gt_max = np.nanmax(depth_gt)

			depthmap_predicted = lsd_depth_fuser.predict(lambda : predict_input_fn(rgb, sparse_idepth, sparse_idepth_var))
			depthmap_scaled = list(depthmap_predicted)[0]['depth'][0]

			#invrting depthmap
			depthmap_scaled = np.where(depthmap_scaled>0.2, 1./depthmap_scaled, np.zeros_like(depthmap_scaled))

			fig, axs = plt.subplots(1, 2)
			norm = colors.Normalize(vmin=0, vmax=gt_max)
			cmap = 'hot'
			images = []
			images.append(axs[0].imshow(depthmap_scaled, cmap=cmap))
			images.append(axs[1].imshow(depth_gt, cmap=cmap))
			for im in images:
			    im.set_norm(norm)
			fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
			#plt.figure("sparse_depth")
			#plt.imshow(sparse_depth, cmap='hot', norm=norm)
			#plt.clim(0,gt_max)
			#plt.colorbar()
			#plt.figure("gt_depth")
			#plt.imshow(depth_gt, cmap='hot', norm=norm)
			#plt.clim(0,gt_max)
			#plt.colorbar()
			#plt.figure("sparse_depth_variance")
			#plt.imshow(sparse_idepthVar, cmap='hot')
			#plt.figure("rgb")
			#plt.imshow(rgb)
			plt.show()

			plt.waitforbuttonpress()

			#plt.imshow(depthmap_scaled, cmap='hot')
			#plt.colorbar()
			#plt.show()

def run():
    #parser = argparse.ArgumentParser(description=( "Checks validity of one Keyframe batch inside training data generated from LSD Slam to refine depth."))
    #parser.add_argument("--dataset_dir", type=str, required=True, help="training data directory to be verified")
    #parser.add_argument("--basename", type=str, required=True, help="basename of the Keyframe batch in training data")
    #parser.add_argument('--debug', action='store_true', help="enable debug outputs")

    #args = parser.parse_args()
    #assert os.path.isdir(args.dataset_dir)
    model_name = "NetV03_L1SigEqL1_down_aug"
    model_function = model_fn_Netv3_LossL1SigL1
    #configure(model_name, model_function)
    test_network(model_name, model_function)


if __name__ == "__main__":
    run()
    sys.exit()