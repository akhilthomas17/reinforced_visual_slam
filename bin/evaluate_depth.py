#!/usr/bin/env python
import argparse
import numpy as np
import sys
import os
import glob
from enum import Enum
from collections import namedtuple

#for cv2
sys.path.insert(0,'/misc/software/opencv/opencv-3.2.0_cuda8_with_contrib-x86_64-gcc5.4.0/lib/python3.5/dist-packages')
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors


debug = False

read_png = lambda depth_png: cv2.imread(depth_png, -1).astype(np.float32)/5000
read_sparse = lambda sparse_bin: np.reciprocal(
	np.fromfile(sparse_bin, dtype=np.float16).astype(np.float32).reshape((480, 640)))
find_id = lambda filename: filename.split('/')[-1].split('_')[1]

def evaluate_sparse(gt_depths, sparse_depths, ids, eval_folder):
	sum_error = 0
	plot_dir = os.path.join(eval_folder, "sparse_depths")
	os.makedirs(plot_dir, exist_ok=True)
	for gt_depth, sparse_depth, iid in zip(gt_depths, sparse_depths, ids):
		nan_mask = np.ones_like(gt_depth)*np.nan 
		diff = np.where( sparse_depth>0, sparse_depth-gt_depth, nan_mask)
		sum_error += np.nanmean(diff**2)
		if debug:
			print("nan_mask:", nan_mask)
			print("sparse_depth:", sparse_depth)
			print("gt_depth", gt_depth)
			print("diff_r:", sparse_depth-gt_depth)
			print("diff:", diff)
			print("sum_error:", sum_error)
		plot_file = os.path.join(plot_dir, iid+".png")
		plot_coloured_depths(gt_depth, sparse_depth, plot_file)
	rms = np.sqrt(sum_error/len(gt_depths))
	with open(eval_folder+"/summary_sparse.txt", "w") as f:
		f.write("number of images compared: %d images\n"%len(gt_depths))
		f.write("RMSE(linear): %f m\n"%rms)

def evaluate_dense(gt_depths, dense_depths, ids, eval_folder):
	sum_error = 0
	for gt_depth, dense_depth in zip(gt_depths, dense_depths, ids):
		diff_image = dense_depth - gt_depth
		sum_error += np.mean((diff_image)*2)
		plot_file = os.path.join(plot_dir, iid+".png")
		plot_coloured_depths(gt_depth, dense_depth, plot_file)
	rms = np.sqrt(sum_error/len(gt_depths))
	with open(eval_folder+"/summary_dense.txt", "w") as f:
		f.write("number of images compared: %d images\n"%len(gt_depths))
		f.write("RMSE(linear): %f m\n"%rms)

def plot_coloured_depths(gt_depth, predicted_depth, plot_file, show_plots=debug):
	fig, axs = plt.subplots(1, 2)
	norm = colors.Normalize(vmin=0, vmax=6)
	cmap = 'hot'
	images = []
	images.append(axs[0].imshow(predicted_depth, cmap=cmap))
	images.append(axs[1].imshow(gt_depth, cmap=cmap))
	for im in images:
	    im.set_norm(norm)
	fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
	fig.savefig(plot_file)
	if show_plots:
		plt.show()
		key = plt.waitforbuttonpress()
	plt.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=( "Evaluate results of SLAM system configurations on datasets."))
	parser.add_argument("--result_folder", type=str, required=True, help="Folder which contains all sparse depth, gt_depth and predicted depths.")
	parser.add_argument("--no_dense_depths", action='store_true', help="Select this if no dense_depth_prediction is made.")
	parser.add_argument("--eval_folder", type=str, help="Specify the file to save summary of the evaluation.")
	parser.add_argument("--show_live_plots", action="store_true", help=("Select true to display plots while running. Active only when --plot_dir is provided"))
	args = parser.parse_args()

	assert(os.path.isdir(args.result_folder))
	if args.eval_folder:
		assert(os.path.isdir(eval_folder))
		eval_folder = args.eval_folder
	else:
		eval_folder = os.path.join(args.result_folder, "eval")
		os.makedirs(eval_folder, exist_ok=True)

	# read sparse depths and gt depths
	sparse_depth_files = sorted(glob.glob(args.result_folder+"/*sparse_depth.bin"))
	sparse_depths = list(map(read_sparse, sparse_depth_files))
	gt_depth_files = sorted(glob.glob(args.result_folder+"/*depthGT.png"))
	gt_depths = list(map(read_png, gt_depth_files))

	# read frame ids from filenames
	ids = list(map(find_id, gt_depth_files))

	if args.show_live_plots:
		debug = True

	if debug:
		print("sparse_depths[0]:", sparse_depths[0])
		print("len(sparse_depths):", len(sparse_depths))
		print("gt_depths[0]:", gt_depths[0])
		print("len(gt_depths):", len(gt_depths))
		print("ids:", ids)
		print("eval_folder:", eval_folder)


	# analyze sparse depths
	evaluate_sparse(gt_depths, sparse_depths, ids, eval_folder)

	if not args.no_dense_depths:
		# read dense depth predictions
		dense_depth_files = sorted(glob.glob(args.result_folder+"/*depthPrediction.png"))
		gt_depths = map(read_png, gt_depth_files)

		# analyze dense depths
		evaluate_dense(gt_depths, dense_depths, ids, eval_folder)

	print("Finished processing ", args.result_folder)
	sys.exit()