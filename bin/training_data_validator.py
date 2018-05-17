#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def check_validity_LSD_depth_dataset(dataset_folder, basename, matplotlib=True, debug=False):
    """ 
    Function to check validity of the generated LSD Slam depth refinement dataset.
    Provide the function with datasetset_folder and basename of the Keyframe batch to be verified. 
    """

    #Inside notebook uncomment the following
    #matplotlib.rcsetup.interactive_bk
    #plt.switch_backend('nbAgg')
    
    # Setting paths of the keyframe batch
    depth_gt_png = dataset_folder + basename + "depthGT.png"
    print(depth_gt_png)
    rgb_png = dataset_folder + basename + "rgb.png"
    sparse_depth_bin = dataset_folder + basename + "sparse_depth.bin"
    sparse_variance_bin = dataset_folder + basename + "sparse_depthVar.bin"
    
    # opening depth_gt and converting it to float
    print("GT depth---")
    depth_gt = cv2.imread(depth_gt_png, -1).astype(np.float32)/5000
    rgb = cv2.imread(rgb_png, -1)
    gt_max = np.nanmax(depth_gt)
    print("shape: ", depth_gt.shape)
    print("max: ", gt_max)
    print("dtype: ", depth_gt.dtype)
    
    # loading sparse idepth from bin (as half-float) and converting it to float32
    print("Sparse idepth ---")
    sparse_idepth = np.fromfile(sparse_depth_bin, dtype=np.float16)
    if(debug):
        print("min half-float sparse_idepth: ", np.nanmin(sparse_idepth))
        print("max half-float sparse_idepth: ", np.nanmax(sparse_idepth))
    sparse_idepth = sparse_idepth.astype(np.float32)
    sparse_idepth = sparse_idepth.reshape((480, 640))
    print("max: ", np.nanmax(sparse_idepth))
    print("shape: ",sparse_idepth.shape)
    print("dtype: ", sparse_idepth.dtype)
    
    # converting sparse idepth to depth
    print("Sparse depth ---")
    sparse_depth = (1./sparse_idepth)
    print("max: ", np.nanmax(sparse_depth))
    print("shape: ", sparse_depth.shape)
    print("dtype: ", sparse_depth.dtype)
    
    # loading sparse idepthVar from bin (as half-float) and converting it to float32
    print("Sparse idepthVar ---")
    sparse_idepthVar = np.fromfile(sparse_variance_bin, dtype=np.float16)
    if(debug):
        print("min half-float sparse_idepthVar: ", np.nanmin(sparse_idepthVar))
        print("max half-float sparse_idepthVar: ", np.nanmax(sparse_idepthVar))
    sparse_idepthVar = sparse_idepthVar.astype(np.float32)
    sparse_idepthVar = sparse_idepthVar.reshape((480, 640))
    print("max: ", np.nanmax(sparse_idepthVar))
    print("shape: ", sparse_idepthVar.shape)
    print("dtype: ", sparse_idepthVar.dtype)
    
    # plotting images
    if matplotlib:
        # plot using matplotlib
        fig, axs = plt.subplots(1, 2)
        norm = colors.Normalize(vmin=0, vmax=gt_max)
        cmap = 'hot'
        images = []
        images.append(axs[0].imshow(sparse_depth, cmap=cmap))
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
        plt.figure("sparse_depth_variance")
        plt.imshow(sparse_idepthVar, cmap='hot')
        plt.figure("rgb")
        plt.imshow(rgb)
        plt.show()
    else:
        # plot using opencv
        sparse_plot = cv2.convertScaleAbs(sparse_depth*255./4.5)
        gt_plot = cv2.convertScaleAbs(depth_gt*255./4.5)
        cv2.imshow("sparse_depth", sparse_plot)
        cv2.imshow("gt_depth", gt_plot)
        cv2.imshow("rgb", rgb_gt)
        cv2.waitKey(0)

    
    if(debug):
        print("min depth_gt: ", np.nanmin(depth_gt))
        print("min sparse_idepth: ", np.nanmin(sparse_idepth))
        print("min sparse_depth: ", np.nanmin(sparse_depth))
        print("min sparse_idepthVar: ", np.nanmin(sparse_idepthVar))

def run():
    parser = argparse.ArgumentParser(description=( "Checks validity of one Keyframe batch inside training data generated from LSD Slam to refine depth."))
    parser.add_argument("--dataset_dir", type=str, required=True, help="training data directory to be verified")
    parser.add_argument("--basename", type=str, required=True, help="basename of the Keyframe batch in training data")
    parser.add_argument('--debug', action='store_true', help="enable debug outputs")

    args = parser.parse_args()
    assert os.path.isdir(args.dataset_dir)
    check_validity_LSD_depth_dataset(dataset_folder=args.dataset_dir+"/", basename=args.basename, debug=args.debug)


if __name__ == "__main__":
    run()
    sys.exit()