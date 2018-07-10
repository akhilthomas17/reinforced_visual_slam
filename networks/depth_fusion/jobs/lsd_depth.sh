#!/bin/bash
source /misc/software/cuda/add_environment_cuda9.0.176_cudnnv7.sh
export LMBSPECIALOPS_LIB=/misc/lmbraid19/thomasa/deep-networks/lmbspecialops/build_tf1.8/lib/lmbspecialops.so
#roscd reinforced_visual_slam/networks/depth_fusion/
cd /misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion
python trainer_rgbd.py --num_epochs 30 --model_name NetV04_L1Sig4L1_down_tr1
