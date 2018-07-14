#!/bin/bash
source /misc/software/cuda/add_environment_cuda9.0.176_cudnnv7.sh
export LMBSPECIALOPS_LIB=/misc/lmbraid19/thomasa/deep-networks/lmbspecialops/build_tf1.8/lib/lmbspecialops.so
#roscd reinforced_visual_slam/networks/depth_fusion/
cd /misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion
python trainer_singleImage.py --num_epochs 70 --model_name NetV0l_L1SigL1_tr2