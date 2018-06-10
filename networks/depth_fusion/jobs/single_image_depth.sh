#!/bin/bash
source /misc/software/cuda/add_environment_cuda9.0.176_cudnnv7.sh
export LMBSPECIALOPS_LIB=/misc/lmbraid19/thomasa/deep-networks/lmbspecialops/build_tf1.8/lib/lmbspecialops.so
#roscd reinforced_visual_slam/networks/depth_fusion/
cd /misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion
python trainer_singleImage.py --num_epochs 50 --model_name NetV0Res_L1SigL1ExpResL1_down_aug_tr1_1