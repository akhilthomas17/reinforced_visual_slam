#!/usr/bin/env python3
import os
import sys
import argparse
import time
import subprocess
import numpy as np

def dataset_len(dataset_loc):
    with open(dataset_loc) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def gen_lsd_depth_training_data(dataset_loc, len_traj=200, keyframe_rate = 10, pause=2.5, test=False, debug=False):
    # static params (not to be changed)
    bin_loc = "rosrun lsd_slam_core depth_training_data_gen"
    hz = "0"
    doSlam = "true"
    KFDistWeight = 7
    
    # params that may be changed for different datasets
    # uncomment below for tum-fr1
    #calib_file = "/misc/lmbraid19/thomasa/rosbuild_ws/packages/lsd_slam/lsd_slam_core/calib/OpenCV_example_calib.cfg"
    # uncomment below for tum-fr2
    #calib_file = "/misc/lmbraid19/thomasa/rosbuild_ws/packages/lsd_slam/lsd_slam_core/calib/tum_rgbd_fr2_calib.cfg"
    # uncomment below for tum-fr3
    calib_file = "/misc/lmbraid19/thomasa/rosbuild_ws/packages/lsd_slam/lsd_slam_core/calib/tum_rgbd_fr3_calib.cfg"
    minUseGrad = "3"
    
    # Finding number of iterations needed: (Assuming "keyframe_rate" frames avg between each KFs)
    len_sequence = dataset_len(dataset_loc)
    number_itrns = keyframe_rate * int(len_sequence/len_traj)

    # Finding dataset basename
    base = "/misc/scratchSSD/ummenhof/rgbd_tum/"
    seq_name = os.path.dirname(dataset_loc).split('/')[-1]
    basename = base + seq_name
    
    # Finding start index for each iteration in the trajectory
    max_start = len_sequence - len_traj
    start_indx_array = np.random.randint(max_start, size=number_itrns)
    start_indx_array = start_indx_array.astype(np.unicode_)
    
    if test:
        itrn_max = 1
    else:
        itrn_max = number_itrns
    
    if debug:
        print("len_sequence: ", len_sequence)
        print("number_itrns: ", number_itrns)
        print("max_start: ", max_start)
        print("start_indx_array: ", start_indx_array)
        print("len_traj: ", len_traj)

    process_list= []
    args_prefix = (bin_loc + " _calib:=" + calib_file +" _files:=" + dataset_loc + " _basename:=" + basename +
                   " _hz:=" + hz + " _doSlam:=" + doSlam + " _len_traj:=" + str(len_traj) + " _minUseGrad:=" 
                   + minUseGrad + " _KFUsageWeight:=3 _KFDistWeight:=" + str(KFDistWeight) + " _start_indx:=")
    node_name_remap = " /LSD_SLAM_TRAIN_DATA_GEN:=/LSDDepthtraining"
    itr_num = " _itr_num:="

    # running roscore
    command = "roscore"
    popen = subprocess.Popen(command, shell=True)

    for ii in range(itrn_max):
        start_indx = start_indx_array[ii]
        command = args_prefix + start_indx + node_name_remap + str(ii) + itr_num + str(ii)
        process_list.append( subprocess.Popen(command, shell=True) )
        time.sleep(pause)

def run():
    parser = argparse.ArgumentParser(description=( "Generate training data for depth refinement by running LSD Slam on the given dataset."))
    parser.add_argument("--sequence", type=str, required=True, help="Path to input sequence")
    parser.add_argument("--len_traj", type=int, help="Length of partial sequence on which LSD Slam runs")
    parser.add_argument("--pause", type=float, help="Time to pause between two iterations in seconds")
    parser.add_argument("--keyframe_rate", type=int, help="Average number of frames between 2 keyframes")
    parser.add_argument('--test', action='store_true', help="start program in test mode")
    parser.add_argument('--debug', action='store_true', help="enable debug outputs")
    parser.add_argument("--intro", action="store_true", help="print an intro to the i/p dataset and exit")

    args = parser.parse_args()
    assert os.path.isfile(args.sequence)

    len_traj = 200
    keyframe_rate = 10

    kwargs = dict(dataset_loc=args.sequence)
    if args.len_traj:
        kwargs['len_traj'] = args.len_traj
        len_traj = args.len_traj
    if args.keyframe_rate:
        kwargs['keyframe_rate'] = args.keyframe_rate
        keyframe_rate = args.keyframe_rate
    if args.pause:
    	kwargs['pause'] = args.pause
    if args.test:
    	kwargs['test'] = args.test
    if args.debug:
    	kwargs['debug'] = args.debug

    if args.intro:
        len_sequence = dataset_len(args.sequence)
        number_itrns = keyframe_rate * int(len_sequence/len_traj)
        print("len_sequence: ", len_sequence)
        print("number_itrns: ", number_itrns)
    else:
        gen_lsd_depth_training_data(**kwargs)


if __name__ == "__main__":
    run()
    sys.exit()