#!/usr/bin/env python
from enum import Enum
from collections import namedtuple
import os
import sys
import subprocess
import time
import argparse
import re
import rostopic
import shutil

FLAGS = {
	"first_run":True,
	"deepTAM_tracker_ON":False,
	"single_image_predictor_ON": False,
	"depth_completion_net_ON":False
}

run_config = namedtuple("run_config", ["doSlam", "gtBootstrap", "useGtDepth", "predictDepth", 
	"depthCompletion", "readSparse"])

class mode(Enum):
	#LSDSLAM = run_config(True, True, False, False, False, False)
	DEEPTAM = run_config(False, True, True, False, False ,False)
	DEEPTAMSLAM = run_config(True, True, True, False, False ,False)
	CNNSLAM_GT = run_config(True, True, True, True, False ,False)
	CNNSLAM_GTINIT = run_config(True, True, False, True, False ,False)
	CNNSLAM = run_config(True, False, False, True, False ,False)
	SPARSEDEPTHCOMPSLAM = run_config(True, False, False, False, True ,True)
	DENSEDEPTHCOMPSLAM = run_config(True, False, False, False, True ,False)
	SPARSEDEPTHCOMPSLAM_GTINIT = run_config(True, True, False, False, True ,True)
	DENSEDEPTHCOMPSLAM_GTINIT = run_config(True, True, False, False, True ,False)

class calib(Enum):
	Freiburg1 = "/misc/lmbraid19/thomasa/rosbuild_ws/packages/lsd_slam/lsd_slam_core/calib/OpenCV_example_calib.cfg"
	Freiburg2 = "/misc/lmbraid19/thomasa/rosbuild_ws/packages/lsd_slam/lsd_slam_core/calib/tum_rgbd_fr2_calib.cfg"
	Freiburg3 = "/misc/lmbraid19/thomasa/rosbuild_ws/packages/lsd_slam/lsd_slam_core/calib/tum_rgbd_fr3_calib.cfg"

bool_str_maker = lambda x: "true" if x else "false"

def get_calib_file(seq_name):
	found_calib = False
	for c in calib:
		if re.search(c.name, seq_name, re.IGNORECASE):
			found_calib = True
			print("Found calib file, returning!!")
			return c.value
	print("Cannot find calib file, returning default calib")
	return calib.Freiburg3.value


def walkdir(folder, level):
	assert(os.path.isdir(folder))
	num_sep = folder.count(os.path.sep)
	for root, dirs, files in os.walk(folder):
	    yield root, dirs, files
	    num_sep_this = root.count(os.path.sep)
	    if num_sep + level <= num_sep_this:
	        del dirs[:]
	return


def run_SLAM(sequence, config, debug=False, use_siasa=False):
	if FLAGS["first_run"]:
		## run the roscore if needed ##
		try:
			rostopic.get_topic_class('/rosout')
			roscore_running = True
		except rostopic.ROSTopicIOException as e:
			roscore_running = False
		if not roscore_running:
			command = "roscore"
			popen1 = subprocess.Popen(command, shell=True)

		if use_siasa:
			## run siasa_viewer node
			command = "rosrun reinforced_visual_slam siasa_viewer.py"
			popen5 = subprocess.Popen(command, shell=True)

		FLAGS["first_run"] = False

	if not FLAGS["single_image_predictor_ON"]:
		if config.value.predictDepth and config.value.depthCompletion:
			## run depthcompletion node
			command = "rosrun reinforced_visual_slam single_image_depth_predictor.py"
			popen3 = subprocess.Popen(command, shell=True)
			FLAGS["single_image_predictor_ON"] = True

	if not FLAGS["depth_completion_net_ON"]:
		if config.value.predictDepth and (not config.value.depthCompletion or not config.value.gtBootstrap):
			## run singleimage depth prediction node
			command = "rosrun reinforced_visual_slam depthmap_fuser.py"
			popen4 = subprocess.Popen(command, shell=True)
			FLAGS["depth_completion_net_ON"] = True

	if not FLAGS["deepTAM_tracker_ON"]:
		## run the required nodes depending on the configuration under test ##
		if config.name != "LSDSLAM":
			## run DeepTAM tracker node
			command = "rosrun reinforced_visual_slam deepTAM_tracker.py"
			popen2 = subprocess.Popen(command, shell=True)
			FLAGS["deepTAM_tracker_ON"] = True
			#input("Wait until the DeepTAM is loaded and then Press Enter...")

	if config.name == "LSDSLAM":
		KFDistWeight = "5"
		KFUsageWeight = "3"
	else:
		## These variables are used differently in LSD SLAM
		KFDistWeight = "0.15"
		KFUsageWeight = "5"

	## set defaults for the SLAM system ##
	run_command = "rosrun lsd_slam_core deepTAM_dataset_slam"
	hz = "0"
	minUseGrad = "3"
	testMode = "true"
	writeTestDepths = "true"
	doKFReActivation = "true"
	showDebugWindow = "false"
		

	if debug:
		print("sequence: ", sequence)
		print("config: ", config.name)
		#input("Press enter to confirm ...")

	# Finding dataset base location
	base = "/misc/scratchSSD/ummenhof/rgbd_tum"
	seq_name = os.path.dirname(sequence).split('/')[-1]
	basename = os.path.join(base, seq_name)

	# Finding result folder
	result_base_dir = "/misc/lmbraid19/thomasa/results/tum_rgbd"
	seq_base_dir = os.path.join(result_base_dir, seq_name)
	output_folder = os.path.join(seq_base_dir, config.name)
	if os.path.isdir(output_folder):
		shutil.rmtree(output_folder)
	os.makedirs(output_folder, exist_ok=True)

	calib_file = get_calib_file(seq_name)

	# remapping nodename
	node_name = config.name + "_" + seq_name
	node_name_remap = " __name:=" + node_name

	command = (run_command + node_name_remap + " _calib:=" + calib_file +" _files:=" + sequence + " _basename:=" + basename +
	               " _hz:=" + hz + " _minUseGrad:=" + minUseGrad + " _doKFReActivation:=" + doKFReActivation
	               + " _testMode:=" + testMode + " _writeTestDepths:=" + writeTestDepths + " _showDebugWindow:=" + showDebugWindow
	               + " _KFUsageWeight:=" + (KFUsageWeight) + " _KFDistWeight:=" + (KFDistWeight) 
	               + " _outputFolder:=" + output_folder
	               + " _doSlam:=" + bool_str_maker(config.value.doSlam)
	               + " _gtBootstrap:=" + bool_str_maker(config.value.gtBootstrap)
	               + " _useGtDepth:=" + bool_str_maker(config.value.useGtDepth)
	               + " _predictDepth:=" + bool_str_maker(config.value.predictDepth)
	               + " _depthCompletion:=" + bool_str_maker(config.value.depthCompletion)
	               + " _readSparse:=" + bool_str_maker(config.value.readSparse)
	               )

	if debug:
		print("command: ", command)
		print("seq_name: ", seq_name)
		print("calib_file: ", calib_file)

	subprocess.Popen(command, shell=True)

def evaluate_result(sequence_name, config, result_folder):
	## evaluate trajectory errors ##
	base = "/misc/scratchSSD/ummenhof/rgbd_tum"
	gt_file = os.path.join(base, sequence_name + "/groundtruth.txt")
	traj_file = os.path.join(result_folder, "final_trajectory.txt")

	while not os.path.isfile(traj_file):
		try:
			time.sleep(1)
			print("waiting for "+ config.name+ " on " + sequence_name + " to finish")
		except KeyboardInterrupt:
			skip = input('Enter s to skip this sequence:')
			if skip == 's':
				return 

	eval_folder = os.path.join(result_folder, "eval")
	os.makedirs(eval_folder, exist_ok=True)
	ate_plot = os.path.join(eval_folder, "ate.png")
	rpe_plot = os.path.join(eval_folder, "rpe.png")
	summary_file_ate = os.path.join(eval_folder, "summary_ate.txt")
	result_file_ate = os.path.join(eval_folder, "result_ate.txt")
	summary_file_rpe = os.path.join(eval_folder, "summary_rpe.txt")
	result_file_rpe = os.path.join(eval_folder, "result_rpe.txt")

	eval_ate = ("evaluate_ate.py " + gt_file +" " + traj_file + " --plot "+ ate_plot +" --verbose" + 
		" --save_associations " + result_file_ate + " --summary " + summary_file_ate)
	subprocess.Popen(eval_ate, shell=True)
	time.sleep(0.5)
	eval_rpe = ("evaluate_rpe.py " + gt_file +" " + traj_file + " --plot "+ rpe_plot + " --save " + 
		result_file_rpe + " --summary " + summary_file_rpe +" --fixed_delta --verbose")
	subprocess.Popen(eval_rpe, shell=True)

	eval_depth = "evaluate_depth.py --result_folder " + result_folder
	if not config.value.predictDepth:
		eval_depth = eval_depth + " --no_dense_depths"
	subprocess.Popen(eval_depth, shell=True)
		
def start_test():
	parser = argparse.ArgumentParser(description=( "Test different SLAM system configurations on datasets."))
	parser.add_argument("--sequence_name", type=str, help="Name of sequence in tum_rgbd to be tested. If not provided, test will be run on whole of tum_rgbd sequences")
	parser.add_argument("--config", type=str, help="Configuration of SLAM system to be tested. If not provided, all designed configurations will be tested.")
	parser.add_argument('--use_siasa', action='store_true', help="start siasa_viewer alongside")
	parser.add_argument('--only_eval', action='store_true', help="Run only evaluation, assuming that results are already made")
	parser.add_argument('--start_seq', type=str, help="Start eval from this sequence (make sure --sequence_name is not provided")

	root_seq_dir = "/misc/lmbraid19/thomasa/datasets/rgbd"
	sequences = []

	args = parser.parse_args()
	assert(not(args.sequence_name and args.start_seq))

	if args.sequence_name:
		sequence_name = args.sequence_name
		sequence = os.path.join(root_seq_dir, sequence_name, "rgb_depth_associated.txt")
		assert os.path.isfile(sequence)
		sequences.append(sequence)
	else:
		regex = re.compile('rgbd')
		for root, dirs, files in walkdir(root_seq_dir, 0):
			for dir in sorted(dirs):
				if regex.match(dir):
					sequences.append(os.path.join(root_seq_dir, dir, "rgb_depth_associated.txt"))
	assert(sequences)

	configs = []
	found_config = False
	if args.config:
		for m in mode:
			if re.search(args.config, m.name, re.IGNORECASE):
				conf = m
				found_config = True
				print("Found configuration:", conf.value)
				configs.append(conf)
				break
		if not found_config:
			print("wrong configuration: available configurations are:")
			for m in mode:
				print(m.name)
				return
	else:
		for m in mode:
			configs.append(m)
	assert(configs)

	print("configs to test: ", configs)
	print("sequences to test: ", sequences)
	input("Press Enter to continue")

	start_eval = False
	if not args.only_eval:
		for sequence in sequences:
			for config in configs:
				run_SLAM(sequence=sequence, config=config, debug=True, use_siasa=args.use_siasa)
				seq_name = os.path.dirname(sequence).split('/')[-1]
				if args.start_seq and not start_eval:
					if args.start_seq != seq_name:
						break
					else:
						start_eval = True
						break
				# Finding result folder
				result_base_dir = "/misc/lmbraid19/thomasa/results/tum_rgbd"
				seq_base_dir = os.path.join(result_base_dir, seq_name)
				output_folder = os.path.join(seq_base_dir, config.name)
				evaluate_result(sequence_name=seq_name, config=config, result_folder=output_folder)
				time.sleep(10)

	return

	start_eval = False
	for sequence in sequences:
		for config in configs:
			seq_name = os.path.dirname(sequence).split('/')[-1]
			if args.start_seq and not start_eval:
				if args.start_seq != seq_name:
					break
				else:
					start_eval = True
					break
			# Finding result folder
			result_base_dir = "/misc/lmbraid19/thomasa/results/tum_rgbd"
			seq_base_dir = os.path.join(result_base_dir, seq_name)
			output_folder = os.path.join(seq_base_dir, config.name)
			evaluate_result(sequence_name=seq_name, config=config, result_folder=output_folder)

if __name__ == "__main__":
	start_test()
	sys.exit()