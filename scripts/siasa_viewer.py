#!/usr/bin/env python

import siasainterface as siasa
from reinforced_visual_slam.msg import *
from deepTAM.common_netcode.myTests import send_tra_to_siasa
from deepTAM.datatypes import Pose
from helpers import quaternion_to_rotation_matrix, invert_transformation

import rospy
from std_msgs.msg import String

import numpy as np


def visualize_groundtruth(path_to_gt_file):
	gt_data = np.loadtxt(path_to_gt_file, dtype=np.float32)
	gt_pose_list = [None]*len(gt_data)

	for indx in range(len(gt_data)):
		t = gt_data[indx, 1:4]
		R = quaternion_to_rotation_matrix(gt_data[indx, 4:].astype(np.float64))
		
		# Need to invert the pose since we require worldToCam pose
		Rinv, tinv = invert_transformation(R, t)

		gt_pose_list[indx] = Pose(R=Rinv, t=tinv)

	send_tra_to_siasa(gt_pose_list, color=(0,1,0), name='/cam_gt')


def liveframe_callback(keyframeMsg_data, conn=siasa.Connection('tcp://localhost:51454')):
    rospy.loginfo("I heard %s", str(keyframeMsg_data))

    camToWorld = np.asarray(keyframeMsg_data.camToWorld, dtype=np.float32)
    print("camToWorld = ", camToWorld)

    t = camToWorld[4:]
    print("t = ", t)

    q = camToWorld[:4]
    scale = np.linalg.norm(q).astype(np.float32)
    print("scale = ", scale)

    q_norm = q/scale
    R = scale * quaternion_to_rotation_matrix(q_norm.astype(np.float64))

    # Need to invert the pose since we require worldToCam pose
    Rinv, tinv = invert_transformation(R, t)

    siasa.setCameraData(np.array(Rinv).astype(np.float32), np.array(tinv).astype(np.float32), '/cam_live_reinforced', keyframeMsg_data.id, conn)


def listener():
    rospy.init_node('siasa_viewer')

    # Visualize the ground truth
    if rospy.has_param('~gt'):
    	path_to_gt_file = rospy.get_param('~gt')
    	visualize_groundtruth(path_to_gt_file)

    # Add subscribers for live frame
    rospy.Subscriber("/lsd_slam_reinforced/liveframes", keyframeMsg, liveframe_callback)
    
    print("Initialized siasa viewer node")
    rospy.spin()

if __name__ == '__main__':
    listener()
