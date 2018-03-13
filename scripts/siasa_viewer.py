#!/usr/bin/env python

import siasainterface as siasa
from reinforced_visual_slam.msg import *
from deepTAM.common_netcode.myTests import send_tra_to_siasa
from deepTAM.datatypes import Pose
from helpers import *
import rospy
from std_msgs.msg import String
import numpy as np

_debug = True


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

def keyframe_callback(keyframeMsg_data):
    rospy.loginfo("Received keyframe message")
    camToWorld = np.asarray(keyframeMsg_data.camToWorld, dtype=np.float32)
    siasa.setProperties('{ "pathActor": { "Visible": true} }', '/cam_keyframe_reinforced')
    T_camToWorld = send_frame_to_siasa(camToWorld, '/cam_keyframe_reinforced', keyframeMsg_data.id)
    pointcloud_bytes = keyframeMsg_data.pointcloud
    arr_len = keyframeMsg_data.width * keyframeMsg_data.height
    pointcloud_array = read_pointcloud_struct(pointcloud_bytes, arr_len)
    keyframeMsg_data.pointcloud = b''
    pointcloud_siasa = convert_to_siasa_pointcloud(pointcloud_array, T_camToWorld, keyframeMsg_data)
    #pointcloud = compute_point_cloud_from_depthmap(depth_abs, K, R1, t1, colors=image)#image1_2)
    aa = siasa.AttributeArray('Colors',pointcloud_siasa['colors'])
    siasa.setPolyData(pointcloud_siasa['points'],'/points_keyframe', 0, point_attributes=[aa])
    if _debug:
        print("Pointcloud: ", pointcloud_bytes[:36])
        rospy.logdebug("Pointcloud data input length: %d", len(pointcloud_bytes))
        rospy.logdebug("Pointcloud struct length: %d", arr_len)
        rospy.logerr("Pointcloud siasa length: %d", len(pointcloud_siasa['points']))

def liveframe_callback(keyframeMsg_data):
    rospy.loginfo("Received liveframe message: /n %s", str(keyframeMsg_data))
    camToWorld = np.asarray(keyframeMsg_data.camToWorld, dtype=np.float32)
    T_camToWorld = send_frame_to_siasa(camToWorld, '/cam_live_reinforced', keyframeMsg_data.id)

def send_frame_to_siasa(camToWorld, cam_name, idx, conn=siasa.Connection('tcp://localhost:51454')):
    print("camToWorld = ", camToWorld)
    t = camToWorld[4:]
    print("t = ", t)
    q = camToWorld[:4]
    scale = np.linalg.norm(q).astype(np.float32)
    print("scale = ", scale)
    q_norm = q/scale
    #R = scale * quaternion_to_rotation_matrix(q_norm.astype(np.float64))
    R = quaternion_to_rotation_matrix(q_norm.astype(np.float64))
    # Need to invert the pose since we require worldToCam pose
    Rinv, tinv, T_camToWorld = invert_transformation(R, t)
    siasa.setCameraData(np.array(Rinv).astype(np.float32), np.array(tinv).astype(np.float32), cam_name, idx, conn)
    return T_camToWorld

def convert_to_siasa_pointcloud(pointcloud_array, T_camToWorld, keyframeMsg_data):
    valid_points = 0
    valid_xy = []

    fxi = 1/keyframeMsg_data.fx
    fyi = 1/keyframeMsg_data.fy
    cxi = -keyframeMsg_data.cx / keyframeMsg_data.fx
    cyi = -keyframeMsg_data.cy / keyframeMsg_data.fy

    for x in range(keyframeMsg_data.width):
        for y in range(keyframeMsg_data.height):
            idepth = pointcloud_array[x + keyframeMsg_data.width*y].idepth
            if np.isfinite(idepth) and idepth > 0:
                valid_points += 1
                valid_xy.append((x,y))
    points_arr = np.empty((valid_points,3), dtype=np.float32)
    colors_attr_arr = np.empty((valid_points,3), dtype=np.uint8)

    num = 0
    for x,y in valid_xy:
        depth = 1/pointcloud_array[x + keyframeMsg_data.width*y].idepth
        point_kf = np.array([[ (x*fxi + cxi), (y*fyi + cyi), 1 ]]).T * depth
        point_kf = np.vstack((point_kf, 1))
        point = np.dot(T_camToWorld, point_kf)
        points_arr[num, :] = point[:3, 0]
        colors_attr_arr[num, :] = np.asarray(pointcloud_array[x + keyframeMsg_data.width*y].color)[:3]
        num += 1
    result = {'points':points_arr}
    result['colors'] = colors_attr_arr
    return result


def listener():
    rospy.init_node('siasa_viewer')
    # Visualize the groundtruth
    if rospy.has_param('~gt'):
    	path_to_gt_file = rospy.get_param('~gt')
    	visualize_groundtruth(path_to_gt_file)
    # Add subscribers for liveframes and keyframes
    rospy.Subscriber("/lsd_slam_reinforced/liveframes", keyframeMsg, liveframe_callback)
    rospy.Subscriber("/lsd_slam_reinforced/keyframes", keyframeMsg, keyframe_callback)
    print("Initialized siasa viewer node")
    rospy.spin()

if __name__ == '__main__':
    listener()
