#!/usr/bin/env python

import siasainterface as siasa
from reinforced_visual_slam.msg import *
from deepTAM.common_netcode.myTests import send_tra_to_siasa
from deepTAM.datatypes import Pose
from helpers import *
import rospy
from std_msgs.msg import String
import numpy as np
import matplotlib.colors as colors
import json
from minieigen import Quaternion
from depthmotionnet.vis import compute_point_cloud_from_depthmap

# for cv_bridge
sys.path.insert(0, '/misc/lmbraid19/thomasa/catkin_ws/install/lib/python3/dist-packages')
# for cv2
sys.path.insert(0,'/misc/software/opencv/opencv-3.2.0_cuda8_with_contrib-x86_64-gcc5.4.0/lib/python3.5/dist-packages')

from cv_bridge import CvBridge


class SiasaViewer(object):
    """Visualizer for lsd-deepTAM slam using siasa"""
    def __init__(self):
        super(SiasaViewer, self).__init__()
        rospy.init_node('siasa_viewer')
        self.debug = False
        #Setting properties of siasa viewer
        self.connection = siasa.Connection('tcp://localhost:51454')
        color_liveframe = colors.rgb2hex((1,0,0))
        color_keyframe = colors.rgb2hex((0,0,1))
        prop_dict_liveframe = {'cameraActor':{'property':{'Color':color_liveframe}},
                        'camera':{'Scale':0.01},
                        'path':{'Scale':0.01},
                        'pathActor':{'property':{'Color':color_liveframe}, 'Visible':True}
                       }
        prop_dict_keyframe = {'cameraActor':{'property':{'Color':color_keyframe}},
                        'camera':{'Scale':0.01},
                        'path':{'Scale':0.01},
                        'pathActor':{'property':{'Color':color_keyframe}, 'Visible':True}
                       }
        self.prop_liveframe = json.dumps(prop_dict_liveframe)
        self.prop_keyframe = json.dumps(prop_dict_keyframe)

        # calcualting intrinsics
        self.sun3d_intrinsics = np.array([[0.89115971, 1.18821299, 0.5, 0.5]],dtype=np.float32)
        self.sun3d_K = np.eye(3)
        self.sun3d_K[0,0] = self.sun3d_intrinsics[0][0]*640
        self.sun3d_K[1,1] = self.sun3d_intrinsics[0][1]*480
        self.sun3d_K[0,2] = self.sun3d_intrinsics[0][2]*640
        self.sun3d_K[1,2] = self.sun3d_intrinsics[0][3]*480

        # initializing cv_bridge
        self._cv_bridge = CvBridge()

        # initializing dictionary of keyframes
        self.keyframes = {}

    def run_listener(self):
        # Visualize the groundtruth
        if rospy.has_param('~gt'):
            self.has_gt = True
            self.gt_start = 0
            path_to_gt_file = rospy.get_param('~gt')
            self.gt_pose_list, self.gt_timestamps = self.shift_and_visualize_gt(path_to_gt_file)
        else:
            self.has_gt = False
        # Add subscribers for liveframes and keyframes
        self.liveframe_sub = rospy.Subscriber("/lsd_slam/liveframes", keyframeMsg, self.liveframe_callback)
        #self.keyframe_sub = rospy.Subscriber("/lsd_slam/keyframes", keyframeMsg, self.keyframe_callback)
        self.siasa_sub = rospy.Subscriber("/lsd_slam/siasa_keyframes", keyframeMsgSiasa, self.keyframe_siasa_callback)
        #self.keyframe_graph_sub = rospy.Subscriber("/lsd_slam/graph", keyframeGraphMsg, self.keyframe_graph_callback)
        rospy.loginfo("Initialized siasa viewer node")
        rospy.spin()

    def shift_and_visualize_gt(self, path_to_gt_file):
        """ Read associated groundtruth file and shift origin of the pose list so that it matches with the estimated trajectory """
        gt_data = np.loadtxt(path_to_gt_file, usecols=(0,1,2,3,4,5,6,7), dtype=float)
        gt_pose_list = [None]*len(gt_data)
        gt_pose_list[0] = Pose( R=np.eye(3), t=np.zeros((1,3)) )
        gt_timestamps = gt_data[:, 0]
        t_np = gt_data[:, 1:4]
        q_np = np.roll(gt_data[:, 4:], 1, axis=1)
        # Storing the initial pose of the groundtruth list as 3D Rigidbody Transform, assuming that it corresponds to origin of estimated trjectory
        R_origin = np.asarray( Quaternion(*(q_np[0,:])).toRotationMatrix() )
        T_gt_to_est = np.eye(4)
        T_gt_to_est[:3, :3] = R_origin
        T_gt_to_est[:3, 3] = t_np[0,:]
        T_gt_to_est = np.linalg.inv(T_gt_to_est)
        for ii in range(1, len(t_np)):
            T_current = np.eye(4)
            T_current[:3, 3] = t_np[ii,:]
            mini_q = Quaternion(*(q_np[ii, :]))
            T_current[:3, :3] = np.asarray(mini_q.toRotationMatrix())
            # Shift origin of current pose by multiplying it with Transformation matrix
            T_shifted = np.dot(T_gt_to_est, T_current)
            T_shifted = np.linalg.inv(T_shifted)
            shifted_pose = Pose(R=T_shifted[:3, :3], t=T_shifted[:3, 3])
            gt_pose_list[ii] = shifted_pose
        return gt_pose_list, gt_timestamps

    def visualize_groundtruth(self, path_to_gt_file):
        gt_data = np.loadtxt(path_to_gt_file, dtype=float)
        gt_pose_list = [None]*len(gt_data)
        gt_timestamps = gt_data[:,0]
        gt_data = gt_data.astype(np.float32)
        for indx in range(len(gt_data)):
            t = gt_data[indx, 1:4]
            R = quaternion_to_rotation_matrix(gt_data[indx, 4:].astype(np.float64))
            # Need to invert the pose since we require worldToCam pose
            Rinv, tinv, T_camToWorld = invert_transformation(R, t)
            gt_pose_list[indx] = Pose(R=Rinv, t=tinv)
        #send_tra_to_siasa(gt_pose_list, color=(0,1,0), name='/cam_gt')
        return gt_pose_list, gt_timestamps

    def keyframe_graph_callback(self, keyframeGraphMsg_data):
        rospy.loginfo("Received keyframe graph message")
        frame_data_bytes = keyframeGraphMsg_data.frameData
        frame_data = read_frameData_struct(frame_data_bytes, keyframeGraphMsg_data.numFrames)
        for frame in frame_data:
            kf = self.keyframes[frame.id]
            camToWorld_new = np.asarray(frame.camToWorld, dtype=np.float32)
            if not np.array_equal(kf.camToWorld, camToWorld_new):
                rospy.logerr("siasa_viewer: changing graph for keyframe with id: %d", frame.id)
                print("camToWorld_new", camToWorld_new)
                print("kf.camToWorld", kf.camToWorld)
                R, t = self.send_frame_to_siasa(camToWorld_new, '/cam_keyframe_reinforced', frame.id, self.prop_keyframe, 
                    return_inverse=True)
                # Make pointcloud out of the depth image and colour it with the rgb image
                pointcloud_siasa = compute_point_cloud_from_depthmap(kf.depth, self.sun3d_K, R, t, colors=kf.rgb.transpose([2,0,1]))
                aa = siasa.AttributeArray('Colors', pointcloud_siasa['colors'])
                siasa.setPolyData(pointcloud_siasa['points'],'points_keyframe/'+str(frame.id), 0, point_attributes=[aa], 
                    connection=self.connection)
                self.keyframes[frame.id] = Keyframe(camToWorld_new, kf.depth, kf.rgb)


    def keyframe_siasa_callback(self, keyframeMsg_data):
        rospy.loginfo("Received siasa keyframe message")
        
        # Get the Keyframe Depth image from ROS message and convert it to openCV Image format
        try:
            depth = self._cv_bridge.imgmsg_to_cv2(keyframeMsg_data.depth, "passthrough")
        except CvBridgeError as e:
            print(e)
            return

        # Get the Keyframe RGB image from ROS message and convert it to openCV Image format
        try:
            rgb = self._cv_bridge.imgmsg_to_cv2(keyframeMsg_data.rgb, "passthrough")
        except CvBridgeError as e:
            print(e)
            return

        # Read camToWorld from message and convert to world to cam and send to siasa
        camToWorld = np.asarray(keyframeMsg_data.camToWorld, dtype=np.float32)
        self.keyframes[keyframeMsg_data.id] = Keyframe(camToWorld, depth, rgb)
        if(self.debug):
            print("siasa_viewer: keyframeMsg_data.id*******************", keyframeMsg_data.id)
        R, t = self.send_frame_to_siasa(camToWorld, '/cam_keyframe_reinforced', keyframeMsg_data.id, 
            self.prop_keyframe, return_inverse=True)

        # Make pointcloud out of the depth image and colour it with the rgb image
        pointcloud_siasa = compute_point_cloud_from_depthmap(depth, self.sun3d_K, R, t, colors=rgb.transpose([2,0,1]))
        aa = siasa.AttributeArray('Colors', pointcloud_siasa['colors'])
        siasa.setPolyData(pointcloud_siasa['points'],'points_keyframe/'+str(keyframeMsg_data.id), 0, point_attributes=[aa], 
            connection=self.connection)

    def keyframe_callback(self, keyframeMsg_data):
        rospy.loginfo("Received keyframe message")
        camToWorld = np.asarray(keyframeMsg_data.camToWorld, dtype=np.float32)
        T_camToWorld = self.send_frame_to_siasa(camToWorld, '/cam_keyframe_reinforced', keyframeMsg_data.id, self.prop_keyframe)
        pointcloud_bytes = keyframeMsg_data.pointcloud
        arr_len = keyframeMsg_data.width * keyframeMsg_data.height
        pointcloud_array = read_pointcloud_struct(pointcloud_bytes, arr_len)
        keyframeMsg_data.pointcloud = b''
        pointcloud_siasa = self.convert_to_siasa_pointcloud(pointcloud_array, T_camToWorld, keyframeMsg_data)
        #pointcloud = compute_point_cloud_from_depthmap(depth_abs, K, R1, t1, colors=image)#image1_2)
        aa = siasa.AttributeArray('Colors', pointcloud_siasa['colors'])
        siasa.setPolyData(pointcloud_siasa['points'],'/points_keyframe', 0, point_attributes=[aa], connection=self.connection)
        if self.debug:
            print("siasa_viewer: Pointcloud: ", pointcloud_bytes[:36])
            rospy.logdebug("siasa_viewer: Pointcloud data input length: %d", len(pointcloud_bytes))
            rospy.logdebug("siasa_viewer: Pointcloud struct length: %d", arr_len)
            rospy.logdebug("siasa_viewer: Pointcloud siasa length: %d", len(pointcloud_siasa['points']))
            print("siasa_viewer: keyframeMsg_data.id***********", keyframeMsg_data.id)

    def liveframe_callback(self, keyframeMsg_data):
        rospy.loginfo("siasa_viewer: Received liveframe message")
        camToWorld = np.asarray(keyframeMsg_data.camToWorld, dtype=np.float32)
        if self.has_gt:
            timestamp_now = keyframeMsg_data.time
            self.gt_end = np.argmax(self.gt_timestamps > timestamp_now)
            gt_pose_list_now = self.gt_pose_list[self.gt_start : self.gt_end]
            if self.gt_end > self.gt_start:
                send_tra_to_siasa(gt_pose_list_now, color=(0,1,0), name='/cam_gt', ind_prefix=self.gt_start)
                self.gt_start = self.gt_end
        if self.debug:
            print("siasa_viewer: keyframeMsg_data.id********************", keyframeMsg_data.id)
            rospy.loginfo("siasa_viewer: timestamp_now: %f", timestamp_now)
            print("siasa_viewer: self.gt_timestamps: ", self.gt_timestamps)
            rospy.loginfo("siasa_viewer: self.gt_end: %d", self.gt_end)
        T_camToWorld = self.send_frame_to_siasa(camToWorld, '/cam_live_reinforced', keyframeMsg_data.id, self.prop_liveframe)
        

    def send_frame_to_siasa(self, camToWorld, cam_name, idx, cam_prop, return_inverse=False):
        t = camToWorld[4:]
        q = camToWorld[:4]
        scale = np.linalg.norm(q)
        q_norm = q/scale
        #R = scale * quaternion_to_rotation_matrix(q_norm.astype(np.float64))
        R = quaternion_to_rotation_matrix(q_norm.astype(np.float64))
        # Need to invert the pose since we require worldToCam pose
        Rinv, tinv, T_camToWorld = invert_transformation(R, t)
        siasa.setCameraData(np.array(Rinv).astype(np.float32), np.array(tinv).astype(np.float32), cam_name, idx, self.connection)
        siasa.setProperties(cam_prop, cam_name)
        if self.debug:
            print("siasa_viewer: scaled camToWorld message = ", camToWorld)
            print("siasa_viewer: scale = ", scale)
            print("siasa_viewer: t descaled camToWorld= ", t)
            print("siasa_viewer: q descaled camToWorld = ", q_norm)
            print("siasa_viewer: R descaled camToWorld = ", R)
            print("siasa_viewer: T_camToWorld descaled = ", T_camToWorld)
            print("siasa_viewer: t descaled worldToCam= ", tinv)
            print("siasa_viewer: q descaled worldToCam = ", Rinv)
        if return_inverse:
            return np.array(Rinv).astype(np.float32), np.array(tinv).astype(np.float32)
        else:
            return T_camToWorld

    def convert_to_siasa_pointcloud(self, pointcloud_array, T_camToWorld, keyframeMsg_data):
        valid_points = 0
        valid_xy = []

        fxi = 1/keyframeMsg_data.fx
        fyi = 1/keyframeMsg_data.fy
        cxi = -keyframeMsg_data.cx / keyframeMsg_data.fx
        cyi = -keyframeMsg_data.cy / keyframeMsg_data.fy

        for x in range(keyframeMsg_data.width):
            for y in range(keyframeMsg_data.height):
                idepth = pointcloud_array[x + keyframeMsg_data.width*y].idepth
                # setting a limit of 5m to maximum observable depth (not ideal!!)
                if np.isfinite(idepth) and idepth > 0.2:
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

if __name__ == '__main__':
    viewer = SiasaViewer()
    viewer.run_listener()
