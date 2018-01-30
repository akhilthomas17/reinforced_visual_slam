#!/usr/bin/env python

from reinforced_visual_slam.srv import *
import rospy
import tensorflow as tf

from cv_bridge import CvBridge
import cv2
from PIL import Image

from minieigen import Quaternion as quat
#from pyquaternion import Quaternion as quat
from geometry_msgs.msg import Transform, Quaternion, Vector3

from depthmotionnet.vis import angleaxis_to_rotation_matrix
from deepTAM.evaluation.helpers import *
from tfutils.helpers import optimistic_restore

class DeepTAMTracker(object):

    def __init__(self):
        rospy.init_node('deepTAM_tracker')
        print("Configuring tensorflow session")
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction=0.8
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        self.session.run(tf.global_variables_initializer())
        self._cv_bridge = CvBridge()

    def configure(self, network_script_path, network_session_path):
        print("Configuring the deepTAM network")
        tracking_mod = load_myNetworks_module_noname(network_script_path)
        self.tracking_net = tracking_mod.TrackingNetwork()
        [self.image_height, self.image_width] = self.tracking_net.placeholders['image_key'].get_shape().as_list()[-2:]
        self.output = self.tracking_net.build_net(**self.tracking_net.placeholders)
        optimistic_restore(self.session, network_session_path, verbose=True)

    def rotation_matrix_to_quaternion(self, R):
        q = Quaternion()
        q_mini = quat(R)
        q.x = q_mini[0]
        q.y = q_mini[1]
        q.z = q_mini[2]
        q.w = q_mini[3]
        # return Quaternion(*(quat(matrix=R).elements))
        return q


    def track_image_cb(self, req, test=False):
        print("Recieved image to track")

        # Requirement: Depth has to np array float32, in meters, with shape (1,1,240,320)
        # Get the Keyframe depth image and convert it to openCV Image format (Assuming Depth in Meters)
        try:
            keyframe_depth = self._cv_bridge.imgmsg_to_cv2(req.keyframe_depth, "passthrough")
        except CvBridgeError as e:
            print(e)
            return TrackImageResponse(0)
        # Resize Keyframe depth image to (320W,240H)
        keyframe_depth = cv2.resize(keyframe_depth, (320, 240))
        if test:
            cv2.imshow('CV Keyframe Depth',keyframe_depth)
            cv2.waitKey(3)
        # Convert Keyframe depth image to Float32 
        keyframe_depth = keyframe_depth.astype('float32')
        # Find inverse depth
        key_inv_depth = 1/keyframe_depth

        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,240,320)
        # Get the Keyframe RGB image from ROS message and convert it to openCV Image format
        try:
            keyframe_image = self._cv_bridge.imgmsg_to_cv2(req.keyframe_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return TrackImageResponse(0)
        if test:
            cv2.imshow('CV Keyframe RGB', keyframe_image)
            cv2.waitKey(3)
        # Normalize Keyframe RGB image to range (-0.5, 0.5) and save as Float32
        keyframe_image = cv2.normalize(keyframe_image, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Resize Keyframe RGB image to (320W,240H) and reshape it to (3, 240, 320)
        keyframe_image = cv2.resize(keyframe_image, (320, 240))
        keyframe_image = np.transpose(keyframe_image, (2,0,1))

        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,240,320)
        # Get the Current RGB image from ROS message and convert it to openCV Image format
        try:
            current_image = self._cv_bridge.imgmsg_to_cv2(req.current_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return TrackImageResponse(0)
        if test:
            cv2.imshow('CV Current RGB', current_image)
            cv2.waitKey(3)
        # Normalize Current RGB image to range (-0.5, 0.5) and save as Float32
        current_image = cv2.normalize(current_image, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Resize Current RGB image to (320W,240H) and reshape it to (3, 240, 320)
        current_image = cv2.resize(current_image, (320, 240))
        current_image = np.transpose(current_image, (2,0,1))

        # Loading prior rotation and translation
        #prev_rotation = quat(req.rotation_prior[3], req.rotation_prior[0], req.rotation_prior[1], req.rotation_prior[2])
        #prev_rotation = prev_rotation.toAxisAngle()
        intrinsics = np.array(req.intrinsics)
        prev_rotation = np.array(req.rotation_prior)
        prev_translation = np.array(req.translation_prior)

        # Loading the feed dict with converted values:
        feed_dict = {
            self.tracking_net.placeholders['depth_key']: key_inv_depth[np.newaxis,np.newaxis,:,:],
            self.tracking_net.placeholders['image_key']: keyframe_image[np.newaxis,:,:,:],
            self.tracking_net.placeholders['image_current']: current_image[np.newaxis,:,:,:],
            self.tracking_net.placeholders['intrinsics']: intrinsics[np.newaxis,:],
            self.tracking_net.placeholders['prev_rotation']: prev_rotation[np.newaxis,:],
            self.tracking_net.placeholders['prev_translation']: prev_translation[np.newaxis,:],
        }

        output_arrs = self.session.run(self.output, feed_dict=feed_dict)
        t = output_arrs['predict_translation'][0]
        #print("t = :", t)
        R = angleaxis_to_rotation_matrix(output_arrs['predict_rotation'][0])
        #print("R = :", R)
        R_w = R.dot(angleaxis_to_rotation_matrix(prev_rotation))
        print("R_w = :", R_w)
        t_w = R.dot(prev_translation) + t
        #print("t_w = :", t_w)

        print("Returning tracked response")
        return TrackImageResponse(Transform(Vector3(*t_w), self.rotation_matrix_to_quaternion(R_w)))

    def run(self):
        service = rospy.Service('track_image', TrackImage, self.track_image_cb)
        print("Ready to track images")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down deepTAM tracker")

if __name__ == "__main__":
    tracker = DeepTAMTracker()
    tracker.configure('/misc/lmbraid19/thomasa/deep-networks/deepTAM/nets_multiframes_uncertainties/tracking_scm_laplace_pairwise_fullres_nograd_fwd_multires/net/myNetworks.py',
      '/misc/lmbraid19/thomasa/deep-networks/deepTAM/nets_multiframes_uncertainties/tracking_scm_laplace_pairwise_fullres_nograd_fwd_multires/training/5_f1m1f2m2f3m3_frames2_new/checkpoints_new/snapshot-250000')
    tracker.run()
