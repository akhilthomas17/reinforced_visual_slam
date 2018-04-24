#!/usr/bin/env python

from reinforced_visual_slam.srv import *
import rospy

import sys
sys.path.insert(0, '/misc/lmbraid19/thomasa/catkin_ws/install/lib/python3/dist-packages')

from cv_bridge import CvBridge
from std_msgs.msg import Header

import cv2

import tensorflow as tf

from deepTAM.helpers2 import *
from deepTAM.common_netcode.myHelpers import *
from deepTAM.evaluation.helpers import *
from tfutils import optimistic_restore

class DepthmapPredictor(object):

    def __init__(self, input_img_width=640, input_img_height=480):
        rospy.init_node('depthmap_predictor')
        self._cv_bridge_rgb = CvBridge()
        self._cv_bridge_depth = CvBridge()
        rospy.loginfo("Opening tensorflow session")
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction=0.45
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    def configure(self, network_module_path, network_checkpoint, width=320, height=240):
        self.width = 320
        self.height = 240
        depthmap_module = load_myNetworks_module_noname(network_module_path)
        self.single_image_depth_net = depthmap_module.SingleImageDepthNetwork(batch_size=1, width=self.width, height=self.height)
        self.single_image_depth_outputs = self.single_image_depth_net.build_net(**self.single_image_depth_net.placeholders)
        self.session.run(tf.global_variables_initializer())
        optimistic_restore(self.session, network_checkpoint, verbose=False)
        rospy.logwarn("Finished loading the depthmap predictor network")

    def convertOpenCvRGBToTfTensor(self, rgb):
        rgb_new = cv2.normalize(rgb, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Resize Keyframe RGB image to (320W,240H) and reshape it to (3, height, width)
        rgb_new = cv2.resize(rgb_new, (self.width, self.height))
        rgb_new = np.transpose(rgb_new, (2,0,1))
        return rgb_new

    def predict_depthmap_cb(self, req, debug=False):
        rospy.loginfo("Received image to predict depthmap")
        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,240,320)
        # Get the Keyframe RGB image from ROS message and convert it to openCV Image format
        try:
            rgb_image = self._cv_bridge_rgb.imgmsg_to_cv2(req.rgb_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return PredictDepthmapResponse(0)
        if False:
            cv2.imshow('Input RGB', rgb_image)
        rgb_tensor = self.convertOpenCvRGBToTfTensor(rgb_image)
        feed_dict = {
            self.single_image_depth_net.placeholders['image']: rgb_tensor[np.newaxis, :, :, :],
        }
        self.single_image_depth_out = self.session.run(self.single_image_depth_outputs, feed_dict=feed_dict)
        # *** Network gives idepth prediction. Invert it if needed! ***
        depthmap_predicted = np.nan_to_num(self.single_image_depth_out['predict_depth'][0,0])

        # Doing cv_bridge conversion
        depthmap_predicted_msg = self._cv_bridge_depth.cv2_to_imgmsg(depthmap_predicted, "passthrough")
        depthmap_predicted_msg.step = int (depthmap_predicted_msg.step)
        if debug:
            print("depthmap_predicted max:", np.max(depthmap_predicted))
            print("depthmap_predicted min:", np.min(depthmap_predicted))
            print("depthmap_predicted.dtype", depthmap_predicted.dtype)
            print("depthmap_predicted.shape", depthmap_predicted.shape)
            plot_depth_gray = cv2.convertScaleAbs(depthmap_predicted*255./np.max(depthmap_predicted))
            depth_predicted_plot = cv2.cvtColor(plot_depth_gray, cv2.COLOR_GRAY2RGB)
            idepth = np.nan_to_num(1./depthmap_predicted)
            depth_predicted_plot[:,:,0] = 255 - cv2.convertScaleAbs((0-idepth)*255.)
            depth_predicted_plot[:,:,1] = 255 - cv2.convertScaleAbs((1-idepth)*255.)
            depth_predicted_plot[:,:,2] = 255 - cv2.convertScaleAbs((2-idepth)*255.)
            print("depthmap_predicted_msg.header:", depthmap_predicted_msg.header)
            print("depthmap_predicted_msg.step:", depthmap_predicted_msg.step)
            print("depthmap_predicted_msg.encoding:", depthmap_predicted_msg.encoding)
            cv2.imshow('Predicted Depthmap', depth_predicted_plot)
            cv2.waitKey(0)

        return PredictDepthmapResponse(depthmap_predicted_msg)

    def run(self):
        service = rospy.Service('predict_depthmap', PredictDepthmap, self.predict_depthmap_cb)
        print("Ready to predict depthmaps")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down depthmap predictor")

if __name__ == "__main__":
    predictor = DepthmapPredictor()
    predictor.configure('/misc/lmbraid19/thomasa/deep-networks/deepTAM/nets_multiframes/depth_singleimage/net/myNetworks.py', '/misc/lmbraid19/thomasa/deep-networks/deepTAM/nets_multiframes/depth_singleimage/training_v_sun3d/1_d_big/recovery_checkpoints/snapshot-391065')
    predictor.run()
