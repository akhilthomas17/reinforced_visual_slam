#!/usr/bin/env python

import os
import sys

# for cv_bridge
sys.path.insert(0, '/misc/lmbraid19/thomasa/catkin_ws/install/lib/python3/dist-packages')
#for cv2
sys.path.insert(0,'/misc/software/opencv/opencv-3.2.0_cuda8_with_contrib-x86_64-gcc5.4.0/lib/python3.5/dist-packages')

import cv2
import numpy as np

from reinforced_visual_slam.srv import *
import rospy
from cv_bridge import CvBridge

import tensorflow as tf
sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')
from net.my_models import *

class DepthmapPredictor(object):

    def __init__(self, input_img_width=640, input_img_height=480):
        rospy.init_node('single_image_depthmap_predictor')
        self._cv_bridge_rgb = CvBridge()
        self._cv_bridge_depth = CvBridge()
        
    def configure(self, model_dir, model_function, run_config, width=320, height=240):
        self.width = width
        self.height = height
        self.lsd_depth_predictor = tf.estimator.Estimator(model_fn=model_function, model_dir=model_dir, 
            params={'data_format':"channels_first", 'multi_gpu':False},
            config=run_config) 
        rospy.logwarn("Finished loading the single image depth predictor network")

    def convertOpenCvRGBToTfTensor(self, rgb, data_format="channels_first"):
        #rgb_new = cv2.normalize(rgb, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_new = rgb.astype(np.float32)/255
        # Resize Keyframe RGB image to (320W,240H) and reshape it to (3, height, width)
        rgb_new = cv2.resize(rgb_new, (self.width, self.height))
        if data_format=="channels_first":
            rgb_new = np.transpose(rgb_new, (2,0,1))
        return rgb_new

    def predict_input_fn(self, rgb):
        feature_names = [
          'rgb'
        ]
        input_tensors = [rgb]
        inputs = dict(zip(feature_names, input_tensors))
        #print(inputs['rgb'].shape)
        dataset = tf.data.Dataset.from_tensors(inputs)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        #print(features['rgb'].shape, features['sparseInverseDepth'].shape, 
              #features['sparseInverseDepthVariance'].shape)
        return features

    def predict_depthmap_cb(self, req, debug=True):
        rospy.loginfo("Received image to predict depth")
        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,H,W)
        # Get the Keyframe RGB image from ROS message and convert it to openCV Image format
        try:
            rgb_image = self._cv_bridge_rgb.imgmsg_to_cv2(req.rgb_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return PredictDepthmapResponse(0)
        if False:
            cv2.imshow('Input RGB', rgb_image)
        rgb_tensor = self.convertOpenCvRGBToTfTensor(rgb_image)
        print("Input rgb shape:", rgb_tensor.shape)
                
        #***Get idepth prediction from net. Scale it back to input sparse idepth scale and invert it***
        idepth_predictions = self.lsd_depth_predictor.predict(lambda : 
            self.predict_input_fn(rgb_tensor))
        prediction = list(idepth_predictions)[0]['depth']
        print("prediction.shape:", prediction.shape)
        depthmap_predicted = prediction[0]
        #depthmap_predicted = np.where(depthmap_predicted > 0, 1. / depthmap_predicted, np.zeros_like(depthmap_predicted))

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
            depth_predicted_plot[:,:,0] = 255 - cv2.convertScaleAbs((0-depthmap_predicted)*255.)
            depth_predicted_plot[:,:,1] = 255 - cv2.convertScaleAbs((1-depthmap_predicted)*255.)
            depth_predicted_plot[:,:,2] = 255 - cv2.convertScaleAbs((2-depthmap_predicted)*255.)
            cv2.imshow('Predicted Depthmap', depth_predicted_plot)
            cv2.waitKey(0)
            ## uncomment below to plot with matplotlib
            #plt.figure("fused_depthmap (scale in m)")
            #plt.imshow(depthmap_predicted, cmap='hot')
            #plt.colorbar()
            #plt.show()

        return PredictDepthmapResponse(depthmap_predicted_msg)

    def run(self):
        service = rospy.Service('predict_depthmap', PredictDepthmap, self.predict_depthmap_cb)
        print("Ready to predict depthmaps")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down single image depth predictor")

if __name__ == "__main__":
    predictor = DepthmapPredictor()

    # params to change for network:
    model_name = "NetV0_L1SigL1_tr1"
    model_function = modelfn_NetV0_LossL1SigL1
    rospy.loginfo("Loading single image depth predictor model: %s", model_name)
    
    model_base_dir = "/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion/training"
    model_dir = os.path.join(model_base_dir, model_name)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    run_config = tf.estimator.RunConfig(session_config=config)

    predictor.configure(
        model_dir=model_dir,
        model_function = model_function,
        width=640, height=480,
        run_config=run_config)
    predictor.run()
