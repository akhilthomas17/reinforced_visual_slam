#!/usr/bin/env python

import os
import sys

# for cv_bridge
sys.path.insert(0, '/misc/lmbraid19/thomasa/catkin_ws/install/lib/python3/dist-packages')
#for cv2
sys.path.insert(0,'/misc/software/opencv/opencv-3.2.0_cuda8_with_contrib-x86_64-gcc5.4.0/lib/python3.5/dist-packages')

import cv2
import numpy as np

#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

from reinforced_visual_slam.srv import *
import rospy
from cv_bridge import CvBridge

import tensorflow as tf
sys.path.insert(0,'/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion')
from net.my_models import *

class DepthmapFuser(object):

    def __init__(self, input_img_width=640, input_img_height=480):
        rospy.init_node('depthmap_fuser')
        self._cv_bridge_rgb = CvBridge()
        self._cv_bridge_depth = CvBridge()

    def configure(self, model_dir, model_function, width=320, height=240):
        self.width = width
        self.height = height
        self.lsd_depth_fuser = tf.estimator.Estimator(model_fn=model_function, model_dir=model_dir, 
            params={'data_format':"channels_first", 'multi_gpu':False}) 
        rospy.logwarn("Finished loading the depthmap fuser network")

    def convertOpenCvRGBToTfTensor(self, rgb, data_format="channels_first"):
        #rgb_new = cv2.normalize(rgb, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_new = rgb.astype(np.float32)/255
        # Resize Keyframe RGB image to (320W,240H) and reshape it to (3, height, width)
        rgb_new = cv2.resize(rgb_new, (self.width, self.height))
        if data_format=="channels_first":
            rgb_new = np.transpose(rgb_new, (2,0,1))
        return rgb_new

    def predict_input_fn(self, rgb, sparse_idepth, sparse_idepth_var):
        feature_names = [
          'rgb',
          'sparseInverseDepth',
          'sparseInverseDepthVariance'
        ]
        input_tensors = [rgb, sparse_idepth[np.newaxis,:,:], sparse_idepth_var[np.newaxis,:,:]]
        inputs = dict(zip(feature_names, input_tensors))
        #print(inputs['rgb'].shape)
        dataset = tf.data.Dataset.from_tensors(inputs)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        #print(features['rgb'].shape, features['sparseInverseDepth'].shape, 
              #features['sparseInverseDepthVariance'].shape)
        return features

    def fuse_depthmap_cb(self, req, debug=False):
        rospy.loginfo("Received image and sparse idepths to fuse")
        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,240,320)
        # Get the Keyframe RGB image from ROS message and convert it to openCV Image format
        try:
            rgb_image = self._cv_bridge_rgb.imgmsg_to_cv2(req.rgb_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return DepthFusionResponse(0)
        if debug:
            cv2.imshow('Input RGB', rgb_image)
        rgb_tensor = self.convertOpenCvRGBToTfTensor(rgb_image)
        
        # Get input sparse idepth and its variance. Convert it to gt scale (m)
        idepth_descaled = np.array(req.idepth, dtype=np.float32)/req.scale
        idepth_var_descaled = np.array(req.idepth_var, dtype=np.float32)/req.scale

        # Reshape and resize sparse idepths
        sparse_idepth = idepth_descaled.reshape((480, 640))
        sparse_idepth = cv2.resize(sparse_idepth, (320, 240), cv2.INTER_NEAREST)

        sparse_idepth_var = idepth_var_descaled.reshape((480, 640))
        sparse_idepth_var = cv2.resize(sparse_idepth_var, (320, 240), cv2.INTER_NEAREST)
        
        #***Get idepth prediction from net. Scale it back to input sparse idepth scale and invert it***
        idepth_predictions = self.lsd_depth_fuser.predict(lambda : 
            self.predict_input_fn(rgb_tensor, sparse_idepth, sparse_idepth_var))
        #idepthmap_scaled = list(idepth_predictions)[0]['depth'][0] * req.scale
        prediction = list(idepth_predictions)[0]['depth']
        depthmap_predicted = prediction[0]

        if prediction.shape[0] > 1:
            confidence = prediction[1]
        else:
            confidence = np.ones_like(depthmap_predicted)

        #** convert confidence to variance for LSD SLAM compatibality **#
        #** (approach 1): change k_res according to the confidence function used while training the network **#
        #k_res = 2.5
        #residual = -np.log(confidence)/k_res
        #** (approach 2): change minVar and maxVar as per need **#
        minVar = 0.0001
        maxVar = 0.125
        residual = maxVar - (maxVar - minVar)*confidence

        # Doing cv_bridge conversion
        depthmap_predicted_msg = self._cv_bridge_depth.cv2_to_imgmsg(depthmap_predicted, "passthrough")
        depthmap_predicted_msg.step = int (depthmap_predicted_msg.step)
        residual_msg = self._cv_bridge_depth.cv2_to_imgmsg(residual, "passthrough")
        residual_msg.step = int(residual_msg.step)
        if debug:
            print("depthmap_predicted max:", np.max(depthmap_predicted))
            print("depthmap_predicted min:", np.min(depthmap_predicted))
            print("depthmap_predicted.dtype", depthmap_predicted.dtype)
            print("depthmap_predicted.shape", depthmap_predicted.shape)
            plot_depth_gray = cv2.convertScaleAbs(depthmap_predicted*255./np.max(depthmap_predicted))
            depth_predicted_plot = cv2.cvtColor(plot_depth_gray, cv2.COLOR_GRAY2RGB)
            depth_predicted_plot[:,:,0] = 255 - cv2.convertScaleAbs((0-depthmap_predicted*req.scale)*255.)
            depth_predicted_plot[:,:,1] = 255 - cv2.convertScaleAbs((1-depthmap_predicted*req.scale)*255.)
            depth_predicted_plot[:,:,2] = 255 - cv2.convertScaleAbs((2-depthmap_predicted*req.scale)*255.)
            cv2.imshow('Predicted Depthmap', depth_predicted_plot)
            cv2.waitKey(100)
            ## uncomment below to plot with matplotlib
            #plt.figure("fused_depthmap (scale in m)")
            #plt.imshow(depthmap_predicted, cmap='hot')
            #plt.colorbar()
            #plt.show()

        return DepthFusionResponse(depthmap_predicted_msg, residual_msg)

    def run(self):
        service = rospy.Service('fuse_depthmap', DepthFusion, self.fuse_depthmap_cb)
        print("Ready to predict depthmaps")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down depthmap fuser")

if __name__ == "__main__":
    fuser = DepthmapFuser()

    # params to change for network:
    model_name = "NetV02_L1Sig4L1_down_tr1"
    #model_name = "NetV04Res_L1Sig4L1ExpResL1_down_tr2"
    #model_name = "NetV04Res_L1Sig4L1ExpResL1_down_tr1"
    #model_name = "NetV02_L1SigL1_down_aug_"
    #model_name = "NetV03_L1SigEqL1_down_aug"
    #model_function = model_fn_NetV04Res_LossL1SigL1ExpResL1
    model_function = model_fn_Netv2_LossL1SigL1_down
    #model_function = model_fn_Netv3_LossL1SigL1
    rospy.loginfo("Loading depth fusion model: %s", model_name)
    
    model_base_dir = "/misc/lmbraid19/thomasa/catkin_ws/src/reinforced_visual_slam/networks/depth_fusion/training"
    model_dir = os.path.join(model_base_dir, model_name)

    fuser.configure(
        model_dir=model_dir,
        model_function = model_function)
    fuser.run()
