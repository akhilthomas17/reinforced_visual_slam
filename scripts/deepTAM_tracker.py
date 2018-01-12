#!/usr/bin/env python

from dummy_tf_node.srv import *
import rospy
import tensorflow as tf
from cv_bridge import CvBridge
import cv2
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

    def track_image_cb(self, req):
        print("Returning response")
	keyframe_depth = self._cv_bridge.imgmsg_to_cv2(req.keyframe_depth, "bgr8")
        key_inv_depth = 1/keyframe_depth
        feed_dict = {
            self.tracking_net.placeholders['depth_key']: key_inv_depth[np.newaxis,np.newaxis,:,:],
            self.tracking_net.placeholders['image_key']: req.keyframe_image[np.newaxis,:,:,:],
            self.tracking_net.placeholders['image_current']: req.current_image[np.newaxis,:,:,:],
            self.tracking_net.placeholders['intrinsics']: req.intrinsics,
            self.tracking_net.placeholders['prev_rotation']: req.rotation_prior,
            self.tracking_net.placeholders['prev_translation']: req.translation_prior,
        }
#        if 'depth_current' in self.tracking_net.placeholders:
#             feed_dict[tracking_net.placeholders['depth_current']] = 1/req.keyframe_depth[np.newaxis,np.newaxis,:,:]

        output_arrs = self.session.run(output, feed_dict=feed_dict)
        t = output_arrs['predict_translation'][0]
        R = angleaxis_to_rotation_matrix(output_arrs['predict_rotation'][0])
        R_w = R.dot(req.rotation_prior)
        t_w = R.dot(req.translation_prior) + t
        
        return TrackImageResponse(1)

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
