#!/usr/bin/env python

from reinforced_visual_slam.srv import *
import rospy

# for cv_bridge
sys.path.insert(0, '/misc/lmbraid19/thomasa/catkin_ws/install/lib/python3/dist-packages')
#for cv2
sys.path.insert(0,'/misc/software/opencv/opencv-3.2.0_cuda8_with_contrib-x86_64-gcc5.4.0/lib/python3.5/dist-packages')


from cv_bridge import CvBridge
import cv2
from PIL import Image

from minieigen import Quaternion as quat
#from pyquaternion import Quaternion as Quat
from geometry_msgs.msg import Transform, Quaternion, Vector3
from std_msgs.msg import Bool

import tensorflow as tf

from depthmotionnet.vis import angleaxis_to_rotation_matrix
from depthmotionnet.dataset_tools.view_tools import adjust_intrinsics

from deepTAM.evaluation.helpers import *
from deepTAM.datatypes import View

from tfutils.helpers import optimistic_restore

class DeepTAMTracker(object):

    def __init__(self, input_img_width=640, input_img_height=480):
        rospy.init_node('deepTAM_tracker')
        self.debug = False
        
        state = rospy.Service('tracker_status', TrackerStatus, self.tracker_status_cb)
        self.status = False
        self.status_pub = rospy.Publisher('tracker_status', Bool, queue_size=1)
        self.status_pub.publish(Bool(False))

        rospy.loginfo("Configuring tensorflow session")
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction=0.2
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        self.session.run(tf.global_variables_initializer())
        self._cv_bridge = CvBridge()

        # Sun 3d normalized intrinsics
        self.sun3d_intrinsics = np.array([[0.89115971, 1.18821299, 0.5, 0.5]],dtype=np.float32)
        self.sun3d_K = np.eye(3)
        self.sun3d_K[0,0] = self.sun3d_intrinsics[0][0]*320
        self.sun3d_K[1,1] = self.sun3d_intrinsics[0][1]*240
        self.sun3d_K[0,2] = self.sun3d_intrinsics[0][2]*320
        self.sun3d_K[1,2] = self.sun3d_intrinsics[0][3]*240
        self.input_img_height = input_img_height
        self.input_img_width = input_img_width

    def configure(self, network_script_path, network_session_path):
        rospy.loginfo("Loading the deepTAM network")
        tracking_mod = load_myNetworks_module_noname(network_script_path)
        self.tracking_net = tracking_mod.TrackingNetwork()
        [self.image_height, self.image_width] = self.tracking_net.placeholders['image_key'].get_shape().as_list()[-2:]
        self.output = self.tracking_net.build_net(**self.tracking_net.placeholders)
        optimistic_restore(self.session, network_session_path, verbose=True)
        self.status = True
        self.status_pub.publish(Bool(True))

    def rotation_matrix_to_quaternion(self, R):
        q = Quaternion()
        q_mini = quat(R)
        q.x = q_mini[0]
        q.y = q_mini[1]
        q.z = q_mini[2]
        q.w = q_mini[3]
        if self.debug:
            print("Quaternion mini:", q.x, q.y, q.z, q.w)
        return q

    def angle_axis_to_quaternion(self, axis_a):
        q = Quaternion()
        q_mini = quat(axis_a,0)
        q.x = q_mini[0]
        q.y = q_mini[1]
        q.z = q_mini[2]
        q.w = q_mini[3]
        # return Quat(*(quat(axis=axis_a).elements))
        return q

    def convertCvRGBToTensor(self, rgb, width=320, height=240, normalize=True):
        if normalize:
            rgb_new = cv2.normalize(rgb, None, alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Resize Keyframe RGB image to (320W,240H) and reshape it to (3, height, width)
        rgb_new = cv2.resize(rgb_new, (width, height))
        rgb_new = np.transpose(rgb_new, (2,0,1))
        return rgb_new

    def convertPILRgbToAdjustedView(self, intrinsics, rgb, depth=None, width=320, height=240):
        # Convert Keyframe RGB to sun3d intrinsics and reshape to (width, height)
        K = np.eye(3)
        K[0,0] = intrinsics[0]
        K[1,1] = intrinsics[1]
        K[0,2] = intrinsics[2]
        K[1,2] = intrinsics[3]
        R = np.eye(3)
        t = np.array([0,0,0],dtype=np.float)

        if not depth is None:
            depth_metric = 'camera_z'
            if self.debug:
                print("Size Depth before:", depth.size)
        else:
            depth_metric = None


        if self.debug:
            print("Size RGB before:", rgb.size)
            print("K", K)
            print("width", width)
            print("height", height)
        
        rgb_view = View(R=R,t=t,K=K,image=rgb, depth=depth, depth_metric=depth_metric)
        new_rgb_view = adjust_intrinsics(rgb_view, self.sun3d_K, width, height)
        new_rgb = new_rgb_view.image

        if self.debug:
            print("Size RGB after:", new_rgb.size)
            if not depth is None:
                print("Size Depth after:", new_rgb_view.depth.size)
            #new_rgb.show()

        # Normalize Keyframe RGB image to range (-0.5, 0.5) and save as Float32
        new_rgb = np.array(new_rgb)[:, :, ::-1].transpose([2,0,1]).astype(np.float32)/255-0.5
        new_rgb_view = new_rgb_view._replace(image=new_rgb)
        
        if not depth is None:
            d = new_rgb_view.depth
            d[d <= 0] = np.nan
            new_rgb_view = new_rgb_view._replace(depth=d)


        del rgb_view
        return new_rgb_view


    def tracker_status_cb(self, req):
        return TrackerStatusResponse(Bool(self.status))

    def track_image_cb(self, req, plot_debug=False):
        print("Recieved image to track")

        # Load the intrinsics from the request
        intrinsics = np.array(req.intrinsics)
        if self.debug:
            print("intrinsics", intrinsics)

        # Requirement: Depth has to np array float32, in meters, with shape (1,1,240,320)
        # Get the Keyframe depth image and convert it to openCV Image format (Assuming Depth in Meters)
        try:
            keyframe_depth = self._cv_bridge.imgmsg_to_cv2(req.keyframe_depth, "passthrough")
        except CvBridgeError as e:
            print(e)
            return TrackImageResponse(0)

        # Resize Keyframe depth image to (320W,240H)
        #keyframe_depth = cv2.resize(keyframe_depth, (320, 240))

        # Convert Keyframe depth image to Float32 
        #keyframe_depth = keyframe_depth.astype('float32')

        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,240,320)
        # Get the Keyframe RGB image from ROS message and convert it to openCV Image format
        try:
            keyframe_image = self._cv_bridge.imgmsg_to_cv2(req.keyframe_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return TrackImageResponse(0)

        if plot_debug:
            cv2.imshow('CV Keyframe RGB', keyframe_image)
            depth_plot = (keyframe_depth*255/3.5).astype('uint8')
            cv2.imshow('CV Keyframe Depth',depth_plot)
            #keyframe_image.show()


        # Convert CV image to PIL image and resize
        keyframe_image = cv2.cvtColor(keyframe_image, cv2.COLOR_BGR2RGB)
        keyframe_image = Image.fromarray(keyframe_image)

        keyframe_image_view = self.convertPILRgbToAdjustedView(intrinsics, keyframe_image, keyframe_depth)
        keyframe_image = keyframe_image_view.image
        keyframe_depth = keyframe_image_view.depth

        # Find inverse depth
        key_inv_depth = 1/keyframe_depth
        if self.debug:
            print("inverse depth dtype", key_inv_depth.dtype)

        # Requirement: RGB has to np array float32, normalized in range (-0.5,0.5), with shape (1,3,240,320)
        # Get the Current RGB image from ROS message and convert it to openCV Image format
        try:
            current_image = self._cv_bridge.imgmsg_to_cv2(req.current_image, "passthrough")
        except CvBridgeError as e:
            print(e)
            return TrackImageResponse(0)
        
        if False:
            cv2.imshow('CV Current RGB', current_image)

        # Convert CV image to PIL image
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        #current_image = Image.fromarray(current_image).resize((320, 240))
        current_image = Image.fromarray(current_image)

        current_image_view = self.convertPILRgbToAdjustedView(intrinsics, current_image)
        current_image = current_image_view.image

        # Loading prior rotation and translation
        prev_rotation = np.array(req.rotation_prior)
        if self.debug:
            print("prev_rotation", prev_rotation)
        prev_translation = np.array(req.translation_prior)
        if self.debug:
            print("prev_translation", prev_translation)

        # Loading the feed dict with converted values:
        feed_dict = {
            self.tracking_net.placeholders['depth_key']: key_inv_depth[np.newaxis,np.newaxis,:,:],
            self.tracking_net.placeholders['image_key']: keyframe_image[np.newaxis,:,:,:],
            self.tracking_net.placeholders['image_current']: current_image[np.newaxis,:,:,:],
            self.tracking_net.placeholders['intrinsics']: self.sun3d_intrinsics,
            self.tracking_net.placeholders['prev_rotation']: prev_rotation[np.newaxis,:],
            self.tracking_net.placeholders['prev_translation']: prev_translation[np.newaxis,:],
        }

        output_arrs = self.session.run(self.output, feed_dict=feed_dict)

        t = output_arrs['predict_translation'][0]
        if self.debug:
            print("t = :", t)
        R = angleaxis_to_rotation_matrix(output_arrs['predict_rotation'][0])
        q = output_arrs['predict_rotation'][0]

        T = np.vstack((np.hstack((R,t[:,np.newaxis])), np.asarray([0,0,0,1])))
        Tinv = np.linalg.inv(T)
        Rinv = Tinv[:3,:3]
        tinv = Tinv[:3,3]

        # Uncomment below to find SE3 with respect to world coordinates
        #R_w = R.dot(angleaxis_to_rotation_matrix(prev_rotation))
        #print("R_w = :", R_w)
        #t_w = R.dot(prev_translation) + t
        #print("t_w = :", t_w)
        #print("Angle axis:", output_arrs['predict_rotation'][0])
        #q = self.angle_axis_to_quaternion(output_arrs['predict_rotation'][0])
        #print("Quaternion:", q)
        if self.debug:
            print("deepTAM_tracker: q worldToCam= :", q)
            print("deepTAM_tracker: t worldToCam = :", t)
            print("deepTAM_tracker: R worldToCam = :", R)
            print("deepTAM_tracker: t camToWorld = :", tinv)
            print("deepTAM_tracker: R camToWorld = :", Rinv)
            print("deepTAM_tracker: t camToWorld message:", Vector3(*tinv))
                
        rospy.loginfo("Returning tracked response")

        if True:
            #cv2.imshow('depth_map', output_arrs['rendered_depth'][0, 0])
            cv2.imshow('warped_img', output_arrs['warped_image'][0].transpose([1,2,0])+0.5)
            #cv2.imshow('depth_normalized0', output_arrs['depth_normalized0'][0, 0])
            #cv2.imshow('key_image0', output_arrs['key_image0'][0].transpose([1,2,0])+0.5)
            #cv2.imshow('current_image0', output_arrs['current_image0'][0].transpose([1,2,0])+0.5)
            cv2.imshow('diff', np.abs(output_arrs['warped_image'][0] - output_arrs['current_image0'][0]).transpose([1,2,0]))
            cv2.waitKey(100)

        return TrackImageResponse(Transform(Vector3(*tinv), self.rotation_matrix_to_quaternion(Rinv)))

    def run(self):
        service = rospy.Service('track_image', TrackImage, self.track_image_cb)
        rospy.loginfo("Ready to track images")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down deepTAM tracker")

if __name__ == "__main__":
    tracker = DeepTAMTracker()
    tracker.configure('/misc/lmbraid19/thomasa/deep-networks/deepTAM/nets_multiframes_uncertainties/tracking_scm_laplace_pairwise_fullres_nograd_fwd_multires/net/myNetworks.py',
      '/misc/lmbraid19/thomasa/deep-networks/deepTAM/nets_multiframes_uncertainties/tracking_scm_laplace_pairwise_fullres_nograd_fwd_multires/training/5_f1m1f2m2f3m3_frames2_new/checkpoints_new/snapshot-250000')
    tracker.run()
