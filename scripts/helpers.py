#!/usr/bin/env python

from ctypes import *
import numpy as np
from minieigen import Quaternion
from collections import namedtuple

Keyframe = namedtuple('Keyframe', ['camToWorld', 'depth', 'rgb'])

class InputPointCloud(Structure):
    _fields_ = [("idepth", c_float),
                ("idepth_var", c_float),
                ("color", c_ubyte * 4)]

class FrameData(Structure):
	_fields_ = [("id", c_int),
				("camToWorld", c_float * 7)]

def quaternion_to_rotation_matrix(q_np):
	q_mini = Quaternion(*(np.roll(q_np, 1)))
	R = q_mini.toRotationMatrix()
	return np.asarray(R, dtype=np.float32)

def invert_transformation(R, t, scale=1):
	T = np.eye(4)
	T[:3, :3] = R
	T[:3, 3] = t
	# Uncomment below to convert scale of the current output.
	T_scale = np.eye(4)
	R_scale = np.eye(3)*scale
	T_scale[:3,:3] = R_scale
	T = np.dot(T_scale, T)
	Tinv = np.linalg.inv(T)
	Rinv = Tinv[:3,:3]
	tinv = Tinv[:3,3]
	return Rinv, tinv, T

def read_pointcloud_struct(pointcloud_data, arr_len):
	""" Reads point cloud struct array from python bytearray (bytes, to be specific)
		Struct is defined as InputPointCloud
		Args:
		pointcloud_data : bytes
		arr_len : array length (Input image width * Input image height) """
	ctype_arr = InputPointCloud * arr_len
	return ctype_arr.from_buffer_copy(pointcloud_data)

def read_frameData_struct(frame_data, arr_len):
	""" Reads frameData struct array from python bytearray (bytes, to be specific)
		Struct is defined as FrameData
		Args:
		frame_data : bytes
		arr_len : array length (Number of frames) """
	ctype_arr = FrameData * arr_len
	return ctype_arr.from_buffer_copy(frame_data)