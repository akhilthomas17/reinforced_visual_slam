#!/usr/bin/env python

def quaternion_to_rotation_matrix(q_np):
	import numpy as np
	from minieigen import Quaternion

	q_mini = Quaternion()
	for ii in range(4):
		q_mini[ii] = q_np[ii]
	R = q_mini.toRotationMatrix()

	return np.asarray(R, dtype=np.float32)

def invert_transformation(R, t):
	import numpy as np
	T = np.hstack((R,t[:,np.newaxis]))
	T = np.vstack((T,np.asarray([0,0,0,1])))
	Tinv = np.linalg.inv(T)
	Rinv = Tinv[:3,:3]
	tinv = Tinv[:3,3]
	return Rinv, tinv