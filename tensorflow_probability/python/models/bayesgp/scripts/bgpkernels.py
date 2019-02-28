from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math


#-------------------------------------------------------------------------------
#------ Functions for computing the kernel of the GP----------------------------
# For now, the code includes implementations of the RBF and the standard Matern
# kernels.


def scaled_sq_dist(X1, X2, beta):
	# Computes the scaled square Euclidean distance between two arrays of points
	# Inputs:
	# 	X1 : = input array of shape N1 x D
	# 	X2 : = input array of  shape N2 x D
	# 	beta := array of inverse lengthscales
	# Outputs:
	# 	dists := array of shape N1 x N2 where dists[i,j] is the scaled square Euclidean
	#       	distance between X1[i,:] and X2[j,:] for each pair (i,j)

	# scaling
	X1r = X1*beta
	X2r = X2*beta

	X1s  = tf.reduce_sum(tf.square(X1r), axis=1)
	X2s  = tf.reduce_sum(tf.square(X2r), axis=1)

	dists = tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))  -2 * tf.matmul(X1r, X2r, transpose_b=True)

	dists = tf.maximum(dists,1e-30)

	return dists


def RBF(X1, X2, beta):
	# computes the RBF kernel exp(-(beta|x-x'|)^2/2)
	return tf.exp(-scaled_sq_dist(X1, X2, beta) / 2.0)


def Matern12(X1, X2, beta):
    # computes the Matern 1/2 kernel
	r_squared = scaled_sq_dist(X1, X2, beta)
	r = tf.sqrt(r_squared)
	return tf.exp(-r)

def Matern32(X1, X2, beta):
    # computes the Matern 3/2 kernel
	r_squared = scaled_sq_dist(X1, X2, beta)
	r = tf.sqrt(r_squared)
	c_3 = tf.constant(math.sqrt(3.0),tf.float32)
	return (1.0 + c_3*r)*tf.exp(-c_3*r)

def Matern52(X1, X2, beta):
    # computes the Matern 5/2 kernel
	r_squared = scaled_sq_dist(X1, X2, beta)
	r = tf.sqrt(r_squared)
	c_5 = tf.constant(math.sqrt(5.0),tf.float32)
	return (1.0 + c_5*r + 5.0*r_squared/3.0)*tf.exp(-c_5*r)


#-------------------------------------------------------------------------------
#---- Functions used to compute kernels of the GP for the sensitivity analysis --

def scaled_sq_dist_aug(X1, X2, beta, diag = False):
	# Pretty much the same as the function scaled_sq_dist but here X1 and X2 are
	# 3-dimensional arrays of same length
	# If diag == False, for each i we compute the scaled square distance of the points X1[i,:,:] with respect
	# to the points X2[i,:,:]. It is assumed that X1 and X2 have shape N_b x N1 x D and   N_b x N2 x D respectively
	# Otherwise for each i and for each j, we compute the scaled square distance of the point
	# X1[i,j,:] with respect to X2[i,j,:]. It is assumed that X1 and X2 have the same shape.

	if diag == False:
		# rescaling
		X1r = X1*(beta[tf.newaxis, tf.newaxis,:])
		X2r = X2*(beta[tf.newaxis, tf.newaxis,:])

		X1s  = tf.reduce_sum(tf.square(X1r), axis=2)
		X2s  = tf.reduce_sum(tf.square(X2r), axis=2)
		n = X1.shape[0].value

		dists = tf.reshape(X1s, (n,-1, 1)) + tf.reshape(X2s, (n,1, -1))  -2 * tf.matmul(X1r, X2r, transpose_b=True)
		# clipping
		dists = tf.maximum(dists,1e-30)
	else:
		dists = tf.reduce_sum(tf.square((X1-X2)*(beta[tf.newaxis, tf.newaxis,:])), axis = 2)

	return dists


def RBF_aug(X1, X2, beta, diag = False):
	return tf.exp(-scaled_sq_dist_aug(X1, X2, beta, diag) / 2.0)

def Matern12_aug(X1, X2, beta, diag = False):
    # computes the Matern 1/2 kernel
	r_squared = scaled_sq_dist_aug(X1, X2, beta, diag)
	r = tf.sqrt(r_squared)
	return tf.exp(-r)

def Matern32_aug(X1, X2, beta, diag = False):
    # computes the Matern 3/2 kernel
	r_squared = scaled_sq_dist_aug(X1, X2, beta, diag)
	r = tf.sqrt(r_squared)
	sq3 = tf.constant(math.sqrt(3), tf.float32)
	return (1.0 + sq3*r)*tf.exp(-sq3*r)

def Matern52_aug(X1, X2, beta, diag = False):
    # computes the Matern 5/2 kernel
	r_squared = scaled_sq_dist_aug(X1, X2, beta)
	r = tf.sqrt(r_squared)
	sq5 = tf.constant(math.sqrt(5),tf.float32)
	return (1.0 + sq5*r + 5.0*r_squared/3.0)*tf.exp(-sq5*r)



kernel_mapping = {'RBF': [RBF, RBF_aug], 'Matern12': [Matern12, Matern12_aug], 'Matern32': [Matern32, Matern32_aug],
				'Matern52': [Matern52, Matern52_aug]}
