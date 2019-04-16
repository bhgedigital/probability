from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
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
def samples_sq_distances(diff_samples, beta):
    # Inputs:
    #   diff_samples := array of shape n_samples x D containing samples representing difference of points
    #   beta         := array of shape 1 x D containing the inverse lengthscales
    # Outputs:
    #   dists_sq := array of shape n_samples containing the scaled square distances

    dists_sq = tf.reduce_sum(tf.square(diff_samples*beta), axis =-1)
    return dists_sq

def RBF_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    K = tf.exp(-r_squared/2.0)
    K_expected = tf.reduce_mean(K)
    return K_expected

def Matern12_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    r = tf.sqrt(r_squared)
    K = tf.exp(-r)
    K_expected = tf.reduce_mean(K)
    return K_expected

def Matern32_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    r = tf.sqrt(r_squared)
    sq3 = tf.constant(math.sqrt(3), tf.float32)
    K = (1.0 + sq3*r)*tf.exp(-sq3*r)
    K_expected = tf.reduce_mean(K)
    return K_expected

def Matern52_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    r = tf.sqrt(r_squared)
    sq5 = tf.constant(math.sqrt(5),tf.float32)
    K = (1.0 + sq5*r + 5.0*r_squared/3.0)*tf.exp(-sq5*r)
    K_expected = tf.reduce_mean(K)
    return K_expected

#------------------------------------------------------------------------------

kernel_mapping = {'RBF': [RBF, RBF_expected], 'Matern12': [Matern12, Matern12_expected], 'Matern32': [Matern32, Matern32_expected],
'Matern52': [Matern52, Matern52_expected]}
