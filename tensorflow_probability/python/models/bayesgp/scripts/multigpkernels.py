from __future__ import print_function
from __future__ import absolute_import
from  __future__ import division

import math
import tensorflow as tf


def scaled_sq_dist(X1, X2, beta):
    # Computes the scaled square Euclidean distance between two arrays of points
    # This is used to compute LMC kernels
    # X1 : = array of shape N1 x D
    # X2 : = array of  shape N2 x D
    # beta := array of shape Q x D
    # output := array of shape Q x N1 x N2 where
    # D:=  dimension of input space
    # N1 := number of points for X1
    # N2 := number of points for X2
    # Q := components

    # rescaling
    # print(beta.shape)
    X1r = X1[tf.newaxis,:,:]*beta[:, tf.newaxis,:] # shape  Q x N1 x D
    X2r = X2[tf.newaxis,:,:]*beta[:, tf.newaxis,:] # shape  Q x N2 x D

    X1s  = tf.reduce_sum(tf.square(X1r), axis=2) # shape  Q x N1
    X2s  = tf.reduce_sum(tf.square(X2r), axis=2) # shape  Q x N2
    preset_rank = beta.shape[0].value

    dists_sq = tf.reshape(X1s, (preset_rank,-1, 1)) + tf.reshape(X2s, (preset_rank,1, -1))  -2 * tf.matmul(X1r, X2r, transpose_b=True)
                # shape  Q x N1 x 1                    # shape  Q x 1 x N2                         # shape  Q x N1 x N2

    # clipping
    dists_sq = tf.maximum(dists_sq,1e-30)

    return dists_sq



def RBF(X1, X2, beta):
	# computes the stationary RBF kernel exp(-(beta|x-x'|)^2/2)
    r_squared = scaled_sq_dist(X1, X2, beta)
    return tf.exp(-r_squared / 2.0)


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

#-------------------------------------------------------------------------------
#---- Functions used to compute kernels of the GP for the sensitivity analysis --
def samples_sq_distances(diff_samples, beta):
    # Inputs:
    #   diff_samples := array of shape n_samples x Dslice containing the samples for the point difference
    #   beta         := array of shape Q x Dslice containing the inverse lengthscales
    # Outputs:
    #   dists_sq :=     array of shape Q x n_samples containged the scaled square distances

    dists_sq = tf.reduce_sum(tf.square(diff_samples[tf.newaxis,:,:]*beta[:,tf.newaxis,:]), axis =2)
    return dists_sq

def RBF_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    K = tf.exp(-r_squared/2.0)  # shape Q x n_samples
    K_expected = tf.reduce_mean(K, axis = 1) # shape Q
    return K_expected

def Matern12_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    r = tf.sqrt(r_squared)
    K = tf.exp(-r)  # shape Q x n_samples
    K_expected = tf.reduce_mean(K, axis = 1) # shape Q
    return K_expected

def Matern32_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    r = tf.sqrt(r_squared)
    sq3 = tf.constant(math.sqrt(3), tf.float32)
    K = (1.0 + sq3*r)*tf.exp(-sq3*r)  # shape Q x n_samples
    K_expected = tf.reduce_mean(K, axis = 1) #shape Q
    return K_expected

def Matern52_expected(diff_samples, beta):
    r_squared = samples_sq_distances(diff_samples, beta)
    r = tf.sqrt(r_squared)
    sq5 = tf.constant(math.sqrt(5),tf.float32)
    K = (1.0 + sq5*r + 5.0*r_squared/3.0)*tf.exp(-sq5*r) # shape Q x n_samples
    K_expected = tf.reduce_mean(K, axis = 1) # shape Q
    return K_expected




#------------------------------------------------------------------------------



kernel_mapping = {'RBF': [RBF, RBF_expected], 'Matern12': [Matern12, Matern12_expected], 'Matern32': [Matern32,Matern32_expected],
'Matern52': [Matern52,Matern52_expected]}
