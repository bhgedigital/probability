from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import math
from tensorflow_probability.python.mcmc import util as mcmc_util


#------------------------------------------------------------------------------
#--------------- Posterior Gaussian distributions -----------------------------

def posterior_Gaussian(L, Kmn, Knn, Y, fullCov):
	# function that provides mean and variance of
	# a conditional Gaussian:
	# Given normal distributions g1 and g2 with
	# g1 ~ N(0, Kmm)  with LL^t = Kmm
	# g2 ~ N(0, Knn)
	# Covariance(g1,g2) = Kmn
	# and Y observations of g1,
	# the output is the mean and variance of the conditional Gaussian
	# N(g2 | g1 = Y)
	# fullCov is a boolean specifying if a full covariance is computed or not
	# See  [Rasmussen] section 2.2


	alpha = tf.matrix_triangular_solve(L, Kmn, lower=True)

	# computing the variance
	if fullCov:
		var = Knn - tf.matmul(alpha, alpha, transpose_a=True)
	else:
		var = Knn - tf.reduce_sum(tf.square(alpha), 0)
		var = var[:,tf.newaxis]

	# computing the mean
	alpha = tf.matrix_triangular_solve(tf.transpose(L), alpha, lower=False)
	fmean = tf.matmul(alpha, Y, transpose_a=True)

	return fmean, var


def mixing_Covariance(K, Wmix, varm,varc):
	# K := Q x N1 x N2 kernels corresponding to independent Gaussians processes
	# Wmix := M X Q  mixing matrix
	# varm := M x Q array of variances

	Cov1 = Wmix[:,:,tf.newaxis, tf.newaxis]*K[tf.newaxis,:,:,:]*varc  # shape M x Q x N1 x N2
	     # shape M X Q x 1 X 1              # shape 1 X Q x N1 X N2

	Cov1  = tf.tensordot(Cov1, Wmix, [[1],[1]]) # Cov has shape M x N1 x N2 x M
	Cov1  = tf.transpose(Cov1, perm = [0,1,3,2]) # shape is now M x N1 x M x N2

	n_tasks = varm.shape[0].value

	D = tf.eye(n_tasks)
	Cov2 = D[:,:, tf.newaxis]*varm[tf.newaxis,:,:]   # shape M x M x Q
	Cov2 = tf.tensordot(Cov2, K,[[2],[0]]) # shape M x M x N1 x N2
	Cov2 = tf.transpose(Cov2,perm = [0,2,1,3] ) # shape M x N1 x M x N2

	Cov = Cov1 + Cov2

	return Cov

def mixing_Covariance_diag(K_expected, Wmix, varm, varc):
	# K_expected :=  expected kernels of shape Q
	# Wmix := M X Q  mixing matrix
	# varm := M x Q array of variances
	Cov = varc*tf.square(Wmix) + varm # shape M x Q
	Cov = tf.tensordot(Cov, K_expected, [[1],[0]])
	return Cov


#-------------------------------------------------------------------------------
#---------------  HMC sampling tools -------------------------------------------

def step_size_simple_update(step_size_var,kernel_results,target_rate=0.75, decrement_multiplier=0.01,increment_multiplier=0.01):
  if kernel_results is None:
    if mcmc_util.is_list_like(step_size_var):
      return [tf.identity(ss) for ss in step_size_var]
    return tf.identity(step_size_var)
  log_n = tf.log(tf.cast(tf.size(kernel_results.inner_results.log_accept_ratio),
                         kernel_results.inner_results.log_accept_ratio.dtype))
  log_mean_accept_ratio = tf.reduce_logsumexp(
      tf.minimum(kernel_results.inner_results.log_accept_ratio, 0.)) - log_n
  adjustment = tf.where(
      log_mean_accept_ratio < tf.log(target_rate),
      -decrement_multiplier / (1. + decrement_multiplier),
      increment_multiplier)
  if not mcmc_util.is_list_like(step_size_var):
    return step_size_var.assign_add(step_size_var * adjustment)
  step_size_assign = []
  for ss in step_size_var:
    step_size_assign.append(ss.assign_add(ss * adjustment))
  return step_size_assign
