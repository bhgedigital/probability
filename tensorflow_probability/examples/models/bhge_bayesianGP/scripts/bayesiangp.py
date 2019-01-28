import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import time
import functools
import math
from tensorflow_probability.python.mcmc import util as mcmc_util
ds = tf.contrib.distributions


tfd = tfp.distributions
tfb = tfp.bijectors

#------------------------------------------------------------------------------
# This script implements a Gaussian process where the posterior distribution of
# the hyperparameters are obtained using MCMC sampling.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#------------ References ------------------------------------
# - [Rasmussen] "Gaussian process for Machine Learning", C. E. Rasmussen and C. K. I. Williams, MIT press
#-------------------------------------------------------------------------------

#--------- Some notation --------------------#
#  beta  := array containing the inverse lengthscales of the kernel.
#			The model uses an ARD (Automatic Relevance Determination) kernel. There
#			is one inverse lengthscale for each input variable.
# loc    := the constant mean function of the Gaussian process
# varm := the variance of the Gaussian process

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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# The function "step_size_simple_update" is from the tensorflow probabilty github page
# and has been slightly modified

def step_size_simple_update(
    step_size_var,
    kernel_results,
    target_rate=0.75,
    decrement_multiplier=0.01,
    increment_multiplier=0.01):
  """Updates (list of) `step_size` using a standard adaptive MCMC procedure.
  This function increases or decreases the `step_size_var` based on the average
  of `exp(minimum(0., log_accept_ratio))`. It is based on
  [Section 4.2 of Andrieu and Thoms (2008)](
  http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf).
  Args:
    step_size_var: (List of) `tf.Variable`s representing the per `state_part`
      HMC `step_size`.
    kernel_results: `collections.namedtuple` containing `Tensor`s
      representing values from most recent call to `one_step`.
    target_rate: Scalar `Tensor` representing desired `accept_ratio`.
      Default value: `0.75` (i.e., [center of asymptotically optimal
      rate](https://arxiv.org/abs/1411.6669)).
    decrement_multiplier: `Tensor` representing amount to downscale current
      `step_size`.
      Default value: `0.01`.
    increment_multiplier: `Tensor` representing amount to upscale current
      `step_size`.
      Default value: `0.01`.
  Returns:
    step_size_assign: (List of) `Tensor`(s) representing updated
      `step_size_var`(s).
  """
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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------- Bayesian GP model -----------------------------------------------------

kernel_mapping = {'RBF': RBF, 'Matern12': Matern12, 'Matern32': Matern32,
				'Matern52': Matern52}

class BayesianGP():
	def __init__(self, inputs, outputs, kernel_type, noise_level):

		#------- Converting the data to tensors and storing -------------
		if len(inputs.shape) == 1:
			self.dim_input = 1
			self.Xtrain = tf.convert_to_tensor(inputs[:,None], tf.float32)
		else:
			self.dim_input = inputs.shape[1]
			self.Xtrain = tf.convert_to_tensor(inputs, tf.float32)
		self.Ytrain = tf.convert_to_tensor(outputs, tf.float32)
		self.n_train = len(outputs)   # the number of outputs

		#------ Priors for the Gaussian process model-----------
		# Priors on the inverse lengthscale
		self.rv_beta = tfd.Independent(tfd.Gamma(concentration = tf.ones(self.dim_input), rate = tf.ones(self.dim_input)),
                           reinterpreted_batch_ndims=1, name='rv_beta')

		# prior on the mean
		self.rv_loc = tfd.Normal(loc = 0.0, scale = 0.5, name = 'rv_loc')

		# prior on the variance
		self.rv_varm = tfd.Gamma(concentration = 1.0, rate = 1.0,  name = 'rv_varm')

		# prior on the noise variance
		self.rv_noise = tfd.LogNormal(loc = -6.9, scale = 1.5, name = 'rv_noise')

		#-- value for the  variance of the Gaussian noise (when kept fixed)
		self.noise=  np.float32(noise_level)

		self.jitter_level = 1e-6 # jitter level to deal with numerical instability with cholesky factorization

		self.kernel = kernel_mapping[kernel_type]

		return


	def joint_log_prob(self, noise, beta, varm, loc):
		# function for computing the joint log probability of the Gaussian process
		# model  given values for the inverse lengthscales,
		# the variance of the kernel and  the constant mean of the Gaussian process

		#------ forming the kernels -----------------
		Kxx = self.kernel(self.Xtrain, self.Xtrain, beta)
		Cov_train = varm*Kxx + (noise + self.jitter_level)*tf.eye(self.n_train)

		#-------- Computing the cholesky factor ------------
		Cov_temp = tf.cast(Cov_train, tf.float64)
		L_temp = tf.cholesky(Cov_temp)
		L = tf.cast(L_temp, tf.float32)


		#---- Multivariate normal random variable for the combined outputs -----
		mean = loc*tf.ones(self.n_train)
		rv_observations = tfd.MultivariateNormalTriL(loc = mean, scale_tril = L )

		#--- Collecting the log_probs from the different random variables
		sum_log_prob = (rv_observations.log_prob(self.Ytrain)
					 + self.rv_beta.log_prob(beta)
					 +  self.rv_varm.log_prob(varm)
					 +  self.rv_loc.log_prob(loc))

		return sum_log_prob



	def warmup(self, initial_state, num_warmup_iters, num_leapfrog_steps, display_rate = 500):
		# function to generate an adaptive step size that will be needed for
		# HMC sampling

		# Inputs:
		# 	initial_state := list of tensors providing the initial state for the HMC sampler
		# 	num_warmup_iters := number of sampling steps to perfom during the warm-up
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# 	display_rate := rate at which information is printed
		# Outputs:
		# 	step_size_ := estimated value for the step size of the HMC sampler
		#	loc_next_, beta_next_, varm_next_ := last sample values obtained from the warm-up
		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, self.noise)


		#------- Unconstrained representation---------
		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(),tfb.Identity()]

		target_accept_rate = 0.651


		# Setting up the step_size
		step_size = tf.Variable(0.01, name = 'step_size')

		beta, varm, loc = initial_state

		beta_cur = tf.Variable(beta, name = 'beta_cur')
		varm_cur = tf.Variable(varm, name = 'varm_cur')
		loc_cur = tf.Variable(loc, name = 'loc_cur')

		current_state = [beta_cur, varm_cur,loc_cur]


		# Initializing the sampler
		sampler = tfp.mcmc.TransformedTransitionKernel(
						inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
								target_log_prob_fn=unnormalized_posterior_log_prob,
								step_size= step_size,
								num_leapfrog_steps=num_leapfrog_steps),
						bijector=unconstraining_bijectors)

		# One step of the sampler
		[
			beta_next,
			varm_next,
			loc_next
		], kernel_results = sampler.one_step(current_state = current_state,
											previous_kernel_results=sampler.bootstrap_results(current_state))

		# updating the step size
		step_size_update = step_size_simple_update(step_size, kernel_results,
													target_rate = target_accept_rate,
													decrement_multiplier = 0.1,
													increment_multiplier = 0.1)

		# Updating the state
		beta_update = beta_cur.assign(beta_next)
		varm_update = varm_cur.assign(varm_next)
		loc_update = loc_cur.assign(loc_next)

		warmup = tf.group([beta_update, varm_update,loc_update, step_size_update])


		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			print('Warmup: ')
			start = time.time()
			for i in range(num_warmup_iters):
				sess.run(warmup)
				if i % display_rate == 0:
					print('Step ', i)
			end = time.time()
			print('Time per step in warm up: ',(end- start)/num_warmup_iters)
			[step_size_, beta_next_, varm_next_, loc_next_] = sess.run([step_size, beta_next, varm_next, loc_next])

		return step_size_, beta_next_, varm_next_, loc_next_



	def mcmc(self, mcmc_samples, num_burnin_steps, initial_state, step_size, num_leapfrog_steps):
		# Function used to perform the sampling for the posterior distributions of the hyperparameters

		# Inputs:
		#	mcmc_samples := number of samples to collect for the hyperparameters
		#	num_burnin_steps := number of samples to discard
		# 	initial_state := list of tensors providing the initial state for the HMC sampler
		# 	step_size := step_size for the HMC sampler
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# Outputs:
		#	loc_samples_, varm_samples, beta_samples_ := samples for the posterior
		#									distribution of the hyperparameters
		#   acceptance_rate_ := acceptance rate of the sampling

		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, self.noise)

		#------- Unconstrained representation---------
		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Identity()]


		#----Setting up the mcmc sampler
		[
			beta_samples,
			varm_samples,
			loc_samples
		], kernel_results = tfp.mcmc.sample_chain(num_results= mcmc_samples, num_burnin_steps= num_burnin_steps,
																current_state=initial_state,
																kernel=tfp.mcmc.TransformedTransitionKernel(
																	inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= step_size,
																				num_leapfrog_steps=num_leapfrog_steps),
																	bijector=unconstraining_bijectors))



		acceptance_rate = tf.reduce_mean(tf.to_float(kernel_results.inner_results.is_accepted))


		with tf.Session() as sess:
			[
				acceptance_rate_,
				loc_samples_,
				varm_samples_,
				beta_samples_
			] = sess.run(
			[
				acceptance_rate,
				loc_samples,
				varm_samples,
				beta_samples
			])

		print('Acceptance rate of the HMC sampling:', acceptance_rate_)
		return loc_samples_, varm_samples_, beta_samples_, acceptance_rate_

	def EM_with_MCMC(self, initial_state, num_warmup_iters, em_iters, mcmc_samples, num_leapfrog_steps, learning_rate = 0.01, display_rate = 200):
		# Function used to estimate a value for the noise variance and obtain
		# the samples for the posterior distribution of the hyperparameters
		# Inputs:
		# 	initial_state := list of tensors providing the initial state for the mcmc sampler
		# 	num_warmup_iters := number of iterations for the warm-up phase
		# 	em_iters := number of iterations for the EM phase
		# 	mcmc_samples := number of samples to collect for the hyperparameters
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# 	learning_rate := learning rate for the optimizer used in the M-step
		# 	display_rate := rate at which information is printed
		# Outputs:
		#	loc_samples_, varm_samples, beta_samples_ := samples for the posterior
		#									distribution of the hyperparameters
		#   acceptance_rate_ := acceptance rate of the sampling
		#   loss_history := array containing values of the loss fucntion of the M-step
		#   noise_history := array containing values of the noise variance computed in the M-step

		# defining unconstrained version for the noise level
		unc_noise = tf.Variable(tf.log(tf.exp(1e-3)- 1), name = 'unc_noise')

		# Setting up the step_size and targeted acceptance rate for the MCMC part
		step_size = tf.Variable(0.01, name = 'step_size')
		target_accept_rate = 0.651

		beta, varm, loc = initial_state

		beta_cur = tf.Variable(beta, name = 'beta_cur', trainable = False)
		varm_cur = tf.Variable(varm, name = 'varm_cur', trainable = False)
		loc_cur = tf.Variable(loc, name = 'loc_cur', trainable = False)

		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(),tfb.Identity()]


		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, tf.nn.softplus(unc_noise))

		current_state = [beta_cur, varm_cur,loc_cur]

		# Initializing a sampler for warmup:
		sampler = tfp.mcmc.TransformedTransitionKernel(
						inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
								target_log_prob_fn=unnormalized_posterior_log_prob,
								step_size= step_size,
								num_leapfrog_steps=num_leapfrog_steps),
						bijector=unconstraining_bijectors)

		# One step of the sampler
		[
			beta_next,
			varm_next,
			loc_next
		], warmup_kernel_results = sampler.one_step(current_state = current_state,
											previous_kernel_results=sampler.bootstrap_results(current_state))

		# updating the step size
		step_size_update = step_size_simple_update(step_size, warmup_kernel_results,
													target_rate = target_accept_rate,
													decrement_multiplier = 0.1,
													increment_multiplier = 0.1)


		# Updating the state of the hyperparameters
		beta_update1 = beta_cur.assign(beta_next)
		varm_update1 = varm_cur.assign(varm_next)
		loc_update1 = loc_cur.assign(loc_next)

		warmup_update = tf.group([beta_update1, varm_update1,loc_update1, step_size_update])

		# Set up E-step with MCMC
		[
			beta_probs,
			varm_probs,
			loc_probs
		], em_kernel_results = tfp.mcmc.sample_chain(num_results= 10, num_burnin_steps= 10,
																current_state=initial_state,
																kernel=tfp.mcmc.TransformedTransitionKernel(
																	inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= step_size,
																				num_leapfrog_steps=num_leapfrog_steps),
																	bijector=unconstraining_bijectors))


		# Updating the state of the hyperparameters
		beta_update2 = beta_cur.assign(tf.reduce_mean(beta_probs, axis = 0))
		varm_update2 = varm_cur.assign(tf.reduce_mean(varm_probs, axis = 0))
		loc_update2 = loc_cur.assign(tf.reduce_mean(loc_probs, axis = 0))

		expectation_update = tf.group([beta_update1, varm_update1,loc_update1])

		#-- Set up M-step (updating noise variance)
		with tf.control_dependencies([expectation_update]):
			loss = -self.joint_log_prob(tf.nn.softplus(unc_noise),beta_cur, varm_cur, loc_cur) -self.rv_noise.log_prob(tf.nn.softplus(unc_noise))

			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			minimization_update = optimizer.minimize(loss)

		# Collecting samples after estimating the noise variance
		[
			beta_samples,
			varm_samples,
			loc_samples
		], sampling_kernel_results = tfp.mcmc.sample_chain(num_results= mcmc_samples, num_burnin_steps= 0,
																current_state=initial_state,
																kernel=tfp.mcmc.TransformedTransitionKernel(
																	inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= 0.8*step_size,
																				num_leapfrog_steps=num_leapfrog_steps),
																	bijector=unconstraining_bijectors))


		acceptance_rate = tf.reduce_mean(tf.to_float(sampling_kernel_results.inner_results.is_accepted))

		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)

		# Initial  warm-up stage
		print('Warm-up phase.')
		num_accepted = 0
		for t in range(num_warmup_iters):
			_, is_accepted_val = sess.run([warmup_update, warmup_kernel_results.inner_results.is_accepted])
			num_accepted +=  is_accepted_val
			if (t % display_rate == 0) or ( t == num_warmup_iters -1):
		  		print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, num_accepted / (t + 1)))

		loss_history = np.zeros([em_iters])
		noise_history = np.zeros([em_iters])

		print('Estimating the noise variance: ')
		for t in range(em_iters):


			[
			_,
			_,
			unc_noise_,
			loss_
			] = sess.run([
			expectation_update,
			minimization_update,
			unc_noise,
			loss
			])

			loss_history[t] = loss_
			noise_history[t] = np.log(np.exp(unc_noise_) + 1)
			if (t % display_rate == 0) or ( t == em_iters -1):
				print("Iteration: {:>4} Loss: {:.3f}".format(t, loss_))

		# Collecting samples
		print('Collecting samples for the GP hyperparameters.')
		[
			acceptance_rate_,
			loc_samples_,
			varm_samples_,
			beta_samples_
		] = sess.run(
		[
			acceptance_rate,
			loc_samples,
			varm_samples,
			beta_samples
		])

		print('Acceptance rate of the HMC sampling:', acceptance_rate_)

		sess.close()
		self.noise = np.log(np.exp(unc_noise_) + 1)
		self.noise = self.noise.astype(np.float32)

		return loc_samples_, varm_samples_, beta_samples_, acceptance_rate_, loss_history, noise_history


	def posteriormeanVariance(self, Xtest, hyperpars, fullCov = False):
		# Function generates posterior mean and variance for the Gaussian process, given values for the hyperparameters
		# Inputs:
		# 	Xtest := N x D tensorflow array of new inputs
		# Output
		#	mean_and_var := array where the first column is an N x 1 array representing the
		#					mean of the posterior Gaussian  distribution and the remaining columns correspond to
		#        			the (Co)variance which is an N x N array if
		#					fullCov = True or a N x 1 array if fullCov = False

		loc = hyperpars[0]
		varm = hyperpars[1]
		beta = hyperpars[2:]

		# ------- generate covariance matrix for training data and computing the corresponding cholesky factor
		Kxx = self.kernel(self.Xtrain, self.Xtrain, beta)
		Cov_train = varm*Kxx +  (self.noise + self.jitter_level)*tf.eye(self.n_train)
		Cov_temp = tf.cast(Cov_train, tf.float64)
		L_temp = tf.cholesky(Cov_temp)
		L = tf.cast(L_temp, tf.float32)

		#-------- generate covariance matrix for test data
		n_test = Xtest.shape[0].value
		if fullCov:
			Kx2 = self.kernel(Xtest, Xtest, beta)
			Cov_test = varm*Kx2 +  (self.noise + self.jitter_level)*tf.eye(n_test)
		else:
			Cov_test = (varm + self.noise + self.jitter_level )*tf.ones(n_test)


		#------- covariance between test data and training data
		Kx3 = self.kernel(self.Xtrain, Xtest, beta)
		Cov_mixed = varm*Kx3

		Y = self.Ytrain[:, tf.newaxis] - loc

		mean_pos, var_pos = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, fullCov)

		mean_pos = mean_pos + loc

		mean_and_var = tf.concat([mean_pos, var_pos], axis = 1)

		return mean_and_var

	def samples(self, Xtest, hyperpar_samples, num_samples = 20, with_point_samples = False):
		# Computes approximate values of the full posterior mean and variance of the Gaussian process
		# by using samples of the posterior distribution of the hyperparameters
		#Inputs:
		#	Xtest :=  N x D input array
		# 	hyperpar_samples := list of numpy arrays contaning samples for the hyperparameters
		#	with_point_samples = Boolean specifying if we sample from the posterior Gaussian
		#                       for each input and for each sample of the hyperparameters
     	#	num_samples := the number of points to sample from the posterior Gaussian (if with_point_samples == True)
		# Outputs:
		#	mean_pos_ := the mean for the full posterior Gaussian process (vector of length N)
		#  var_pos_ := the variance for the full posterior Gaussian process (vector of length N)
		# if with_point_samples == True, the function also outputs:
		# 	samples_ = samples for the full posterior Gaussian process
		if len(Xtest.shape) == 1:
			Xtest = Xtest[:,None]
		Xtest = tf.convert_to_tensor(Xtest, tf.float32)

		f = lambda hyperpars_in: self.posteriormeanVariance(Xtest, hyperpars_in, False)

		loc_samples, varm_samples, beta_samples = hyperpar_samples
		hyperpars = np.concatenate([loc_samples[:,None], varm_samples[:,None], beta_samples], axis = 1)
		hyperpars = tf.convert_to_tensor(hyperpars, tf.float32)

		mean_and_var = tf.map_fn(f,hyperpars)

		mean_pos_samples = mean_and_var[:,:,0]
		var_pos_samples = mean_and_var[:,:,1]
		var_pos_samples = tf.maximum(var_pos_samples,1e-30)


		if with_point_samples:
			rv_norm  = tfd.Normal(loc = mean_pos_samples, scale = tf.sqrt(var_pos_samples))
			samples = rv_norm.sample(num_samples)

			with tf.Session() as sess:
					[mean_pos_samples_, var_pos_samples_, samples_] = sess.run([mean_pos_samples, var_pos_samples, samples])


			mean_pos_ = np.mean(mean_pos_samples_, axis =0)
			var_pos_ = np.mean(var_pos_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)
			samples_ = np.concatenate(samples_, axis = 0)

			return mean_pos_, var_pos_, samples_

		else:
			with tf.Session() as sess:
					[mean_pos_samples_, var_pos_samples_] = sess.run([mean_pos_samples, var_pos_samples])


			mean_pos_ = np.mean(mean_pos_samples_, axis =0)
			var_pos_ = np.mean(var_pos_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)

			return mean_pos_, var_pos_
