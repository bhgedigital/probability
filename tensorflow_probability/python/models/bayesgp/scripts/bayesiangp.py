from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import functools
import math
ds = tf.contrib.distributions
from tensorflow_probability.python.mcmc import sample_chain, HamiltonianMonteCarlo, TransformedTransitionKernel
from tensorflow_probability.python import distributions  as tfd
from  tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.models.bayesgp.scripts.bgpkernels import *
from tensorflow_probability.python.models.bayesgp.scripts.bgp_utils import posterior_Gaussian, step_size_simple_update


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

class BayesianGP():
	def __init__(self, inputs, outputs, kernel_type, noise_level = 1e-3):
		# Inputs:
		#	inputs := numpy array of inputs
		# 	outpust:= numpy array of outputs
		# kernel_type := string specifying the type of kernel to be used. Options are
		#                'RBF', 'Matern12', 'Matern32', 'Matern52'
		# noise_level := value for the noise variance of the output

		#------- Storing training data-------------
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
		self.rv_beta = tfd.Independent(tfd.Gamma(concentration = tf.ones(self.dim_input, tf.float32), rate = tf.ones(self.dim_input, tf.float32)),
                           reinterpreted_batch_ndims=1, name='rv_beta')

		# prior on the mean
		self.rv_loc = tfd.Normal(loc = 0.0, scale = 0.5, name = 'rv_loc')

		# prior on the variance
		self.rv_varm = tfd.Gamma(concentration = 1.0, rate = 1.0,  name = 'rv_varm')

		# prior on the noise variance
		self.rv_noise = tfd.LogNormal(loc = -6.9, scale = 1.5, name = 'rv_noise')

		#-- value for the  variance of the Gaussian noise
		self.noise=  noise_level

		self.jitter_level = 1e-6 # jitter level to deal with numerical instability with cholesky factorization

		self.kernel, self.aug_kernel = kernel_mapping[kernel_type]

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



	def warmup(self, num_warmup_iters, num_leapfrog_steps, initial_state = None, display_rate = 500):
		# function to generate an adaptive step size that will be needed for
		# HMC sampling

		# Inputs:
		# 	num_warmup_iters := number of sampling steps to perfom during the warm-up
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# initial_state := list ([beta, varm, loc]) of tensors providing the initial state for the HMC sampler
		# 	display_rate := rate at which information is printed
		# Outputs:
		# 	step_size_ := estimated value for the step size of the HMC sampler
		#	next_state = list of the form [beta_next_, varm_next_, loc_next_] that constains the last sample values obtained from the warm-up
		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, self.noise)


		#------- Unconstrained representation---------
		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(),tfb.Identity()]

		target_accept_rate = 0.651


		# Setting up the step_size
		step_size = tf.Variable(0.01, name = 'step_size')

		if initial_state == None:
			beta = 1.2*tf.ones(self.dim_input)
			varm = 0.8
			loc = 0.0
		else:
			beta, varm, loc = initial_state

		beta_cur = tf.Variable(beta, name = 'beta_cur')
		varm_cur = tf.Variable(varm, name = 'varm_cur')
		loc_cur = tf.Variable(loc, name = 'loc_cur')

		current_state = [beta_cur, varm_cur,loc_cur]


		# Initializing the sampler
		sampler = TransformedTransitionKernel(
						inner_kernel= HamiltonianMonteCarlo(
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
		num_accepted  = 0
		with tf.Session() as sess:
			sess.run(init)
			print('Warmup: ')
			for t in range(num_warmup_iters):
				_, is_accepted_val = sess.run([warmup, kernel_results.inner_results.is_accepted])
				num_accepted +=  is_accepted_val
				if (t % display_rate == 0) or ( t == num_warmup_iters -1):
					print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, num_accepted / (t + 1)))
			[step_size_, beta_next_, varm_next_, loc_next_] = sess.run([step_size, beta_next, varm_next, loc_next])
		next_state = [beta_next_, varm_next_, loc_next_]

		return step_size_, next_state



	def mcmc(self, mcmc_samples, num_burnin_steps, step_size, num_leapfrog_steps = 3, initial_state = None):
		# Function used to perform the sampling for the posterior distributions of the hyperparameters

		# Inputs:
		#	mcmc_samples := number of samples to collect for the hyperparameters
		#	num_burnin_steps := number of samples to discard
		# 	step_size := step_size for the HMC sampler
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# initial_state := list ([beta, varm, loc]) of tensors providing the initial state for the HMC sampler
		# Outputs:
		#	hyperpar_samples= list [loc_samples_, varm_samples, beta_samples_] of samples for the posterior
		#									distribution of the hyperparameters
		#   acceptance_rate_ := acceptance rate of the sampling

		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, self.noise)


		#------- Unconstrained representation---------
		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Identity()]

		if initial_state == None:
			beta = 1.2*tf.ones(self.dim_input, tf.float32)
			varm = 0.8
			loc = 0.0
			initial_state = [beta, varm, loc]

		#----Setting up the mcmc sampler
		[
			beta_samples,
			varm_samples,
			loc_samples
		], kernel_results = sample_chain(num_results= mcmc_samples, num_burnin_steps= num_burnin_steps,
																current_state=initial_state,
																kernel= TransformedTransitionKernel(
																	inner_kernel= HamiltonianMonteCarlo(
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
		hyperpar_samples = [loc_samples_, varm_samples_, beta_samples_]
		return hyperpar_samples, acceptance_rate_

	def EM_with_MCMC(self, num_warmup_iters, em_iters, mcmc_samples, num_leapfrog_steps, initial_state = None ,learning_rate = 0.01, display_rate = 200):
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
		#	hyperpar_samples= list [loc_samples_, varm_samples, beta_samples_] of samples for the posterior
		#									distribution of the hyperparameters
		#   loss_history := array containing values of the loss fucntion of the M-step
		#   noise_history := array containing values of the noise variance computed in the M-step

		# defining unconstrained version for the noise level
		unc_noise_init = tf.log(tf.exp(self.noise)- 1)
		unc_noise = tf.Variable(unc_noise_init , name = 'unc_noise')

		# Setting up the step_size and targeted acceptance rate for the MCMC part
		step_size = tf.Variable(0.01, name = 'step_size')
		target_accept_rate = 0.651

		if initial_state == None:
			beta = 1.2*tf.ones(self.dim_input, tf.float32)
			varm = 0.8
			loc = 0.0
		else:
			beta, varm, loc = initial_state

		beta_cur = tf.Variable(beta, name = 'beta_cur', trainable = False)
		varm_cur = tf.Variable(varm, name = 'varm_cur', trainable = False)
		loc_cur = tf.Variable(loc, name = 'loc_cur', trainable = False)

		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(),tfb.Identity()]


		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, tf.nn.softplus(unc_noise))

		current_state = [beta_cur, varm_cur,loc_cur]

		# Initializing a sampler for warmup:
		sampler = TransformedTransitionKernel(
						inner_kernel= HamiltonianMonteCarlo(
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


		# Updating the state of the hyperparameters
		beta_update1 = beta_cur.assign(beta_next)
		varm_update1 = varm_cur.assign(varm_next)
		loc_update1 = loc_cur.assign(loc_next)

		warmup_update = tf.group([beta_update1, varm_update1,loc_update1, step_size_update])
		step_size_update2 = step_size.assign(0.95*step_size)
		simple_update = tf.group([beta_update1, varm_update1,loc_update1])

		# Set up E-step with MCMC
		[
			beta_probs,
			varm_probs,
			loc_probs
		], em_kernel_results = sample_chain(num_results= 10, num_burnin_steps= 10,
																current_state=current_state,
																kernel= TransformedTransitionKernel(
																	inner_kernel= HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= 0.95*step_size,
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

		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)

		# Initial  warm-up stage
		print('First warm-up phase.')
		num_accepted = 0
		for t in range(num_warmup_iters):
			_, is_accepted_val = sess.run([warmup_update, kernel_results.inner_results.is_accepted])
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

		# Second warmup phase
		print('Second warm-up phase.')
		num_accepted = 0
		for t in range(num_warmup_iters):
			_, is_accepted_val = sess.run([warmup_update, kernel_results.inner_results.is_accepted])
			num_accepted +=  is_accepted_val
			if (t % display_rate == 0) or ( t == num_warmup_iters -1):
		  		print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, num_accepted / (t + 1)))

		print('Collecting samples for the GP hyperparameters.')
		sess.run(step_size_update2)
		loc_samples = np.zeros(mcmc_samples)
		varm_samples = np.zeros(mcmc_samples)
		beta_samples = np.zeros((mcmc_samples, self.dim_input))
		num_accepted = 0
		for t in range(mcmc_samples):
			[
				_,
				is_accepted_val,
				loc_next_,
				varm_next_,
				beta_next_
			] = sess.run(
			[
				simple_update,
				kernel_results.inner_results.is_accepted,
				loc_next,
				varm_next,
				beta_next
			])
			loc_samples[t] = loc_next_
			varm_samples[t] = varm_next_
			beta_samples[t,:] = beta_next_
			num_accepted +=  is_accepted_val
			if (t % display_rate == 0) or ( t == mcmc_samples -1):
		  		print("Sampling Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, num_accepted / (t + 1)))

		self.noise = math.log(math.exp(unc_noise_) + 1)
		hyperpar_samples = [loc_samples, varm_samples, beta_samples]

		return hyperpar_samples, loss_history, noise_history


	def posteriormeanVariance(self, Xtest, hyperpars, fullCov = False):
		# Function generates posterior mean and variance for the Gaussian process, given values for the hyperparameters
		# Inputs:
		# 	Xtest := N x D tensorflow array of new inputs
		#	hyperpars := 1d array containing a set of values for the hyperparameters
		#			these values are stacked in the following order: loc, varm, beta
		# fullCov := boolean specifying if a full covariance matrix should be computed or not
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
		L = tf.cholesky(Cov_train)

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
		# 	hyperpar_samples := list (of the form [loc_samples, varm_samples, beta_samples]) of numpy arrays containing samples for the hyperparameters
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

	#---------------------------------------------------------------------------------------------
	#---- The next functions are needed for sensitivity analysis of the latent output (i.e no noise is added to the output)

	def full_PosteriormeanVariance(self, X, L, hyperpars):
		# This is needed for computing the main effecs and interactions
		# Inputs:
		#	X:= is a 2-dimensional array
		#	L:= Cholesky factor of the Covariance matrix of the training data
		# hyperpars := list of values for the kernel hyperparameters (of the form [beta, varm, loc])
		n_new = X.shape[0].value

		beta, varm, loc = hyperpars

		Cov_test = varm*tf.ones(n_new)

		Kx3 = self.kernel(self.Xtrain, X, beta)

		Cov_mixed = varm*Kx3

		Y = self.Ytrain[:, tf.newaxis] - loc


		mean, var = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, False)

		mean_and_var = tf.concat([mean, var], axis = 1)

		return mean_and_var


	def expected_PosteriormeanVariance(self, X, L, hyperpars):
		# This is needed for computing the main effecs and interactions
		# Inputs:
		#	X:= is a 3-dimensional array
		#	L:= Cholesky factor of the Covariance matrix of the training data
		# hyperpars := list of values for the kernel hyperparameters (of the form [beta, varm, loc])
		n_blocks = X.shape[0].value

		beta, varm, loc = hyperpars

		Xm, Xk1, Xk2 = tf.split(X,num_or_size_splits = 3, axis = 2)


		Kx2 = self.aug_kernel(Xk1, Xk2, beta, diag = True)
		Cov_test_expected = varm*tf.reduce_mean(Kx2, axis = 1)

		#------- covariance between test data and simulation training data
		Xtrain_tiled = tf.tile(self.Xtrain[tf.newaxis,:,:], [n_blocks,1,1])


		Kx3 = self.aug_kernel(Xtrain_tiled, Xm, beta)

		Cov_mixed_expected = varm*tf.reduce_mean(Kx3, axis = -1)
		Cov_mixed_expected = tf.transpose(Cov_mixed_expected)

		Y = self.Ytrain[:, tf.newaxis] - loc


		mean, var = posterior_Gaussian(L, Cov_mixed_expected, Cov_test_expected, Y, False)
		var = tf.maximum(var, 1e-30)


		mean_and_var = tf.concat([mean, var], axis = 1)

		return mean_and_var





	def expected_predict_posterior(self, Xnew, hyperpar_samples, devices_list):
		# function used to compute the mean and variance of main effect Gaussian processes
		# These are Gaussian processes of the form
		#          E[Y|X_i]
		# This means that we are keeping a set of variables fixed (in this case the
		# subset X_i) while averaging out over the rest of the variables. For simplicity,
		# the variables are assumed to have uniform distributions. The integrals involved
		# in the computation are approximated with Monte Carlo integration
		# Inputs:
		# Xnew := 4-dimensional numpy array containing the input samples
		# hyperpar_samples := list (of the form [loc_samples, varm_samples, beta_samples]) of numpy arrays containing samples for the hyperparameters
		# devices_list := list of GPU devices available for the computation. This helps with parallelizing the computation.

		loc_samples, varm_samples, beta_samples = hyperpar_samples
		beta = tf.convert_to_tensor(np.median(beta_samples, axis =0), tf.float32)
		varm = tf.convert_to_tensor(np.median(varm_samples,axis =0), tf.float32)
		loc = tf.convert_to_tensor(np.median(loc_samples, axis =0), tf.float32)
		hyperpars = [beta, varm, loc]

		# ------- generate covariance matrix for training data and computing the corresponding cholesky factor
		Kxx = self.kernel(self.Xtrain, self.Xtrain, beta)

		Cov_train = varm*Kxx + (self.noise+self.jitter_level)*tf.eye(self.n_train)
		L = tf.cholesky(Cov_train)


		f = lambda Xin: self.expected_PosteriormeanVariance(Xin, L, hyperpars)

		n_devices = len(devices_list)
		X_list = np.array_split(Xnew, n_devices, axis = 0)
		results = []
		for i in range(n_devices):
			d = devices_list[i]
			with tf.device(d):
				results.append(tf.map_fn(f, tf.convert_to_tensor(X_list[i], tf.float32), swap_memory = False))
		with tf.Session() as sess:
			results_ = sess.run([results[i] for i in range(n_devices)])
		results_= np.concatenate(results_, axis = 0)
		return results_




	def predict_posterior(self, Xnew, hyperpar_samples):
		# function used to compute the mean and variance of posterior Gaussian process
		# Inputs:
		# Xnew := 2-dimensional numpy array containing the input samples
		# hyperpar_samples := list (of the form [loc_samples, varm_samples, beta_samples]) of numpy arrays containing samples for the hyperparameters

		X = tf.convert_to_tensor(Xnew, tf.float32)

		loc_samples, varm_samples, beta_samples = hyperpar_samples
		beta = tf.convert_to_tensor(np.median(beta_samples, axis =0), tf.float32)
		varm = tf.convert_to_tensor(np.median(varm_samples,axis =0), tf.float32)
		loc = tf.convert_to_tensor(np.median(loc_samples, axis =0), tf.float32)
		hyperpars = [beta, varm, loc]

		# ------- generate covariance matrix for training data and computing the corresponding cholesky factor
		Kxx = self.kernel(self.Xtrain, self.Xtrain, beta)

		Cov_train = varm*Kxx + (self.noise+self.jitter_level)*tf.eye(self.n_train)
		L = tf.cholesky(Cov_train)

		results = self.full_PosteriormeanVariance(X, L, hyperpars)

		with tf.Session() as sess:
			results_ = sess.run(results)
		return results_
