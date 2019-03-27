from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import time
import functools
import math
from tensorflow_probability.python.mcmc import sample_chain, HamiltonianMonteCarlo, TransformedTransitionKernel
from tensorflow_probability.python import distributions  as tfd
from  tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.models.bayesgp.scripts.bgpkernels import *
from tensorflow_probability.python.models.bayesgp.scripts.bgputils import posterior_Gaussian, step_size_simple_update
import warnings

#------------------------------------------------------------------------------
# This script implements a version of the Kennedy O' Hagan model. This consists
# of modeling the output of a physical process in the form
#  y  = eta(x, theta) + delta(x)
# where eta(.,.) is a Gaussian process that models the output of a simulator of the
# physical process and delta() is another Gaussian process that models the
# discrepancy between the simulator Gaussian process and the actual process.
# The variables theta represent the "best values" for the calibration parameters.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#------------ References ------------------------------------
# - [Rasmussen] "Gaussian process for Machine Learning", C. E. Rasmussen and C. K. I. Williams, MIT press
# - [Higdon]   "Combining Field Data and Computer Simulation for Calibration and Prediction", D. Higdon,
# M. Kennedy, J. Cavendish, J. Cafeo and R.D. Ryne

#-------------------------------------------------------------------------------

#--------- Some notation --------------------#
# - beta  := the inverse lengthscale of the stationay kernels. It comes in
#            3 categories:
#         	-- betasx := the simulator Gaussian process part for the input variables
#         	-- betaspar :=  the simualtor Gaussian process part for the parameter (i.e calibration) variables
#         	-- betad  := the inadequacy  (i.e "error") Gaussian process part for the input variables
# loc    := the constant mean function of the simulator Gaussian process
# varsim := the variance for the simulator Gaussian process
# vard   := the variance for the discrepancy Gaussian process


class Calibration():
	def __init__(self, sim_inputs, sim_outputs, exp_inputs, exp_outputs, lower_bounds, upper_bounds, kernel_type = 'RBF', noise_level = 1e-3):
		# Inputs:
		# sim_inputs , sim_outputs := inputs and outputs for the simulation data.
		# 						Here the sim_inputs numpy array includes the regular input variables
		# 						as well as the parameter variables
		# exp_inputs , exp_outputs := inputs and outputs for the experiment
		# lower_bounds and upper_bounds := lists containing the bounds for the uniform distribution for
		# 				the calibration parameters. For now, we will assume that the prior distributions for
		# 				the calibration parameters are uniform distributions
		#kernel_type := string specifying the type of kernel to be used. Options are
		#                'RBF', 'Matern12', 'Matern32', 'Matern52'
		# noise_level := value for the noise variance of the output


		#------- Converting the data to tensors and storing -------------
		if len(exp_inputs.shape) == 1:
			self.dim_input = 1
		else:
			self.dim_input = exp_inputs.shape[1]
		self.dim_par = sim_inputs.shape[1] - self.dim_input
		Xsim = sim_inputs[:,:self.dim_input]
		Psim = sim_inputs[:,self.dim_input:]
		self.Psim = tf.convert_to_tensor(Psim, tf.float32)
		Ysim = sim_outputs
		self.Xsim = tf.convert_to_tensor(Xsim, tf.float32)
		self.Ysim = tf.convert_to_tensor(Ysim, tf.float32)
		self.n_sim = len(sim_inputs)

		Xexp = exp_inputs
		Yexp = exp_outputs
		self.Xexp = tf.convert_to_tensor(Xexp, tf.float32)
		self.Yexp = tf.convert_to_tensor(Yexp, tf.float32)
		self.n_exp = len(exp_inputs)   # the numper of experimental outputs

		Xaug =  np.concatenate([Xsim,Xexp], axis =0)
		self.Xaug = tf.convert_to_tensor(Xaug, tf.float32)
		Yaug = np.concatenate([Ysim, Yexp], axis = 0)
		self.Yaug = tf.convert_to_tensor(Yaug, tf.float32)


		self.n_total = len(sim_inputs) + len(exp_inputs) # the total number of outputs

		#------ Priors for the Gaussian process simulator model-----------
		# Priors on the inverse lengthscale for the Gaussian simulator model
		# We define separate inverse lengthscale vectors for the inputs and the parameters
		self.rv_betasx = tfd.Independent(tfd.Gamma(concentration = 1.0*tf.ones(self.dim_input), rate = 1.0*tf.ones(self.dim_input)),
                           reinterpreted_batch_ndims=1, name='rv_betasx')

		self.rv_betaspar= tfd.Independent(tfd.Gamma(concentration = 1.0*tf.ones(self.dim_par), rate = 1.0*tf.ones(self.dim_par)),
                             reinterpreted_batch_ndims=1, name='rv_betaspar')

		# prior on the mean
		self.rv_loc = tfd.Normal(loc = 0.0, scale = 0.5, name = 'rv_loc')

		# prior on the variance of the simulation model
		self.rv_varsim = tfd.Gamma(concentration = 1.0, rate = 1.0,  name = 'rv_varsim')

		#prior on the calibration parameters
		self.rv_par = tfd.Independent(tfd.Uniform(low = lower_bounds, high = upper_bounds),
                        reinterpreted_batch_ndims=1, name= 'rv_par')

		#-----  Priors for the Gaussian process discrepancy model

		# priors on the inverse lengthscale
		self.rv_betad = tfd.Independent(tfd.Gamma(concentration = 1.0*tf.ones(self.dim_input), rate = 1.0*tf.ones(self.dim_input)),
                           reinterpreted_batch_ndims=1, name='rv_betad')

		# prior on the variance of the Gaussian discrepancy model
		self.rv_vard =  tfd.Gamma(concentration = 1.0, rate = 1.0,  name = 'rv_vard')

		# prior on the noise variance
		self.rv_noise = tfd.LogNormal(loc = -6.9, scale = 1.0, name = 'rv_noise')

		#-- value for the noise
		self.noise=  noise_level

		self.jitter_level = 1e-6 # jitter level to deal with numerical instability with cholesky factorization

		self.kernel, self.aug_kernel = kernel_mapping[kernel_type]


		return


	def joint_log_prob(self, noise, betasx, betaspar, varsim, loc, par, betad, vard):
		# function for computing the joint_log_prob given values for the inverse lengthscales
		# the variances of the kernel, the mean of the gaussian process and values for the  calibration parameters

		#------ forming the kernels -----------------
		Kxx = self.kernel(self.Xaug, self.Xaug, betasx)

		par_tiled = tf.tile(par[tf.newaxis,:], [self.n_exp,1])
		Paug = tf.concat([self.Psim, par_tiled], axis = 0)
		Kpp = self.kernel(Paug, Paug, betaspar)
		Cov_sim = varsim*tf.multiply(Kxx, Kpp) + noise*tf.eye(self.n_total)

		Kxx_err = self.kernel(self.Xexp, self.Xexp, betad)

		Cov_err = vard*Kxx_err
		Cov_err = tf.pad(Cov_err, tf.constant([[self.n_sim,0],[self.n_sim,0]]), name = None, constant_values=0)

		#-------- Computing the cholesky factor ------------
		Cov = Cov_sim + Cov_err + self.jitter_level*tf.eye(self.n_total)
		L = tf.linalg.cholesky(Cov)

		#---- Multivariate normal random variable for the combined outputs -----
		mean = loc*tf.ones(self.n_total)
		rv_observations = tfd.MultivariateNormalTriL(loc = mean, scale_tril = L )

		#--- Collecting the log_probs from the different random variables
		sum_log_prob = (rv_observations.log_prob(self.Yaug)
					 + self.rv_betasx.log_prob(betasx)
					 + self.rv_betaspar.log_prob(betaspar)
					 +  self.rv_varsim.log_prob(varsim)
					 +  self.rv_loc.log_prob(loc)
					 + self.rv_par.log_prob(par)
					 + self.rv_betad.log_prob(betad)
					 + self.rv_vard.log_prob(vard))

		return sum_log_prob



	def warmup(self, num_warmup_iters,num_leapfrog_steps, initial_state = None, display_rate = 500):
		# function to generate an adaptive step size that will be needed for
		# HMC sampling

		# Inputs:
		# 	num_warmup_iters := number of sampling steps to perfom during the warm-up
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# 	initial_state := list [betasx, betaspar, varsim, loc, par, betad, vard] of tensors
		#					providing the initial state for the HMC sampler.
		# 	display_rate := rate at which information is printed
		# Outputs:
		# 	step_size_ := estimated value for the step size of the HMC sampler
		#	next_state := list [betasx_next_, betaspar_next_ ....] containing the last sample values obtained from the warm-up

		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, self.noise)


		#------- Unconstrained representation---------
		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Softplus(),
									tfb.Identity(), tfb.Identity(),
									 tfb.Softplus(), tfb.Softplus()]

		target_accept_rate = 0.651


		# Setting up the step_size
		step_size = tf.Variable(0.01, name = 'step_size')

		if initial_state == None:
			betasx = 1.5*tf.ones(self.dim_input, tf.float32)
			betaspar = 1.5*tf.ones(self.dim_par, tf.float32)
			varsim = 0.8
			vard = 0.8
			loc = 0.0
			par = tf.convert_to_tensor(np.zeros(self.dim_par), tf.float32)
			betad = 1.5*tf.ones(self.dim_input, tf.float32)
		else:
			betasx, betaspar, varsim, loc, par, betad, vard = initial_state

		betasx_cur = tf.Variable(betasx, name = 'betasx_cur')
		betaspar_cur = tf.Variable(betaspar, name = 'betaspar_cur')
		varsim_cur = tf.Variable(varsim, name = 'varsim_cur')
		loc_cur = tf.Variable(loc, name = 'loc_cur')
		par_cur = tf.Variable(par, name = 'par_cur')
		betad_cur = tf.Variable(betad, name = 'betad_cur')
		vard_cur = tf.Variable(vard, name = 'vard_cur')

		current_state = [betasx_cur, betaspar_cur, varsim_cur,
						loc_cur, par_cur, betad_cur, vard_cur]

		# Initializing the sampler
		sampler = TransformedTransitionKernel(
						inner_kernel=HamiltonianMonteCarlo(
								target_log_prob_fn=unnormalized_posterior_log_prob,
								step_size= step_size,
								num_leapfrog_steps=num_leapfrog_steps),
						bijector=unconstraining_bijectors)

		# One step of the sampler
		[
			betasx_next,
			betaspar_next,
			varsim_next,
			loc_next,
			par_next,
			betad_next,
			vard_next
		], kernel_results = sampler.one_step(current_state = current_state,
											previous_kernel_results=sampler.bootstrap_results(current_state))

		# updating the step size
		step_size_update = step_size_simple_update(step_size, kernel_results,
													target_rate = target_accept_rate,
													decrement_multiplier = 0.1,
													increment_multiplier = 0.1)

		# Updating the state
		betasx_update = betasx_cur.assign(betasx_next)
		betaspar_update = betaspar_cur.assign(betaspar_next)
		varsim_update = varsim_cur.assign(varsim_next)
		loc_update = loc_cur.assign(loc_next)
		par_update = par_cur.assign(par_next)
		betad_update = betad_cur.assign(betad_next)
		vard_update = vard_cur.assign(vard_next)


		warmup = tf.group([betasx_update, betaspar_update, varsim_update,
							loc_update, par_update, betad_update, vard_update,step_size_update])

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
			[
				step_size_,
				betasx_next_,
				betaspar_next_,
				varsim_next_,
				loc_next_,
				par_next_,
				betad_next_,
				vard_next_
			] = sess.run(
			[
				step_size,
				betasx_next,
				betaspar_next,
				varsim_next,
				loc_next,
				par_next,
				betad_next,
				vard_next
			])

		next_state = [betasx_next_, betaspar_next_, varsim_next_, loc_next_, par_next_, betad_next_, vard_next_]
		return step_size_, next_state

# Continue editing starting here

	def mcmc(self, mcmc_samples, num_burnin_steps, step_size, num_leapfrog_steps = 3, initial_state = None):
		# Function to perform the sampling for the posterior distributios of the hyperparameters
		# and the calibration parameters
		# Inputs:
		#	mcmc_samples := number of samples to collect for the hyperparameters
		#	num_burnin_steps := number of samples to discard
		# 	step_size := step_size for the HMC sampler
		# 	num_leapfrog_steps := number of leapfrog steps for the HMC sampler
		# 	initial_state := list [betasx, betaspar, varsim, loc, par, betad, vard] of tensors
		#					providing the initial state for the HMC sampler.
		# Outputs:
		#	par_samples := numpy array of samples for the calibration parameters
		#	hyperpar_samples= list of samples for the posterior
		#									distribution of the hyperparameters
		#   acceptance_rate_ := acceptance rate of the sampling

		if initial_state == None:
			betasx = 1.5*tf.ones(self.dim_input, tf.float32)
			betaspar = 1.5*tf.ones(self.dim_par, tf.float32)
			varsim = 0.8
			vard = 0.8
			loc = 0.0
			par = tf.convert_to_tensor(np.zeros(self.dim_par), tf.float32)
			betad = 1.5*tf.ones(self.dim_input, tf.float32)
			initial_state  = [betasx, betaspar, varsim, loc, par, betad, vard]

		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, self.noise)

		#------- Unconstrained representation---------
		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Softplus(), tfb.Identity(), tfb.Identity(), tfb.Softplus(), tfb.Softplus()]


		#----Setting up the mcmc sampler
		[
			betasx_samples,
			betaspar_samples,
			varsim_samples,
			loc_samples,
			par_samples,
			betad_samples,
			vard_samples,
		], kernel_results = sample_chain(num_results= mcmc_samples, num_burnin_steps= num_burnin_steps,
																current_state=initial_state,
																kernel=TransformedTransitionKernel(
																	inner_kernel=HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= step_size,
																				num_leapfrog_steps=num_leapfrog_steps),
																	bijector=unconstraining_bijectors))



		acceptance_rate = tf.reduce_mean(tf.to_float(kernel_results.inner_results.is_accepted))


		with tf.Session() as sess:
			[
				acceptance_rate_,
				par_samples_,
				loc_samples_,
				varsim_samples_,
				betaspar_samples_,
				betasx_samples_,
				betad_samples_,
				vard_samples_
			] = sess.run(
			[
				acceptance_rate,
				par_samples,
				loc_samples,
				varsim_samples,
				betaspar_samples,
				betasx_samples,
				betad_samples,
				vard_samples
			])
		hyperpar_samples = [loc_samples_, varsim_samples_, betaspar_samples_, betasx_samples_, betad_samples_, vard_samples_]
		print('acceptance_rate:', acceptance_rate_)
		return par_samples_, hyperpar_samples, acceptance_rate_


	def EM_with_MCMC(self, num_warmup_iters, em_iters, mcmc_samples, num_leapfrog_steps,  initial_state = None, learning_rate = 0.01, display_rate = 200):
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
		#	loc_samples_, varsimm_samples,... := samples for the posterior
		#									distribution of the hyperparameters
		#   acceptance_rate_ := acceptance rate of the sampling
		#   loss_history := array containing values of the loss fucntion of the M-step
		#   noise_history := array containing values of the noise variance computed in the M-step

		# defining unconstrained version for the noise level
		unc_noise = tf.Variable(tf.log(tf.exp(1e-3)- 1), name = 'unc_noise')

		# Setting up the step_size and targeted acceptance rate for the MCMC part
		step_size = tf.Variable(0.01, name = 'step_size')
		target_accept_rate = 0.651

		if initial_state == None:
			betasx = 1.5*tf.ones(self.dim_input, tf.float32)
			betaspar = 1.5*tf.ones(self.dim_par, tf.float32)
			varsim = 0.8
			vard = 0.8
			loc = 0.0
			par = tf.convert_to_tensor(np.zeros(self.dim_par), tf.float32)
			betad = 1.5*tf.ones(self.dim_input, tf.float32)
		else:
			betasx, betaspar, varsim, loc, par, betad, vard = initial_state

		betasx_cur = tf.Variable(betasx, name = 'betasx_cur', trainable = False)
		betaspar_cur = tf.Variable(betaspar, name = 'betaspar_cur', trainable = False)
		varsim_cur = tf.Variable(varsim, name = 'varsim_cur', trainable = False)
		loc_cur = tf.Variable(loc, name = 'loc_cur', trainable = False)
		par_cur = tf.Variable(par, name = 'par_cur', trainable = False)
		betad_cur = tf.Variable(betad, name = 'betad_cur', trainable = False)
		vard_cur = tf.Variable(vard, name = 'vard_cur', trainable = False)

		unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Softplus(), tfb.Identity(), tfb.Identity(), tfb.Softplus(), tfb.Softplus()]


		unnormalized_posterior_log_prob = functools.partial(self.joint_log_prob, tf.nn.softplus(unc_noise))

		current_state = [betasx_cur, betaspar_cur, varsim_cur, loc_cur, par_cur, betad_cur, vard_cur]

		# Initializing a sampler for warmup:
		sampler = TransformedTransitionKernel(
						inner_kernel=HamiltonianMonteCarlo(
								target_log_prob_fn=unnormalized_posterior_log_prob,
								step_size= step_size,
								num_leapfrog_steps=num_leapfrog_steps),
						bijector=unconstraining_bijectors)

		# One step of the sampler
		[
			betasx_next,
			betaspar_next,
			varsim_next,
			loc_next,
			par_next,
			betad_next,
			vard_next
		], kernel_results = sampler.one_step(current_state = current_state,
											previous_kernel_results=sampler.bootstrap_results(current_state))

		# updating the step size
		step_size_update = step_size_simple_update(step_size, kernel_results,
													target_rate = target_accept_rate,
													decrement_multiplier = 0.1,
													increment_multiplier = 0.1)


		# Updating the state of the hyperparameters
		betasx_update1 = betasx_cur.assign(betasx_next)
		betaspar_update1 = betaspar_cur.assign(betaspar_next)
		varsim_update1 = varsim_cur.assign(varsim_next)
		loc_update1 = loc_cur.assign(loc_next)
		par_update1 = par_cur.assign(par_next)
		betad_update1 = betad_cur.assign(betad_next)
		vard_update1 = vard_cur.assign(vard_next)

		warmup_update = tf.group([betasx_update1, betaspar_update1, varsim_update1, loc_update1, par_update1, betad_update1, vard_update1, step_size_update])
		step_size_update2 = step_size.assign(0.95*step_size)
		simple_update = tf.group([betasx_update1, betaspar_update1, varsim_update1, loc_update1, par_update1, betad_update1, vard_update1])

		# Set up E-step with MCMC
		[
			betasx_probs,
			betaspar_probs,
			varsim_probs,
			loc_probs,
			par_probs,
			betad_probs,
			vard_probs
		], em_kernel_results = sample_chain(num_results= 10, num_burnin_steps= 0,
																current_state= current_state,
																kernel=TransformedTransitionKernel(
																	inner_kernel=HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= step_size,
																				num_leapfrog_steps=num_leapfrog_steps),
																	bijector=unconstraining_bijectors))


		# Updating the state of the hyperparameters
		betasx_update2 = betasx_cur.assign(tf.reduce_mean(betasx_probs, axis = 0))
		betaspar_update2 = betaspar_cur.assign(tf.reduce_mean(betaspar_probs, axis = 0))
		varsim_update2 = varsim_cur.assign(tf.reduce_mean(varsim_probs, axis = 0))
		loc_update2 = loc_cur.assign(tf.reduce_mean(loc_probs, axis = 0))
		par_update2 = par_cur.assign(tf.reduce_mean(par_probs, axis = 0))
		betad_update2 = betad_cur.assign(tf.reduce_mean(betad_probs, axis = 0))
		vard_update2 = vard_cur.assign(tf.reduce_mean(vard_probs, axis = 0))

		expectation_update = tf.group([betasx_update2, betaspar_update2, varsim_update2, loc_update2, par_update2, betad_update2, vard_update2])
		#-- Set up M-step (updating noise variance)
		with tf.control_dependencies([expectation_update]):
			loss = -self.joint_log_prob(tf.nn.softplus(unc_noise),betasx_cur, betaspar_cur, varsim_cur, loc_cur, par_cur, betad_cur, vard_cur) -self.rv_noise.log_prob(tf.nn.softplus(unc_noise))

			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			minimization_update = optimizer.minimize(loss)

		# Collecting samples after estimating the noise variance
		[
			betasx_samples,
			betaspar_samples,
			varsim_samples,
			loc_samples,
			par_samples,
			betad_samples,
			vard_samples
		], sampling_kernel_results = sample_chain(num_results= mcmc_samples, num_burnin_steps= 0,
																current_state=current_state,
																kernel=TransformedTransitionKernel(
																	inner_kernel=HamiltonianMonteCarlo(
			    																target_log_prob_fn=unnormalized_posterior_log_prob,
																				step_size= 0.7*step_size,
																				num_leapfrog_steps=num_leapfrog_steps),
																	bijector=unconstraining_bijectors))


		acceptance_rate = tf.reduce_mean(tf.to_float(sampling_kernel_results.inner_results.is_accepted))

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

		# Second warm-up phase
		num_accepted = 0
		for t in range(num_warmup_iters):
			_, is_accepted_val = sess.run([warmup_update, kernel_results.inner_results.is_accepted])
			num_accepted +=  is_accepted_val
			if (t % display_rate == 0) or ( t == num_warmup_iters -1):
		  		print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, num_accepted / (t + 1)))

		step_size_ = sess.run(step_size)
		if step_size_  < 1e-4:
			warnings.warn("Estimated step size is low. (less than 1e-4)")

		print('Collecting samples for the GP hyperparameters and the calibration parameters.')
		sess.run(step_size_update2)
		par_samples = np.zeros((mcmc_samples, self.dim_par))
		loc_samples = np.zeros(mcmc_samples)
		varsim_samples = np.zeros(mcmc_samples)
		betaspar_samples = np.zeros((mcmc_samples, self.dim_par))
		betasx_samples = np.zeros((mcmc_samples, self.dim_input))
		betad_samples = np.zeros((mcmc_samples, self.dim_input))
		vard_samples = np.zeros(mcmc_samples)
		num_accepted = 0

		for t in range(mcmc_samples):
			[
				_,
				is_accepted_val,
				par_next_,
				loc_next_,
				varsim_next_,
				betaspar_next_,
				betasx_next_,
				betad_next_,
				vard_next_
			] = sess.run(
			[
				simple_update,
				kernel_results.inner_results.is_accepted,
				par_next,
				loc_next,
				varsim_next,
				betaspar_next,
				betasx_next,
				betad_next,
				vard_next
			])
			par_samples[t,:] = par_next_
			loc_samples[t] = loc_next_
			varsim_samples[t] = varsim_next_
			betaspar_samples[t,:] = betaspar_next_
			betasx_samples[t,:] = betasx_next_
			betad_samples[t,:] = betad_next_
			vard_samples[t] = vard_next_
			num_accepted +=  is_accepted_val
			if (t % display_rate == 0) or ( t == mcmc_samples -1):
				acceptance_rate = num_accepted / (t + 1)
				print("Sampling Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, acceptance_rate))
		hyperpar_samples = [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples]

		if acceptance_rate < 0.1:
			warnings.warn("Acceptance rate was low  (less than 0.1)")


		sess.close()
		self.noise = np.log(np.exp(unc_noise_) + 1)
		self.noise = self.noise.astype(np.float32)

		return par_samples, hyperpar_samples , loss_history, noise_history

	def posteriormeanVariance(self, Xtest,  hyperpars_and_par, fullCov = False):
		# generate posterior samples for the full process, given values for the parameters and hyperparameters
		# Inputs:
		# 	Xtest := N x D tensorflow array of new inputs
		#	hyperpars_and_par := 1d array containing a set of values for the hyperparameters
		#			these values are stacked in the following order: loc, varsim, betaspar, betasx, betad, vard, par
		# fullCov := boolean specifying if a full covariance matrix should be computed or not
		# Output
		#	mean_and_var := array where the first column is an N x 1 array representing the
		#					mean of the posterior Gaussian  distribution and the remaining columns correspond to
		#        			the (Co)variance which is an N x N array if
		#					fullCov = True or a N x 1 array if fullCov = False
		splitting_shape = [1,1,self.dim_par, self.dim_input, self.dim_input, 1,self.dim_par]

		loc, varsim, betaspar, betasx, betad, vard, par = tf.split(hyperpars_and_par, splitting_shape)

		# ------- generate covariance matrix for training data and computing the corresponding cholesky factor
		Kxx = self.kernel(self.Xaug, self.Xaug, betasx)

		par_tiled = tf.tile(par[tf.newaxis,:], [self.n_exp,1])
		Paug = tf.concat([self.Psim, par_tiled], axis = 0)
		Kpp = self.kernel(Paug, Paug, betaspar)
		Cov_sim = varsim*tf.multiply(Kxx, Kpp) +  self.noise*tf.eye(self.n_total)

		Kxx_err = self.kernel(self.Xexp, self.Xexp, betad)

		Cov_err = vard*Kxx_err

		Cov_err = tf.pad(Cov_err, tf.constant([[self.n_sim,0],[self.n_sim,0]]), name = None, constant_values=0)

		Cov_train = Cov_sim + Cov_err + self.jitter_level*tf.eye(self.n_total)
		L = tf.linalg.cholesky(Cov_train)

		#-------- generate covariance matrix for test data
		n_test = Xtest.shape[0].value
		Paug2 = tf.tile(par[tf.newaxis,:], [n_test,1])
		if fullCov:
			Kx2 = self.kernel(Xtest, Xtest, betasx)
			Cov_sim2 = varsim*Kx2 +  self.noise*tf.eye(n_test)
			Kx_err2 =  self.kernel(Xtest, Xtest, betad)
			Cov_err2 = vard*Kx_err2
			Cov_test = Cov_sim2 + Cov_err2 + self.jitter_level*tf.eye(n_test)

		else:
			Cov_test = (varsim + vard + self.noise + self.jitter_level )*tf.ones(n_test)


		#------- covariance between test data and training data
		Kx3 =self.kernel(self.Xaug, Xtest, betasx)

		Kp3 = self.kernel(Paug, Paug2, betaspar)
		Cov_sim3 = varsim*tf.multiply(Kx3, Kp3)


		Kx_err3 = self.kernel(self.Xexp, Xtest, betad)

		Cov_err3 =  vard*Kx_err3

		Cov_err3 = tf.pad(Cov_err3, tf.constant([[self.n_sim,0],[0,0]]), name = None, constant_values=0)


		Cov_mixed = Cov_sim3 + Cov_err3

		Y = self.Yaug[:, tf.newaxis] - loc


		mean_pos, var_pos = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, fullCov)

		mean_pos = mean_pos + loc

		mean_and_var = tf.concat([mean_pos, var_pos], axis = 1)

		return mean_and_var


	def simPosteriormeanVariance(self, Xtest, hyperpars_and_par, fullCov = False):
		# generate posterior samples for the simulation process, given values for the parameters and the hyperparameters
		# generate posterior samples for the full process, given values for the parameters and hyperparameters
		# Inputs:
		# 	Xtest := N x D tensorflow array of new inputs
		#	hyperpars_and_par := 1d array containing a set of values for the hyperparameters
		#			these values are stacked in the following order: loc, varsim, betaspar, betasx, betad, vard, par
		# fullCov := boolean specifying if a full covariance matrix should be computed or not
		# Output
		#	mean_and_var := array where the first column is an N x 1 array representing the
		#					mean of the posterior Gaussian  distribution and the remaining columns correspond to
		#        			the (Co)variance which is an N x N array if
		#					fullCov = True or a N x 1 array if fullCov = False
		splitting_shape = [1,1,self.dim_par, self.dim_input, self.dim_input, 1,self.dim_par]

		loc, varsim, betaspar, betasx,  betad, vard, par = tf.split(hyperpars_and_par, splitting_shape)


		# ------- generate covariance matrix for simulation training data and computing the corresponding cholesky factor

		Kxx = self.kernel(self.Xaug, self.Xaug, betasx)
		par_tiled = tf.tile(par[tf.newaxis,:], [self.n_exp,1])
		Paug = tf.concat([self.Psim, par_tiled], axis = 0)
		Kpp = self.kernel(Paug, Paug, betaspar)
		Cov_sim = varsim*tf.multiply(Kxx, Kpp) +  self.noise*tf.eye(self.n_total)

		Cov_train = Cov_sim + self.jitter_level*tf.eye(self.n_total)
		L = tf.linalg.cholesky(Cov_train)

		#-------- generate covariance matrix for test data
		n_test = Xtest.shape[0].value
		Paug2 = tf.tile(par[tf.newaxis,:], [n_test,1])
		if fullCov:
			Kx2 = self.kernel(Xtest, Xtest, betasx)
			Cov_sim2 = varsim*Kx2 + self.noise*tf.eye(n_test)
			Cov_test = Cov_sim2 +  self.jitter_level*tf.eye(n_test)

		else:
			Cov_test = (varsim + self.noise + self.jitter_level )*tf.ones(n_test)


		#------- covariance between test data and simulation training data
		Kx3 = self.kernel(self.Xaug, Xtest, betasx)

		Kp3 = self.kernel(Paug, Paug2, betaspar)
		Cov_sim3 = varsim*tf.multiply(Kx3, Kp3)
		Cov_mixed = Cov_sim3

		Y = self.Yaug[:, tf.newaxis] - loc

		mean_pos, var_pos = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, fullCov)

		mean_pos = mean_pos + loc

		mean_and_var = tf.concat([mean_pos, var_pos], axis = 1)

		return mean_and_var


	def errorPosteriormeanVariance(self, Xtest, hyperpars_and_par, fullCov = False):
		# generate posterior samples for the inadequacy Gaussian process, given values for the parameters and the hyperparameters
		# generate posterior samples for the full process, given values for the parameters and hyperparameters
		# Inputs:
		# 	Xtest := N x D tensorflow array of new inputs
		#	hyperpars_and_par := 1d array containing a set of values for the hyperparameters
		#			these values are stacked in the following order: loc, varsim, betaspar, betasx, betad, vard, par
		# fullCov := boolean specifying if a full covariance matrix should be computed or not
		# Output
		#	mean_and_var := array where the first column is an N x 1 array representing the
		#					mean of the posterior Gaussian  distribution and the remaining columns correspond to
		#        			the (Co)variance which is an N x N array if
		#					fullCov = True or a N x 1 array if fullCov = False

		splitting_shape = [1,1,self.dim_par, self.dim_input, self.dim_input, 1,self.dim_par]

		loc, varsim, betaspar, betasx, betad, vard, par = tf.split(hyperpars_and_par, splitting_shape)


		# ------- generate covariance matrix for training data and computing the corresponding cholesky factor
		Kxx = self.kernel(self.Xaug, self.Xaug, betasx)

		par_tiled = tf.tile(par[tf.newaxis,:], [self.n_exp,1])
		Paug = tf.concat([self.Psim, par_tiled], axis = 0)
		Kpp = self.kernel(Paug, Paug, betaspar)
		Cov_sim = varsim*tf.multiply(Kxx, Kpp) +  self.noise*tf.eye(self.n_total)

		Kxx_err = self.kernel(self.Xexp, self.Xexp, betad)

		Cov_err = vard*Kxx_err

		Cov_err = tf.pad(Cov_err, tf.constant([[self.n_sim,0],[self.n_sim,0]]), name = None, constant_values=0)

		Cov_train = Cov_sim + Cov_err + self.jitter_level*tf.eye(self.n_total)
		L = tf.linalg.cholesky(Cov_train)


		#-------- generate covariance matrix for test data
		n_test = Xtest.shape[0].value
		if fullCov:
			Kx2 = self.kernel(Xtest, Xtest, betad)
			Cov_test = vard*Kx2  +  self.jitter_level*tf.eye(n_test)

		else:
			Cov_test = (vard + self.jitter_level )*tf.ones(n_test)


		#------- covariance between test data and experimental training data
		Kx3 = self.kernel(self.Xexp, Xtest, betad)

		Cov_mixed = vard*Kx3

		Cov_mixed = tf.pad(Cov_mixed, tf.constant([[self.n_sim,0],[0,0]]), name = None, constant_values=0)

		Y = self.Yaug[:, tf.newaxis] - loc

		mean_pos, var_pos = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, fullCov)

		mean_and_var = tf.concat([mean_pos, var_pos], axis = 1)

		return mean_and_var


	def samples_withpar(self, Xtest, hyperpar_samples, par_samples, num_samples = 20, with_point_samples = False):
		# Computes approximate values of the full posterior mean and variance of the Gaussian process
		# by using samples of the posterior distribution of the hyperparameters and the calibration parameters
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

		fsim = lambda hyperpars_and_par_in: self.simPosteriormeanVariance(Xtest, hyperpars_and_par_in, False)
		ferr = lambda hyperpars_and_par_in: self.errorPosteriormeanVariance(Xtest, hyperpars_and_par_in, False)
		f = lambda hyperpars_and_par_in: self.posteriormeanVariance(Xtest, hyperpars_and_par_in, False)

		loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples = hyperpar_samples
		hyperpars_and_par = np.concatenate([loc_samples[:,None], varsim_samples[:,None],betaspar_samples, betasx_samples, betad_samples, vard_samples[:,None],par_samples], axis=1)

		hyperpars_and_par = tf.convert_to_tensor(hyperpars_and_par, tf.float32)

		mean_and_var = tf.map_fn(f,hyperpars_and_par)
		mean_and_var_sim = tf.map_fn(fsim,hyperpars_and_par)
		mean_and_var_err = tf.map_fn(ferr,hyperpars_and_par)

		mean_pos_samples = mean_and_var[:,:,0]
		var_pos_samples = mean_and_var[:,:,1]
		var_pos_samples = tf.maximum(var_pos_samples,1e-30)

		mean_posim_samples = mean_and_var_sim[:,:,0]
		var_posim_samples = mean_and_var_sim[:,:,1]
		var_posim_samples = tf.maximum(var_posim_samples,1e-30)

		mean_poserr_samples = mean_and_var_err[:,:,0]
		var_poserr_samples = mean_and_var_err[:,:,1]
		var_poserr_samples = tf.maximum(var_poserr_samples,1e-30)

		if with_point_samples:
			rv_norm  = tfd.Normal(loc = mean_pos_samples, scale = tf.sqrt(var_pos_samples))
			samples = rv_norm.sample(num_samples)

			rv_norm_sim  = tfd.Normal(loc = mean_posim_samples, scale = tf.sqrt(var_posim_samples))
			samplesim = rv_norm_sim.sample(num_samples)

			rv_norm_err  = tfd.Normal(loc = mean_poserr_samples, scale = tf.sqrt(var_poserr_samples))
			sampleserr = rv_norm_err.sample(num_samples)


			with tf.Session() as sess:
					[
						mean_pos_samples_,
						var_pos_samples_,
						samples_,
						mean_posim_samples_,
						var_posim_samples_,
						samplesim_,
						mean_poserr_samples_,
						var_poserr_samples_,
						sampleserr_
					] = sess.run(
					[
						mean_pos_samples,
						var_pos_samples,
						samples,
						mean_posim_samples,
						var_posim_samples,
						samplesim,
						mean_poserr_samples,
						var_poserr_samples,
						sampleserr
					])


			mean_pos_ = np.mean(mean_pos_samples_, axis =0)
			var_pos_ = np.mean(var_pos_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)
			samples_ = np.concatenate(samples_, axis = 0)

			mean_posim_ = np.mean(mean_posim_samples_, axis =0)
			var_posim_ = np.mean(var_posim_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)
			samplesim_ = np.concatenate(samplesim_, axis = 0)

			mean_poserr_ = np.mean(mean_poserr_samples_, axis =0)
			var_poserr_ = np.mean(var_poserr_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)
			sampleserr_ = np.concatenate(sampleserr_, axis = 0)


			return mean_pos_, var_pos_, samples_, mean_posim_, var_posim_, samplesim_, mean_poserr_, var_poserr_, sampleserr_

		else:
			with tf.Session() as sess:
					[
						mean_pos_samples_,
						var_pos_samples_,
						mean_posim_samples_,
						var_posim_samples_,
						mean_poserr_samples_,
						var_poserr_samples_
					] = sess.run(
					[
						mean_pos_samples,
						var_pos_samples,
						mean_posim_samples,
						var_posim_samples,
						mean_poserr_samples,
						var_poserr_samples
					])


			mean_pos_ = np.mean(mean_pos_samples_, axis =0)
			var_pos_ = np.mean(var_pos_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)

			mean_posim_ = np.mean(mean_posim_samples_, axis =0)
			var_posim_ = np.mean(var_posim_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)


			mean_poserr_ = np.mean(mean_poserr_samples_, axis =0)
			var_poserr_ = np.mean(var_poserr_samples_, axis =0) + np.var(mean_pos_samples_, axis = 0)

			return mean_pos_, var_pos_, mean_posim_, var_posim_,  mean_poserr_, var_poserr_


	#-------------------------------------------------------------------------------
	#-------------------------------------------------------------------------------
	#--------------------- For sensitivity analysis ------------------------------


	def full_simPosteriormeanVariance(self, XPin, L, hyperpars):
		# This is needed for computing the main effecs and interactions of the simulation Gaussian process
		# Inputs:
		#	XPin:= is a 2-dimensional array
		#	L:= Cholesky factor of the Covariance matrix of the training data
		# hyperpars := list of values for the kernel hyperparameters (of the form [betasx, betaspar, varsim, loc, betad, vard])
		n_new = XPin.shape[0].value


		betasx, betaspar, varsim, loc, betad, vard = hyperpars

		X = XPin[:,:self.dim_input]
		P = XPin[:,self.dim_input:]


		Kx2 = self.kernel(X, X, betasx)
		Kp2 = self.kernel(P, P, betaspar)
		Cov_test = (varsim + self.jitter_level)*tf.ones(n_new)

		Kx3 = self.kernel(self.Xsim, X, betasx)

		Kp3 = self.kernel(self.Psim, P, betaspar)
		Cov_mixed = varsim*tf.multiply(Kx3, Kp3)

		Y = self.Ysim[:, tf.newaxis] - loc


		mean, var = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, False)


		mean_and_var = tf.concat([mean, var], axis = 1)

		return mean_and_var


	def expected_simPosteriormeanVariance(self, XPin, L, hyperpars):
		# This is needed for computing the main effecs and interactions of the simulation Gaussian process
		# Inputs:
		#	XPin:= is a 3-dimensional array
		#	L:= Cholesky factor of the Covariance matrix of the training data
		# hyperpars := list of values for the kernel hyperparameters (of the form [betasx, betaspar, varsim, loc, betad, vard])
		n_blocks = XPin.shape[0].value

		betasx, betaspar, varsim, loc, betad, vard = hyperpars

		XPm, XPk1, XPk2 = tf.split(XPin,num_or_size_splits = 3, axis = 2)
		Xm = XPm[:,:,:self.dim_input]
		Pm = XPm[:,:,self.dim_input:]
		Xk1 = XPk1[:,:,:self.dim_input]
		Pk1 = XPk1[:,:,self.dim_input:]
		Xk2 = XPk2[:,:,:self.dim_input]
		Pk2 = XPk2[:,:,self.dim_input:]


		Kx2 = self.aug_kernel(Xk1, Xk2, betasx, diag = True)
		Kp2 = self.aug_kernel(Pk1, Pk2, betaspar, diag = True)
		Cov_test_expected = varsim*tf.reduce_mean(tf.multiply(Kx2,Kp2), axis = 1)

		#------- covariance between test data and simulation training data
		Xsim_tiled = tf.tile(self.Xsim[tf.newaxis,:,:], [n_blocks,1,1])
		Psim_tiled = tf.tile(self.Psim[tf.newaxis,:,:], [n_blocks,1,1])

		Kx3 = self.aug_kernel(Xsim_tiled, Xm, betasx, diag = False)

		Kp3 = self.aug_kernel(Psim_tiled, Pm, betaspar, )
		Cov_mixed_expected = varsim*tf.reduce_mean(tf.multiply(Kx3, Kp3), axis = -1)
		Cov_mixed_expected = tf.transpose(Cov_mixed_expected)

		Y = self.Ysim[:, tf.newaxis] - loc


		mean, var = posterior_Gaussian(L, Cov_mixed_expected, Cov_test_expected, Y, False)
		var = tf.maximum(var, 1e-30)


		mean_and_var = tf.concat([mean, var], axis = 1)

		return mean_and_var


	def full_errorPosteriormeanVariance(self, Xin, L, hyperpars):
		# This is needed for computing the main effecs and interactions of the discrepancy Gaussian process
		# Inputs:
		#	Xin:= is a 2-dimensional array
		#	L:= Cholesky factor of the Covariance matrix of the training data
		# hyperpars := list of values for the kernel hyperparameters (of the form [betasx, betaspar, varsim, loc, betad, vard])
		n_new = Xin.shape[0].value

		betasx, betaspar, varsim, loc, betad, vard = hyperpars

		Cov_test = (vard + self.jitter_level)*tf.ones(n_new)

		#------- covariance between test data and experimental training data
		Kx3 = self.kernel(self.Xexp, Xin, betad)
		Cov_mixed = vard*Kx3

		Cov_mixed = tf.pad(Cov_mixed, tf.constant([[self.n_sim,0],[0,0]]), name = None, constant_values=0)

		Y = self.Yaug[:, tf.newaxis] - loc

		mean, var = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, False)

		mean_and_var = tf.concat([mean, var], axis = 1)

		return mean_and_var





	def expected_errorPosteriormeanVariance(self, Xin, L, hyperpars):
		# This is needed for computing the main effecs and interactions of the discrepancy Gaussian process
		# Inputs:
		#	Xin:= is a 3-dimensional array
		#	L:= Cholesky factor of the Covariance matrix of the training data
		# hyperpars := list of values for the kernel hyperparameters (of the form [betasx, betaspar, varsim, loc, betad, vard])

		n_new = Xin.shape[1].value
		n_blocks = Xin.shape[0].value

		betasx, betaspar, varsim, loc, betad, vard = hyperpars

		Xm, Xk1, Xk2 = tf.split(Xin,num_or_size_splits = 3, axis = 2)

		#-------- generate covariance matrix for test data
		Kx2 = self.aug_kernel(Xk1, Xk2, betad, diag = True)
		Cov_test_expected = vard*tf.reduce_mean(Kx2, axis = 1)

		#------- covariance between test data and experimental training data

		Xexp_tiled = tf.tile(self.Xexp[tf.newaxis,:,:], [n_blocks,1,1])
		Kx3 = self.aug_kernel(Xexp_tiled, Xm, betad)

		Cov_mixed_expected = vard*tf.reduce_mean(Kx3, axis = -1)
		Cov_mixed_expected = tf.transpose(Cov_mixed_expected)

		Cov_mixed_expected = tf.pad(Cov_mixed_expected, tf.constant([[self.n_sim,0],[0,0]]), name = None, constant_values=0)

		Y = self.Yaug[:, tf.newaxis] - loc

		mean, var = posterior_Gaussian(L, Cov_mixed_expected, Cov_test_expected, Y, False)
		var = tf.maximum(var, 1e-30)

		mean_and_var = tf.concat([mean, var], axis = 1)

		return mean_and_var




	def expected_predict_posterior(self, Vnew, hyperpar_samples, par_samples, devices_list,  type):
		# function used to compute the mean and variance of main effect for simulator Gaussian process
		# or discrepancy Gaussian process
		# These are Gaussian processes of the form
		#          E[Y|v_i]
		# This means that we are keeping a set of variables fixed (in this case the
		# subset v_i) while averaging out over the rest of the variables. For simplicity,
		# the variables are assumed to have uniform distributions. The integrals involved
		# in the computation are approximated with Monte Carlo integration
		# Inputs:
		# 	Vnew := 4-dimensional numpy array containing the input samples. In the case of the
		#     	simulator model, it consists of samples for the input variables and the calibration parameters.
		#		In the case of the discrepancy model, it consists of only samples fro the input variables
		# 	hyperpar_samples := list [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples] of numpy arrays containing samples for the hyperparameters
		# 	par_samples := numpy array of samples for the calibration parameters
		# 	devices_list := list of GPU devices available for the computation. This helps with parallelizing the computation.
		# 	type := string specifying if the computation is done for the simulator or the discrepancy Gaussian process

		loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples = hyperpar_samples
		betasx = tf.convert_to_tensor(np.median(betasx_samples, axis =0), tf.float32)
		betaspar = tf.convert_to_tensor(np.median(betaspar_samples, axis =0), tf.float32)
		varsim = tf.convert_to_tensor(np.median(varsim_samples,axis =0), tf.float32)
		loc = tf.convert_to_tensor(np.median(loc_samples, axis =0), tf.float32)
		betad = tf.convert_to_tensor(np.median(betad_samples,axis =0), tf.float32)
		vard = tf.convert_to_tensor(np.median(vard_samples,axis =0), tf.float32)
		hyperpars = [betasx, betaspar, varsim, loc, betad, vard]

		if type == 'simulator':
			# ------- generate covariance matrix for simulation training data and computing the corresponding cholesky factor
			Kxx = self.kernel(self.Xsim, self.Xsim, betasx)

			Kpp = self.kernel(self.Psim, self.Psim, betaspar)
			Cov_sim = varsim*tf.multiply(Kxx, Kpp) + self.noise*tf.eye(self.n_sim)

			Cov_train = Cov_sim + self.jitter_level*tf.eye(self.n_sim)
			L = tf.linalg.cholesky(Cov_train)

			f = lambda Vin: self.expected_simPosteriormeanVariance(Vin, L, hyperpars)
			n_devices = len(devices_list)
			V_list = np.array_split(Vnew, n_devices, axis = 0)
			results = []
			for i in range(n_devices):
				d = devices_list[i]
				with tf.device(d):
					results.append(tf.map_fn(f, tf.convert_to_tensor(V_list[i], tf.float32), swap_memory = False))
			with tf.Session() as sess:
				results_ = sess.run([results[i] for i in range(n_devices)])
			results_= np.concatenate(results_, axis = 0)
			return results_

		if type == 'discrepancy':

			par = tf.convert_to_tensor(np.median(par_samples, axis =0), tf.float32)

			Kxx = self.kernel(self.Xaug, self.Xaug, betasx)

			par_tiled = tf.tile(par[tf.newaxis,:], [self.n_exp,1])
			Paug = tf.concat([self.Psim, par_tiled], axis = 0)
			Kpp = self.kernel(Paug, Paug, betaspar)
			Cov_sim = varsim*tf.multiply(Kxx, Kpp) +  self.noise*tf.eye(self.n_total)

			Kxx_err = self.kernel(self.Xexp, self.Xexp, betad)

			Cov_err = vard*Kxx_err

			Cov_err = tf.pad(Cov_err, tf.constant([[self.n_sim,0],[self.n_sim,0]]), name = None, constant_values=0)

			Cov_train = Cov_sim + Cov_err + self.jitter_level*tf.eye(self.n_total)
			L = tf.linalg.cholesky(Cov_train)
			f = lambda Vin: self.expected_errorPosteriormeanVariance(Vin, L, hyperpars)
			n_devices = len(devices_list)
			V_list = np.array_split(Vnew, n_devices, axis = 0)
			results = []
			for i in range(n_devices):
				d = devices_list[i]
				with tf.device(d):
					results.append(tf.map_fn(f, tf.convert_to_tensor(V_list[i], tf.float32), swap_memory = False))
			with tf.Session() as sess:
				results_ = sess.run([results[i] for i in range(n_devices)])
			results_= np.concatenate(results_, axis = 0)
			return results_




	def predict_posterior(self, Vnew, hyperpar_samples, par_samples, type):
		# function used to compute the mean and variance of posterior Gaussian process
		# Inputs:
		# 	Vnew := 2-dimensional numpy array containing the input samples. In the case of the
		#     	simulator model, it consists of samples for the input variables and the calibration parameters.
		#		In the case of the discrepancy model, it consists of only samples fro the input variables
		# 	hyperpar_samples := list [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples] of numpy arrays containing samples for the hyperparameters
		# 	par_samples := numpy array of samples for the calibration parameters
		# 	type := string specifying if the computation is done for the simulator or the discrepancy Gaussian process
		V = tf.convert_to_tensor(Vnew, tf.float32)

		loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples = hyperpar_samples
		betasx = tf.convert_to_tensor(np.median(betasx_samples, axis =0), tf.float32)
		betaspar = tf.convert_to_tensor(np.median(betaspar_samples, axis =0), tf.float32)
		varsim = tf.convert_to_tensor(np.median(varsim_samples,axis =0), tf.float32)
		loc = tf.convert_to_tensor(np.median(loc_samples, axis =0), tf.float32)
		betad = tf.convert_to_tensor(np.median(betad_samples,axis =0), tf.float32)
		vard = tf.convert_to_tensor(np.median(vard_samples,axis =0), tf.float32)
		hyperpars = [betasx, betaspar, varsim, loc, betad, vard]

		if type == 'simulator':
			# ------- generate covariance matrix for training data and computing the corresponding cholesky factor
			Kxx = self.kernel(self.Xsim, self.Xsim, betasx)

			Kpp = self.kernel(self.Psim, self.Psim, betaspar)
			Cov_sim = varsim*tf.multiply(Kxx, Kpp) + self.noise*tf.eye(self.n_sim)

			Cov_train = Cov_sim + self.jitter_level*tf.eye(self.n_sim)
			Cov_temp = tf.cast(Cov_train, tf.float64)
			L_temp = tf.linalg.cholesky(Cov_temp)
			L = tf.cast(L_temp, tf.float32)
			results = self.full_simPosteriormeanVariance(V, L, hyperpars)

			with tf.Session() as sess:
				results_ = sess.run(results)
			return results_

		if type == 'discrepancy':
			Kxx = self.kernel(self.Xaug, self.Xaug, betasx)
			par = tf.convert_to_tensor(np.median(par_samples, axis =0), tf.float32)

			par_tiled = tf.tile(par[tf.newaxis,:], [self.n_exp,1])
			Paug = tf.concat([self.Psim, par_tiled], axis = 0)
			Kpp = self.kernel(Paug, Paug, betaspar)
			Cov_sim = varsim*tf.multiply(Kxx, Kpp) +  self.noise*tf.eye(self.n_total)

			Kxx_err = self.kernel(self.Xexp, self.Xexp, betad)

			Cov_err = vard*Kxx_err

			Cov_err = tf.pad(Cov_err, tf.constant([[self.n_sim,0],[self.n_sim,0]]), name = None, constant_values=0)

			Cov_train = Cov_sim + Cov_err + self.jitter_level*tf.eye(self.n_total)
			Cov_temp = tf.cast(Cov_train, tf.float64)
			L_temp = tf.linalg.cholesky(Cov_temp)
			L = tf.cast(L_temp, tf.float32)
			results = self.full_errorPosteriormeanVariance(V, L, hyperpars)

			with tf.Session() as sess:
				results_ = sess.run(results)
			return results_
