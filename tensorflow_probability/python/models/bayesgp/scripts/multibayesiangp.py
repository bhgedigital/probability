from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import functools
import math
from tensorflow_probability.python.mcmc import sample_chain, HamiltonianMonteCarlo, TransformedTransitionKernel
from tensorflow_probability.python import distributions  as tfd
from  tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.models.bayesgp.scripts.multigpkernels import *
from tensorflow_probability.python.models.bayesgp.scripts.bgputils import posterior_Gaussian, mixing_Covariance, mixing_Covariance_diag, step_size_simple_update
import warnings
import traceback

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# This script implements a multi-output Gaussian process where the posterior distribution of
# the hyperparameters are obtained using MCMC sampling.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#------------ References ------------------------------------
# - [Rasmussen] "Gaussian process for Machine Learning", C. E. Rasmussen and C. K. I. Williams, MIT press
#-------------------------------------------------------------------------------

#--------- Some notation --------------------#
#  beta  := array containing the inverse lengthscales of the kernels.
#			The model uses an ARD (Automatic Relevance Determination) kernel. There
#			is one inverse lengthscale for each input variable.
# loc    := array containing the constant mean functions
# varm := array containing the variance of the latent Gaussian processes


class MultiBayesianGP():

    def __init__(self, inputs, outputs, Wmix_init, noise_level, kernel_type = 'RBF', hyp_priors = {}):
    	# inputs := N x D array of inputs
    	# outputs := N x M array of outputs
    	# Wmix   := M x Q array of initial mixing coefficients
    	# noise_level := array of length M containing the variance of the noise
        # N := number of training points
        # D := dimension of the input space
        # Q := number of independent latent Gaussian processes
        # M := number of outputs

    	#------- Converting the data to tensors and storing -------------
        if len(inputs.shape) == 1:
        	self.dim_input = 1
        	self.Xtrain = tf.convert_to_tensor(inputs[:,None], dtype = tf.float32)
        else:
        	self.dim_input = inputs.shape[1]
        	self.Xtrain = tf.convert_to_tensor(inputs, dtype = tf.float32)
        self.Ytrain = tf.convert_to_tensor(outputs, dtype = tf.float32)
        self.n_train, self.n_tasks = outputs.shape  # the number of outputs
        self.n_latent = Wmix_init.shape[1]

        #------ Priors for the Gaussian process model-----------
        # Priors on the inverse lengthscale for the Gaussian simulator model
        if 'beta' in hyp_priors:
            try:
                concentration = tf.convert_to_tensor(hyp_priors['beta']['concentration'],tf.float32)
                rate = tf.convert_to_tensor(hyp_priors['beta']['rate'],tf.float32)
            except Exception as e:
                traceback.print_exc()
                print('Could not retrieve prior distibution information for beta.')
            conc_shape = np.array(concentration.shape)
            rate_shape = np.array(rate.shape)
            correct_shape = np.array([self.n_latent, self.dim_input])
            invalid = not(np.array_equal(conc_shape, correct_shape)) or not(np.array_equal(rate_shape, correct_shape))
            if invalid:
                raise Exception('Incorrect numpy array shape for rate or concentration for beta hyperparameter.')
        else:
            concentration = tf.ones([self.n_latent, self.dim_input])
            rate = tf.ones([self.n_latent, self.dim_input])
        self.rv_beta = tfd.Independent(tfd.Gamma(concentration = concentration,
        				rate = rate),
                        reinterpreted_batch_ndims=2, name='rv_beta')

        # prior on the mean
        if 'loc' in hyp_priors:
            try:
                loc = tf.convert_to_tensor(hyp_priors['loc']['loc'],tf.float32)
                scale = tf.convert_to_tensor(hyp_priors['loc']['scale'],tf.float32)
            except Exception as e:
                traceback.print_exc()
                print('Could not retrieve prior distibution information for loc.')
            loc_shape = np.array(loc.shape)
            scale_shape = np.array(scale.shape)
            correct_shape = np.array([self.n_tasks])
            invalid = not(np.array_equal(loc_shape, correct_shape)) or not(np.array_equal(scale_shape, correct_shape))
            if invalid:
                raise Exception('Incorrect numpy array shape for loc or scale for loc hyperparameter.')
        else:
            loc = tf.zeros([self.n_tasks])
            scale = tf.ones([self.n_tasks])
        self.rv_loc = tfd.Independent(tfd.Normal(loc = loc, scale = scale),
                                    reinterpreted_batch_ndims=1, name = 'rv_loc')

        # prior on the array of variances
        if 'varm' in hyp_priors:
            try:
                concentration = tf.convert_to_tensor(hyp_priors['varm']['concentration'],tf.float32)
                rate = tf.convert_to_tensor(hyp_priors['varm']['rate'],tf.float32)
            except Exception as e:
                traceback.print_exc()
                print('Could not retrieve prior distibution information for varm.')
            conc_shape = np.array(concentration.shape)
            rate_shape = np.array(rate.shape)
            correct_shape = np.array([self.n_tasks, self.n_latent])
            invalid = not(np.array_equal(conc_shape, correct_shape)) or not(np.array_equal(rate_shape, correct_shape))
            if invalid:
                raise Exception('Incorrect numpy array shape for rate or concentration for varm hyperparameter.')
        else:
            concentration = tf.ones([self.n_tasks, self.n_latent])
            rate = tf.ones([self.n_tasks, self.n_latent])
        self.rv_varm = tfd.Independent(tfd.Gamma(concentration = concentration,
        				rate = rate),
                        reinterpreted_batch_ndims=2, name='rv_varm')

        # prior on the common variance
        if 'varc' in hyp_priors:
            try:
                concentration = hyp_priors['varc']['concentration']
                rate = hyp_priors['varc']['rate']
            except Exception as e:
                traceback.print_exc()
                print('Could not retrieve prior distibution information for varc.')
            invalid = not(type(concentration) == float) or not(type(rate) == float)
            if invalid:
                raise Exception('Incorrect type for rate or concentration for varc hyperparameter. Values must be of type float.')
        else:
            concentration = 2.0
            rate = 2.0
        self.rv_varc = tfd.Gamma(concentration = concentration, rate = rate, name = 'rv_varc')


        self.Wmix = tf.convert_to_tensor(Wmix_init, dtype = tf.float32)

        #prior on the mixing matrix
        self.rv_Wmix = tfd.Independent(tfd.Normal(loc= self.Wmix,
        				scale = (0.5/self.n_latent)*tf.ones([self.n_tasks, self.n_latent])),
        				reinterpreted_batch_ndims=2, name= 'rv_Wmix')

        #-- value for the noise variance
        self.noise = tf.convert_to_tensor(noise_level, tf.float32)

        self.rv_noise = tfd.Independent(tfd.LogNormal(loc = -6.9*tf.ones(self.n_tasks), scale = 1.5*tf.ones(self.n_tasks)),
                        reinterpreted_batch_ndims=1, name = 'rv_noise')

        self.jitter_level = 1e-6 # jitter level to deal with numerical instability with cholesky factorization

        self.kernel, self.expected_kernel = kernel_mapping[kernel_type]

        return


    def joint_log_prob(self, noise, Wmix, beta, varm, loc, varc):
        # function for computing the joint_log_prob given values for the inverse lengthscales
        # the variance of the kernel and  the mean of the simulator Gaussian process

        #------ forming the kernels -----------------
        Kxx = self.kernel(self.Xtrain, self.Xtrain, beta)    # shape Q x N x N

        Cov_train = mixing_Covariance(Kxx, Wmix, varm, varc) # with shape M x N x M x N

        size = self.n_tasks*self.n_train

        noise_matrix = tf.tile(self.noise[:,tf.newaxis],[1, self.n_train])
        noise_matrix = tf.linalg.diag(tf.reshape(noise_matrix,[-1]))

        Cov_train = tf.reshape(Cov_train, [size, size]) + noise_matrix + self.jitter_level*tf.eye(size)

        #-------- Computing the cholesky factor ------------
        L = tf.linalg.cholesky(Cov_train)

        #---- Multivariate normal random variable for the combined outputs -----
        mean = tf.tile(loc[:, tf.newaxis], [1, self.n_train])
        mean = tf.reshape(mean, [size])
        rv_observations = tfd.MultivariateNormalTriL(loc = mean, scale_tril = L)

        Yreshaped = tf.reshape(tf.transpose(self.Ytrain),[size])

        #--- Collecting the log_probs from the different random variables
        sum_log_prob = (rv_observations.log_prob(Yreshaped)
        			 + self.rv_beta.log_prob(beta)
        			 +  self.rv_varm.log_prob(varm)
        			 +  self.rv_loc.log_prob(loc)
                     + self.rv_varc.log_prob(varc))

        return sum_log_prob

    def warmup(self, initial_state = None, num_warmup_iters = 1000, num_leapfrog_steps = 3, display_rate = 500):
        # function to generate an adaptive step size that will be needed for
        # HMC sampling
        noise = self.noise
        Wmix = self.Wmix

        if initial_state == None:
            beta_init = 1.2*tf.ones([self.n_latent,self.dim_input], dtype = tf.float32)
            varm_init = 0.8*tf.ones([self.n_tasks,self.n_latent], dtype = tf.float32)
            loc_init = tf.zeros(self.n_tasks)
            varc_init = 1.0
        else:
            beta_init, varm_init, loc_init, varc_init = initial_state


        unnormalized_posterior_log_prob = lambda *args: self.joint_log_prob(noise, Wmix, *args)


        #------- Unconstrained representation---------
        unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(),tfb.Identity(), tfb.Softplus()]

        target_accept_rate = 0.651

        # Setting up the step_size
        step_size = tf.Variable(0.01, name = 'step_size')


        beta_cur = tf.Variable(beta_init, name = 'beta_cur')
        varm_cur = tf.Variable(varm_init, name = 'varm_cur')
        loc_cur = tf.Variable(loc_init, name = 'loc_cur')
        varc_cur = tf.Variable(varc_init, name = 'varc_cur')

        current_state = [beta_cur, varm_cur,loc_cur, varc_cur]


        # Initializing the sampler
        sampler = TransformedTransitionKernel(
        				inner_kernel=HamiltonianMonteCarlo(
        						target_log_prob_fn=unnormalized_posterior_log_prob,
        						step_size= step_size,
        						num_leapfrog_steps=num_leapfrog_steps),
        				bijector=unconstraining_bijectors)

        # One step of the sampler
        [
        	beta_next,
        	varm_next,
        	loc_next,
            varc_next
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
        varc_update = varc_cur.assign(varc_next)

        warmup_update = tf.group([beta_update, varm_update,loc_update, varc_update, step_size_update])

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            print('Warmup: ')
            num_accepted = 0
            for t in range(num_warmup_iters):
                _, is_accepted_val = sess.run([warmup_update, kernel_results.inner_results.is_accepted])
                num_accepted += is_accepted_val
                if (t % display_rate == 0) or ( t == num_warmup_iters -1):
                	print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(t, num_accepted / (t + 1)))
            [step_size_, beta_next_, varm_next_, loc_next_, varc_next_] = sess.run([step_size, beta_next, varm_next, loc_next, varc_next])
        next_state =  [beta_next_, varm_next_, loc_next_, varc_next_]

        return step_size_, next_state


    def mcmc(self, mcmc_samples, num_burnin_steps, step_size, initial_state = None, prev_kernel_results = None, num_leapfrog_steps = 3):
    	# Function to perform the sampling for the posterior distributions of the hyperparameters

        noise = self.noise
        Wmix = self.Wmix

        unnormalized_posterior_log_prob = lambda *args: self.joint_log_prob(noise, Wmix,*args)

        if initial_state == None:
            print('generating initial state')
            beta_init = 1.2*tf.ones([self.n_latent,self.dim_input], dtype = tf.float32)
            varm_init = 0.8*tf.ones([self.n_tasks,self.n_latent], dtype = tf.float32)
            loc_init = tf.zeros(self.n_tasks)
            varc_init = 1.0
            initial_state = [beta_init, varm_init, loc_init, varc_init]

        #------- Unconstrained representation---------
        unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(), tfb.Identity(), tfb.Softplus()]


        #----Setting up the mcmc sampler
        [
        	beta_probs,
        	varm_probs,
        	loc_probs,
            varc_probs
        ], kernel_results = sample_chain(num_results= mcmc_samples, num_burnin_steps= num_burnin_steps,
                                                                num_steps_between_results= 4,
        														current_state=initial_state,
                                                                previous_kernel_results = prev_kernel_results,
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
        		loc_probs_,
        		varm_probs_,
        		beta_probs_,
                varc_probs_
        	] = sess.run(
        	[
        		acceptance_rate,
        		loc_probs,
        		varm_probs,
        		beta_probs,
                varc_probs
        	])

        print('acceptance_rate:', acceptance_rate_)
        hyperpar_samples = [loc_probs_, varm_probs_, beta_probs_, varc_probs_]
        return hyperpar_samples, acceptance_rate_

    def EM_with_MCMC(self, num_warmup_iters, em_iters, mcmc_samples, num_leapfrog_steps, initial_state = None, learning_rate = 0.01, display_rate = 200):

        Wmix = tf.Variable(self.Wmix, name = 'Wmix_cur')
        unc_noise_init = tf.math.log(tf.exp(self.noise)- 1)
        unc_noise = tf.Variable(unc_noise_init , name = 'unc_noise')

        # Setting up the step_size and targeted acceptance rate for the MCMC part
        step_size = tf.Variable(0.01, name = 'step_size')
        target_accept_rate = 0.651

        if initial_state == None:
            beta_init = 1.2*tf.ones([self.n_latent,self.dim_input], dtype = tf.float32)
            varm_init = 0.8*tf.ones([self.n_tasks,self.n_latent], dtype = tf.float32)
            loc_init = tf.zeros(self.n_tasks)
            varc_init = 1.0
        else:
            beta_init, varm_init, loc_init, varc_init = initial_state

        beta_cur = tf.Variable(beta_init, name = 'beta_cur', trainable = False)
        varm_cur = tf.Variable(varm_init, name = 'varm_cur', trainable = False)
        loc_cur = tf.Variable(loc_init, name = 'loc_cur', trainable = False)
        varc_cur = tf.Variable(varc_init, name = 'varc_cur', trainable = False)

        unconstraining_bijectors = [tfb.Softplus(), tfb.Softplus(),tfb.Identity(), tfb.Softplus()]

        unnormalized_posterior_log_prob = lambda *args: self.joint_log_prob(tf.nn.softplus(unc_noise), Wmix,*args)

        current_state = [beta_cur, varm_cur,loc_cur, varc_cur]

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
        	loc_next,
            varc_next
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
        varc_update1 = varc_cur.assign(varc_next)

        warmup_update = tf.group([beta_update1, varm_update1,loc_update1, varc_update1, step_size_update])
        step_size_update2 = step_size.assign(0.95*step_size)
        simple_update = tf.group([beta_update1, varm_update1,loc_update1, varc_update1])


        # Set up E-step with MCMC
        [
        	beta_probs,
        	varm_probs,
        	loc_probs,
            varc_probs
        ], em_kernel_results = sample_chain(num_results= 10, num_burnin_steps= 0,
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
        varc_update2 = varc_cur.assign(tf.reduce_mean(varc_probs, axis = 0))

        expectation_update = tf.group([beta_update2, varm_update2,loc_update2, varc_update2])

        #-- Set up M-step (updating noise variance)
        with tf.control_dependencies([expectation_update]):
        	loss = -self.joint_log_prob(tf.nn.softplus(unc_noise), Wmix, beta_cur, varm_cur, loc_cur, varc_cur) -self.rv_noise.log_prob(tf.nn.softplus(unc_noise)) \
                    -self.rv_Wmix.log_prob(Wmix)

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

        loss_history = np.zeros(em_iters)
        noise_history = np.zeros((em_iters, self.n_tasks))

        print('Estimating the noise variance: ')
        for t in range(em_iters):
            [
            _,
            _,
            unc_noise_,
            Wmix_,
            loss_
            ] = sess.run([
            expectation_update,
            minimization_update,
            unc_noise,
            Wmix,
            loss
            ])
            loss_history[t] = loss_
            noise_history[t,:] = np.log(np.exp(unc_noise_) + 1)
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

        step_size_ = sess.run(step_size)
        if step_size_  < 1e-4:
        	warnings.warn("Estimated step size is low. (less than 1e-4)")

        print('Collecting samples for the GP hyperparameters.')
        sess.run(step_size_update2)
        loc_samples = np.zeros((mcmc_samples, self.n_tasks))
        varm_samples = np.zeros((mcmc_samples, self.n_tasks, self.n_latent))
        beta_samples = np.zeros((mcmc_samples, self.n_latent,self.dim_input))
        varc_samples = np.zeros(mcmc_samples)
        num_accepted = 0
        total_runs = 4 * mcmc_samples
        for t in range(total_runs):
            [
            	_,
            	is_accepted_val,
            	loc_next_,
            	varm_next_,
            	beta_next_,
                varc_next_
            ] = sess.run(
            [
            	simple_update,
            	kernel_results.inner_results.is_accepted,
            	loc_next,
            	varm_next,
            	beta_next,
                varc_next
            ])
            if (t % 4 == 0) :
                idx = t//4
                loc_samples[idx,:] = loc_next_
                varm_samples[idx,:,:] = varm_next_
                beta_samples[idx,:,:] = beta_next_
                varc_samples[idx] = varc_next_
            num_accepted +=  is_accepted_val
            if (t % display_rate == 0) or ( t == total_runs -1):
            	acceptance_rate = num_accepted / (t + 1)
            	print("Sampling Iteration: {:>3} Acceptance Rate: {:.3f}".format(t,acceptance_rate))
        self.noise = np.log(np.exp(unc_noise_) + 1)
        self.noise = tf.convert_to_tensor(self.noise, tf.float32)
        self.Wmix = tf.convert_to_tensor(Wmix_, tf.float32)
        hyperpar_samples = [loc_samples, varm_samples, beta_samples, varc_samples]
        if acceptance_rate < 0.1:
        	warnings.warn("Acceptance rate was low  (less than 0.1)")

        sess.close()

        return hyperpar_samples, loss_history, noise_history


    def posteriormeanVariance(self, Xtest, hyperpars, fullCov = False):
        # generate posterior mean and variance for the Gaussian process, given values for the hyperparameters
        # Xtest := N x D tensorflow array of new inputs
        # output := mean of posterior distribution in the form of a N x Q array
        #         and (Co)variance of posterior in the form of a N x N X Q array if
        #		fullCov = True or a N x 1 array if fullCov = False
        beta, varm, loc, varc = hyperpars
        noise = self.noise
        Wmix = self.Wmix

        # ------- generate covariance matrix for training data and computing the corresponding cholesky factor
        Kxx = self.kernel(self.Xtrain, self.Xtrain, beta) # kernels for the latent Gaussian processes
        # Kxx has shape R x N x N

        Cov_train = mixing_Covariance(Kxx, Wmix, varm, varc) # with shape M x N x M x N

        size = self.n_tasks*self.n_train
        noise_matrix = tf.tile(self.noise[:,tf.newaxis],[1, self.n_train])
        noise_matrix = tf.linalg.diag(tf.reshape(noise_matrix,[-1]))


        Cov_train = tf.reshape(Cov_train, [size, size]) + noise_matrix + (self.jitter_level)*tf.eye(size)

        #-------- Computing the cholesky factor ------------
        L = tf.linalg.cholesky(Cov_train)

        #-------- generate covariance matrix for test data
        n_test = Xtest.shape[0].value
        size_test = self.n_tasks*n_test
        if fullCov:
        	Kx2 = self.kernel(Xtest, Xtest, beta)
        	Cov_test = mixing_Covariance(Kx2, Wmix, varm, varc)

        	Cov_test = tf.reshape(Cov_test, [size_test, size_test]) + (noise + self.jitter_level)*tf.eye(size_test)
        else:
            Cov_test = varc*tf.reduce_sum(tf.square(Wmix), axis =1) + tf.reduce_sum(varm, axis = 1)
            Cov_test = tf.tile(Cov_test[:, tf.newaxis], [1, n_test])
            Cov_test = tf.reshape(Cov_test,[size_test])

        #------- covariance between test data and training data
        Kx3 = self.kernel(self.Xtrain, Xtest, beta)

        Cov_mixed = mixing_Covariance(Kx3, Wmix, varm, varc) # with shape M x N x M x N_test

        Cov_mixed = tf.reshape(Cov_mixed, [size, size_test])


        mean_training = tf.tile(loc[:, tf.newaxis], [1, self.n_train])
        mean_training = tf.reshape(mean_training, [size,1])

        mean_test = tf.tile(loc[:, tf.newaxis], [1, n_test])
        mean_test = tf.reshape(mean_test, [size_test,1])

        Y = tf.transpose(self.Ytrain)
        Y = tf.reshape(Y, [size,1]) - mean_training

        mean_pos, var_pos = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, fullCov)

        mean_pos = mean_pos + mean_test

        return mean_pos, var_pos

    def samples(self, Xtest, hyperpar_samples, num_samples = 20, with_point_samples = False):
        # Sampling for the full model and the simulator model
        # hyperpar_samples is a list of numpy arrays contaning samples for the hyperparameters
        # Xtest is a 2-dimesional numpy array
        n_test = len(Xtest)
        if len(Xtest.shape) == 1:
        	Xtest = Xtest[:,None]
        Xtest = tf.convert_to_tensor(Xtest, tf.float32)
        size_test = n_test*self.n_tasks



        loc_samples, varm_samples, beta_samples, varc_samples = hyperpar_samples
        loc_samples = loc_samples.astype(np.float32)
        varm_samples = varm_samples.astype(np.float32)
        beta_samples = beta_samples.astype(np.float32)
        varc_samples = varc_samples.astype(np.float32)
        n_samples = len(loc_samples)
        i0 = tf.constant(0)
        collect_mean0 = tf.zeros([size_test,1])
        collect_variance0 = tf.zeros([size_test,1])

        if with_point_samples:
            collect_samples0 = tf.zeros([1,size_test,1])

            def condition(i, collect_mean, collect_variance, collect_samples):
                return i < n_samples

            def body(i,collect_mean, collect_variance, collect_samples):
                beta = tf.gather(beta_samples,i, axis = 0)
                varm = tf.gather(varm_samples,i, axis = 0)
                loc = tf.gather(loc_samples, i, axis = 0)
                varc = tf.gather(varc_samples, i, axis = 0)
                hyperpars = [beta,varm, loc, varc]
                mean_pos, var_pos = self.posteriormeanVariance(Xtest, hyperpars, fullCov = False)
                rv_norm  = tfd.Normal(loc = mean_pos, scale = tf.sqrt(var_pos))
                samples = rv_norm.sample(num_samples)
                out = [i+1, tf.concat([collect_mean, mean_pos], axis = 1),
                        tf.concat([collect_variance, var_pos], axis = 1), tf.concat([collect_samples,samples], axis =0)]
                return out

            results = tf.while_loop(condition, body, loop_vars = [i0, collect_mean0, collect_variance0, collect_samples0],
                        parallel_iterations = 20,
                        shape_invariants = [i0.get_shape(), tf.TensorShape([size_test, None]), tf.TensorShape([size_test, None]),tf.TensorShape([None,size_test,1])])

            with tf.Session() as sess:
                _, mean_pos_, var_pos_, samples_ = sess.run(results)


            mean_pos =np.mean(mean_pos_[:,1:],axis =1)
            var_pos = np.mean(var_pos_[:,1:], axis =1) + np.var(mean_pos_[:,1:],axis =1)

            # Reshaping
            mean_pos = np.reshape(mean_pos,(self.n_tasks,n_test))
            mean_pos = np.transpose(mean_pos)

            var_pos = np.reshape(var_pos,(self.n_tasks,n_test))
            var_pos = np.transpose(var_pos)

            samples = samples_[1:,:,0]
            n_total = len(samples)
            samples = np.reshape(samples,(n_total,self.n_tasks,n_test))
            samples = np.transpose(samples, (0,2,1))

            return mean_pos, var_pos, samples

        else:
            def condition(i, collect_mean, collect_variance):
                return i < n_samples

            def body(i,collect_mean, collect_variance):
                beta = tf.gather(beta_samples,i, axis = 0)
                varm = tf.gather(varm_samples,i, axis = 0)
                loc = tf.gather(loc_samples, i, axis = 0)
                varc = tf.gather(varc_samples, i, axis = 0)
                hyperpars = [beta,varm, loc, varc]
                mean_pos, var_pos = self.posteriormeanVariance(Xtest, hyperpars, fullCov = False)
                out = [i+1, tf.concat([collect_mean, mean_pos], axis = 1),
                        tf.concat([collect_variance, var_pos], axis = 1)]
                return out

            results = tf.while_loop(condition, body, loop_vars = [i0, collect_mean0, collect_variance0],
                        parallel_iterations = 20,
                        shape_invariants = [i0.get_shape(), tf.TensorShape([size_test, None]), tf.TensorShape([size_test, None])])

            with tf.Session() as sess:
                _, mean_pos_, var_pos_ = sess.run(results)

            mean_pos =np.mean(mean_pos_[:,1:],axis =1)
            var_pos = np.mean(var_pos_[:,1:], axis =1) + np.var(mean_pos_[:,1:],axis =1)

            # Reshaping
            mean_pos = np.reshape(mean_pos,(self.n_tasks,n_test))
            mean_pos = np.transpose(mean_pos)

            var_pos = np.reshape(var_pos,(self.n_tasks,n_test))
            var_pos = np.transpose(var_pos)

            return mean_pos, var_pos


    #-------------------------------------------------------------------------------------------
    #---- The next functions are used for sensitivity analysis
    def full_PosteriormeanVariance(self, X, L, hyperpars):
        n_new = X.shape[0].value
        size_new = self.n_tasks*n_new

        beta, varm, loc, varc = hyperpars

        Cov_test = varc*tf.reduce_sum(tf.square(self.Wmix), axis =1) + tf.reduce_sum(varm, axis = 1)
        Cov_test = tf.tile(Cov_test[:, tf.newaxis], [1, n_new])
        Cov_test = tf.reshape(Cov_test,[size_new])

        Kx3 = self.kernel(self.Xtrain, X, beta)

        size = self.n_tasks*self.n_train

        Cov_mixed = mixing_Covariance(Kx3, self.Wmix, varm, varc) # with shape M x N x M x N_test

        Cov_mixed = tf.reshape(Cov_mixed, [size, size_new])

        mean_training = tf.tile(loc[:, tf.newaxis], [1, self.n_train])
        mean_training = tf.reshape(mean_training, [size,1])

        mean_test = tf.tile(loc[:, tf.newaxis], [1, n_new])
        mean_test = tf.reshape(mean_test, [size_new,1])

        Y = tf.transpose(self.Ytrain)
        Y = tf.reshape(Y, [size,1]) - mean_training

        mean, var = posterior_Gaussian(L, Cov_mixed, Cov_test, Y, False)

        mean = mean + mean_test

        mean_and_var = tf.concat([mean, var], axis = 1)
        return mean_and_var


    def expected_PosteriormeanVariance(self, X, L, hyperpars, K_expected):
        size = self.n_tasks*self.n_train

        beta, varm, loc, varc = hyperpars

        Cov_test_expected = mixing_Covariance_diag(K_expected, self.Wmix, varm, varc) # shape M

        #----- covariance between test data and simulationn training data
        Kx3 = self.kernel(self.Xtrain, X, beta) # shape Q x Ntrain x Nb
        Kx3_expected = tf.reduce_mean(Kx3, axis = -1, keepdims = True) # shape Q x Ntrain x 1
        Cov_mixed_expected = mixing_Covariance(Kx3_expected, self.Wmix, varm, varc) # shape M x N_train x M x 1
        Cov_mixed_expected = tf.reshape(Cov_mixed_expected, [size, self.n_tasks])


        mean_training = tf.tile(loc[:, tf.newaxis], [1, self.n_train])
        mean_training = tf.reshape(mean_training, [size,1])

        Y = tf.transpose(self.Ytrain)
        Y = tf.reshape(Y, [size,1]) - mean_training

        mean, var = posterior_Gaussian(L, Cov_mixed_expected, Cov_test_expected, Y, False)
        var = tf.maximum(var,1e-40)

        mean = mean + loc[:,tf.newaxis]

        mean_and_var = tf.concat([mean, var], axis = 1)
        return mean_and_var

    def expected_predict_posterior(self, sampling_dict, hyperpar_samples):
		# function used to compute the mean and variance of main effect Gaussian processes
		# These are Gaussian processes of the form
		#          E[Y|X_i]
		# This means that we are keeping a set of variables fixed (in this case the
		# subset X_i) while averaging out over the rest of the variables. For simplicity,
		# the variables are assumed to have uniform distributions. The integrals involved
		# in the computation are approximated with Monte Carlo integration
		# Inputs:
		# sampling_dict := 4-dimensional numpy array containing the input samples
		# hyperpar_samples := list (of the form [loc_samples, varm_samples, beta_samples]) of numpy arrays containing samples for the hyperparameters

        loc_samples, varm_samples, beta_samples, varc_samples = hyperpar_samples
        beta_median = np.median(beta_samples, axis =0)
        varm_median = np.median(varm_samples,axis =0)
        loc_median = np.median(loc_samples, axis =0)
        varc_median = np.median(varc_samples, axis = 0)
        beta = tf.convert_to_tensor(beta_median, tf.float32)
        varm = tf.convert_to_tensor(varm_median, tf.float32)
        loc = tf.convert_to_tensor(loc_median, tf.float32)
        varc = tf.convert_to_tensor(varc_median, tf.float32)
        hyperpars = [beta, varm, loc, varc]

        Wmix = self.Wmix
        noise = self.noise

        # ------- generate covariance matrix for training data and computing the corresponding cholesky factor
        Kxx = self.kernel(self.Xtrain, self.Xtrain, beta) # kernels for the latent Gaussian processes
        # Kxx has shape R x N x N

        Cov_train = mixing_Covariance(Kxx, Wmix, varm, varc) # with shape M x N x M x N

        size = self.n_tasks*self.n_train
        noise_matrix = tf.tile(self.noise[:,tf.newaxis],[1, self.n_train])
        noise_matrix = tf.linalg.diag(tf.reshape(noise_matrix,[-1]))

        Cov_train = tf.reshape(Cov_train, [size, size]) + noise_matrix + (self.jitter_level)*tf.eye(size)

        #-------- Computing the cholesky factor ------------
        L = tf.linalg.cholesky(Cov_train)

        K_expected = tf.Variable(tf.ones(self.n_latent), name = 'K_expected')

        f = lambda Xin: self.expected_PosteriormeanVariance(Xin, L, hyperpars, K_expected)
        k = list(sampling_dict.keys())[0]
        grid_points, num_samples1, _ = sampling_dict[k]['X_sampling'].shape
        num_samples2, _ = sampling_dict[k]['diff_samples'].shape
        n_input = self.dim_input
        indices = set([ i for i in range(n_input)])
        beta_slice = tf.placeholder(tf.float32, shape = [self.n_latent,n_input], name = 'beta_slice')
        diff_samples = tf.placeholder(tf.float32, shape = [num_samples2, n_input], name = 'diff_samples')
        Xin  = tf.placeholder(tf.float32, shape = [grid_points, num_samples1, n_input], name =  'Xin')
        K_expected_new = self.expected_kernel(diff_samples, beta_slice)
        K_expected_update = K_expected.assign(K_expected_new)
        with tf.control_dependencies([K_expected_update]):
        	results = tf.map_fn(f,Xin)
        collect_results = {}
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
        	sess.run(init)
        	for i in sampling_dict.keys():
        		ids = sampling_dict[i]['fixed_indices_list']
        		ids_left = list(indices - set(ids))
        		ids_left.sort()
        		pad_size = len(ids)
        		diff_samples_batch = sampling_dict[i]['diff_samples']
        		diff_samples_batch = np.pad(diff_samples_batch, ((0,0), (0, pad_size)), mode = 'constant')
        		Xbatch = sampling_dict[i]['X_sampling']
        		beta_input = beta_median[:,ids_left]
        		beta_input = np.pad(beta_input, ((0,0),(0, pad_size)), mode = 'constant')

        		_, collect_results[i]= sess.run([K_expected_update,results], feed_dict={beta_slice: beta_input, diff_samples: diff_samples_batch, Xin: Xbatch})

        return collect_results







    def predict_posterior(self, Xnew, hyperpar_samples):
        # function used to compute the mean and variance of posterior Gaussian process
        # Inputs:
        # Xnew := 2-dimensional numpy array containing the input samples
        # hyperpar_samples := list (of the form [loc_samples, varm_samples, beta_samples]) of numpy arrays containing samples for the hyperparameters
        Wmix = self.Wmix
        noise = self.noise
        noise_matrix = tf.tile(self.noise[:,tf.newaxis],[1, self.n_train])
        noise_matrix = tf.linalg.diag(tf.reshape(noise_matrix,[-1]))

        X = tf.convert_to_tensor(Xnew, tf.float32)

        loc_samples, varm_samples, beta_samples, varc_samples = hyperpar_samples
        beta = tf.convert_to_tensor(np.median(beta_samples, axis =0), tf.float32)
        varm = tf.convert_to_tensor(np.median(varm_samples,axis =0), tf.float32)
        loc = tf.convert_to_tensor(np.median(loc_samples, axis =0), tf.float32)
        varc = tf.convert_to_tensor(np.median(varc_samples, axis =0), tf.float32)
        hyperpars = [beta, varm, loc, varc]

        # ------- generate covariance matrix for training data and computing the corresponding cholesky factor
        Kxx = self.kernel(self.Xtrain, self.Xtrain, beta) # kernels for the latent Gaussian processes
        # Kxx has shape R x N x N

        Cov_train = mixing_Covariance(Kxx, Wmix, varm, varc) # with shape M x N x M x N

        size = self.n_tasks*self.n_train

        Cov_train = tf.reshape(Cov_train, [size, size]) + noise_matrix + (self.jitter_level)*tf.eye(size)

        #-------- Computing the cholesky factor ------------
        L = tf.linalg.cholesky(Cov_train)

        results = self.full_PosteriormeanVariance(X, L, hyperpars)

        with tf.Session() as sess:
        	results_ = sess.run(results)
        return results_
