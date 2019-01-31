import tensorflow as tf
import numpy as np
import bayesiangp
import os
import matplotlib.pyplot as plt
import traceback
import warnings




# This class is used to run the Bayesian GP model. To initialize the model, the
# user needs to provide:
#   a) the input
#   b) the output
#   c) The type of kernel to use: 'RBF', 'Matern12', 'Matern32' or 'Matern52'
#   d) A value for the variance of the Gaussian noise of the output
# The data will be normalized internally. So the specified noise variance is
# expected to be between 0 and 1

class BGP_model():

    def __init__(self, Xtr, Ytr, kernel_type, noise_level = 1e-3):
        # Inputs:
        #   Xtr := N x D numpy array of inputs
        #  Ytr := N-dimensional numpy vector of outputs
        # noise_level := variance of the Gaussian noise for the normalized data

        # Checking that the Gaussian noise variance is between 0 and 1
        if (noise_level > 1) or (noise_level < 0):
            raise Exception('Invalid value for the noise_level: ' + str(noise_level) + '. It should be between 0 and 1.')

        # normalizing the input
        mean_x = np.mean(Xtr, axis = 0)
        std_x = np.std(Xtr, axis = 0, keepdims = True)
        Xnorm = (Xtr - mean_x)/std_x
        self.scaling_input = [mean_x, std_x]

        # Normalizing the outputs
        mean_y = np.mean(Ytr)
        std_y = np.std(Ytr)
        Ynorm = (Ytr - mean_y)/std_y
        self.scaling_output = [mean_y, std_y]

        self.n_inputs = Xtr.shape[1]

        self.model = bayesiangp.BayesianGP(Xnorm, Ynorm, kernel_type, noise_level) # initializing internal GP model
        self.hyperpar_samples = {} # dictionary to store the samples for the posterior distribution
                                    # of the hyperparameters

        self.step_size = None # step_size for the HMC sampler
        return

    def run_mcmc(self, mcmc_samples,num_leapfrog_steps, estimate_noise = False, warm_up = True, step_size = None):
        # Inputs:
        #   mcmc_samples := number of desired samples for the hyperparameters
        # num_leap_frog_steps = number of leap frog steps for the HMC sampler
        # estimated_noise := Boolean that indicates if the model should estimate a noise variance.
        #                    If the noise variance is
        # warm_up := Boolean that indicates if an adaptive step size is computed during warm up
        # step_size := step size to use for the HMC sampler if warm_up == False

        # initilal state for the hyperparameters
        beta = 1.5*tf.ones(self.n_inputs, tf.float32)
        varm = 0.8
        loc = 0.0
        initial_state = [beta, varm, loc]

        if estimate_noise == False:
            print('Noise variance is fixed.')
            if warm_up:
                # Execute a warmup phase to compute an adaptive step size
                burn_in  = mcmc_samples//2
                num_warmup_iters = burn_in
                try:
                    print('Excecuting the warmup.')
                    step_size, beta_next, varm_next, loc_next = self.model.warmup(initial_state,num_warmup_iters, num_leapfrog_steps)
                    initial_state = [beta_next, varm_next, loc_next]
                    print('Sampling in progress.')
                    loc_samples, varm_samples, beta_samples, acceptance_rate = self.model.mcmc(mcmc_samples, burn_in, initial_state, step_size, num_leapfrog_steps)
                    if acceptance_rate < 0.1:
                        warnings.warn("Acceptance rate was low  (less than 0.1)")
                    self.step_size = step_size
                except Exception as e:
                    traceback.print_exc()
                    print('Sampling failed. Increase the noise level or decrease the step size or the number of leap frog steps if necessary.')
                self.hyperpar_samples['kernel_variance'] = varm_samples
                self.hyperpar_samples['kernel_inverse_lengthscales'] = beta_samples
                self.hyperpar_samples['gp_constant_mean_function'] = loc_samples
                return
            else:
                if step_size == None:
                    raise Exception('You must specify a step size or set warm_up to be True for an adaptive step size')
                else:
                    try:
                        burn_in  = mcmc_samples
                        print('Sampling in progress.')
                        loc_samples, varm_samples, beta_samples, acceptance_rate = self.model.mcmc(mcmc_samples, burn_in, initial_state, step_size, num_leapfrog_steps)
                        if acceptance_rate < 0.1:
                            warnings.warn("Acceptance rate was low  (less than 0.1)")
                        self.step_size = step_size
                    except Exception as e:
                        traceback.print_exc()
                        print('Sampling failed. Increase the noise level or decrease the step size or the number of leap frog steps if necessary.')
                    self.hyperpar_samples['kernel_variance'] = varm_samples
                    self.hyperpar_samples['kernel_inverse_lengthscales'] = beta_samples
                    self.hyperpar_samples['gp_constant_mean_function'] = loc_samples
                    return

        else:
            print('Estimating the noise variance using EMMCMC')
            try:
                num_warmup_iters = mcmc_samples//2
                em_iters = mcmc_samples//3
                loc_samples, varm_samples, beta_samples, acceptance_rate, loss_history,_ = self.model.EM_with_MCMC(initial_state, num_warmup_iters,
                                                                                            em_iters, mcmc_samples, num_leapfrog_steps,learning_rate = 0.01)
            except Exception as e:
                traceback.print_exc()
            self.hyperpar_samples['kernel_variance'] = varm_samples
            self.hyperpar_samples['kernel_inverse_lengthscales'] = beta_samples
            self.hyperpar_samples['gp_constant_mean_function'] = loc_samples
            self.hyperpar_samples['loss_function_history'] = loss_history
            return


    def plot_chains(self, path = None, labels = []):
        # Function used to plot the chains from the  mcmc sampling
        # Inputs:
        #   path:= directory where to save the plots. Defaults to current directory
        #       if not specified
        #  labels:= list containing labels for the input variables. A default list is
        #       generated if this is not specified
        if len(self.hyperpar_samples) == 0:
            raise Exception('Execute run_mcmc first.')
        if labels == []:
            labels = ['x' + str(i) for i in range(self.n_inputs)]
        elif  (len(labels) != self.n_inputs) or not(all(isinstance(s, str) for s in labels)):
            raise Exception('Invalid input for labels')
        if path == None:
            path = './'
        if os.path.isdir(path):
            nplots = self.n_inputs +  2
            fig, axes = plt.subplots(nplots, 1, figsize=(20, 2.0*nplots),sharex=True)
            # plotting the samples for the kernel variance
            axes[0].plot(self.hyperpar_samples['kernel_variance'])
            title = 'kernel_variance_samples'
            axes[0].set_title(title)
            # plotting the samples for the constant mean function of the GP
            axes[1].plot(self.hyperpar_samples['gp_constant_mean_function'])
            title = 'gp_constant_mean_function_samples'
            axes[1].set_title(title)
            # plotting the samples for the inverse lengthscales
            for i in range(0,self.n_inputs):
                axes[i+2].plot(self.hyperpar_samples['kernel_inverse_lengthscales'][:,i])
                title =  labels[i] + '_inverse_lengthscale_samples'
                axes[i+2].set_title(title)
            figpath ='mcmc_chains.png'
            figpath = os.path.join(path, figpath)
            plt.savefig(figpath)
            plt.close()
        else:
            raise Exception('Invalid directory path ', path)
        return

    def plot_loss_function(self, path = None):
        # Function used to compute the loss function from the M-step executed while estimating
        # the noise variance
        # Inputs:
        #   path:= directory where to save the plots. Defaults to current directory
        #       if not specified
        if path == None:
            path = './'
        if os.path.isdir(path):
            if 'loss_function_history' in self.hyperpar_samples.keys():
                plt.figure(figsize=(12,10))
                plt.plot(self.hyperpar_samples['loss_function_history'])
                title = 'loss_function'
                plt.title(title)
                figpath = title + '.png'
                figpath = os.path.join(path, figpath)
                plt.savefig(figpath)
                plt.close()
            else:
                raise Exception('Loss function is not available.')
        else:
            raise Exception('Invalid directory path ', path)
        return
    def predict(self, Xtest, with_point_samples = False):
        # Computes approximate values of the full posterior mean and variance of the Gaussian process
		# by using samples of the posterior distribution of the hyperparameters
		#Inputs:
		#	Xtest :=  N x D input array
		#	with_point_samples = Boolean specifying if we sample from the posterior Gaussian
		#                       for each input and for each sample of the hyperparameters
		# Outputs:
		#	mean_pos_ := the mean for the full posterior Gaussian process (vector of length N)
		#  std_pos_ := the standard deviation  for the full posterior Gaussian process (vector of length N)
		# if with_point_samples == True, the function also outputs:
		# 	samples = samples for the full posterior Gaussian process (array of shape n_samples x N)
        # where n_samples = 30*max(1000,mcmc_samples) and mcmc_samples denotes the number of mcmc
        # samples obtained for the hyperparameters.
        mean_x, std_x = self.scaling_input
        mean_y, std_y = self.scaling_output

        Xtest_norm = (Xtest - mean_x)/std_x

        mcmc_samples = len(self.hyperpar_samples['kernel_variance'])
        # Limiting the number of mcmc samples used if necessary
        if mcmc_samples > 3000:
            selected = np.random.permutation(mcmc_samples)
            loc_samples = self.hyperpar_samples['gp_constant_mean_function'][selected]
            varm_samples = self.hyperpar_samples['kernel_variance'][selected]
            beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales'][selected]
        else:
            loc_samples = self.hyperpar_samples['gp_constant_mean_function']
            varm_samples = self.hyperpar_samples['kernel_variance']
            beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples]

        if with_point_samples:
            mean_pos, var_pos, samples = self.model.samples(Xtest_norm, hyperpar_samples, num_samples = 30, with_point_samples = True)
            std_pos = np.sqrt(var_pos)
            # Converting to the proper scale
            mean_pos = mean_pos*std_y + mean_y
            std_pos = std_pos*std_y
            samples = samples*std_y + mean_y

            return mean_pos, std_pos, samples

        else:
            mean_pos, var_pos = self.model.samples(Xtest_norm, hyperpar_samples)
            std_pos = np.sqrt(var_pos)
            # Converting to the proper scale
            mean_pos = mean_pos*std_y + mean_y
            std_pos = std_pos*std_y

            return mean_pos, std_pos
