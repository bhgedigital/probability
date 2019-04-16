from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
from tensorflow_probability.python.models.bayesgp.scripts import multibayesiangp
import os
import matplotlib.pyplot as plt
import traceback
import warnings
from tensorflow_probability.python.models.bayesgp.scripts import sensitivity
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import copy






# This class is used to run the Bayesian GP model. To initialize the model, the
# user needs to provide:
#   a) the input
#   b) the output
#   c) The type of kernel to use: 'RBF', 'Matern12', 'Matern32' or 'Matern52'
#   d) A value for the variance of the Gaussian noise of the output
# The data will be normalized internally. So the specified noise variance is
# expected to be between 0 and 1

class MultiBGP_model():

    def __init__(self, inputs, outputs, model_info = None, min_rank = None, kernel_type = 'RBF', noise_level = 1e-3, input_labels = None, output_labels = None, hyp_priors ={}):
        # Inputs:
        #   inputs := N x D numpy array of inputs
        #   outputs := N-dimensional numpy vector of outputs
        #   model_info = dictionary containing
        #               1) a dictionary containing the hyperparameter samples.
        #               2) the value of the noise variance
        #               3) the type of kernel used
        #   this dictionary must be generated  from a previous initialization of the model with the same set of inputs and  outputs
        # if model_info is not provided, the kernel_type and noise level can be provided or will be set as default
        #    kernel_type := string specifying the type of kernel to be used. Options are
		#                'RBF', 'Matern12', 'Matern32', 'Matern52'
        #   noise_level := variance of the Gaussian noise for the normalized data
        #   input_labels:= list containing labels for the input variables. A default list is
        #           generated if this is not specified
        #   output_labels:= list containing labels for the output variables. A default list is
        #           generated if this is not specified
        #  hyp_priors = dictionary containing information for the prior distribution on the hyperparameters



        if model_info:
            warnings.warn("Retrieving model info from previous run. The inputs and outputs arrays must be the same as the previous run.")
            try:
                self.hyperpar_samples = copy.deepcopy(model_info['samples'])
                self.kernel_type = model_info['kernel_type']
                Wmix = model_info['mixing_matrix']
                noise_level = model_info['noise_level']
            except Exception as e:
                traceback.print_exc()
                print('Failed to retrieve model info.')
        else:
            self.hyperpar_samples = {}
            self.kernel_type = kernel_type
            noise_level = noise_level
            Wmix = None

        if len(inputs.shape) == 1:
            self.n_inputs = 1
            X = inputs[:, None]
        else:
            self.n_inputs = inputs.shape[1]
            X = inputs

        if (len(outputs.shape) == 1) or (outputs.shape[1] == 1):
            raise Exception('Outputs array has only one column. It must have at least two outputs columns!')

        # normalizing the input
        mean_x = np.mean(X, axis = 0)
        std_x = np.std(X, axis = 0, keepdims = True)
        Xnorm = (X - mean_x)/std_x
        self.scaling_input = [mean_x, std_x]

        # Normalizing the outputs
        mean_y = np.mean(outputs, axis = 0)
        std_y = np.std(outputs, axis = 0, keepdims = True)
        Ynorm = (outputs - mean_y)/std_y
        self.n_tasks = outputs.shape[1]
        self.scaling_output = [mean_y, std_y]

        # Checking that the Gaussian noise variance array has entries greater than 0  and has the right shape
        noise_shape = np.array(noise_level.shape)
        correct_shape = np.array([self.n_tasks])
        invalid = not(np.array_equal(noise_shape, correct_shape))
        if invalid:
            raise Exception('Invalid shape for noise_level numpy array')
        invalid = np.sum(noise_level <= 0)
        if invalid:
            raise Exception('Invalid entries in noise_level numpy array. Entries must be positive.')

        # initializing the mixing matrix if necessary
        if Wmix is None:
            U, S, V = np.linalg.svd(Ynorm)
            cum_prop = np.cumsum(S)/np.sum(S)
            q = np.min(np.argwhere(cum_prop > 0.90)) + 1
            if min_rank == None:
                q = max(2,q)
            else:
                if min_rank > self.n_tasks:
                    raise Exception('Invalide value for min_rank. The rank of the mixing matrix cannot exceed the number of outputs.')
                q = max(min_rank, q)
            Wmix = np.transpose(V[:q,:])
        self.n_latent = Wmix.shape[1]


        self.model = multibayesiangp.MultiBayesianGP(inputs = Xnorm, outputs = Ynorm, Wmix_init = Wmix,
                                                    noise_level = noise_level, kernel_type = self.kernel_type, hyp_priors = hyp_priors) # initializing internal GP model

        # Bounds needed for sensitivity analysis
        mins_range = np.min(X, axis = 0,  keepdims = True).T
        maxs_range = np.max(X, axis = 0,keepdims = True).T
        self.Range = np.concatenate([mins_range, maxs_range], axis = 1)

        mins_range = np.min(Xnorm, axis = 0,  keepdims = True).T
        maxs_range = np.max(Xnorm, axis = 0,keepdims = True).T
        self.Rangenorm = np.concatenate([mins_range, maxs_range], axis = 1)

        if input_labels is None:
            self.input_labels = ['x' + str(i) for i in range(self.n_inputs)]
        elif  (len(input_labels) != self.n_inputs) or not(all(isinstance(s, str) for s in input_labels)):
            raise Exception('Invalid input for input_labels')
        else:
            self.input_labels = input_labels[:]

        if output_labels is None:
            self.output_labels = ['x' + str(i) for i in range(self.n_tasks)]
        elif  (len(output_labels) != self.n_tasks) or not(all(isinstance(s, str) for s in output_labels)):
            raise Exception('Invalid input for output_labels')
        else:
            self.output_labels = output_labels[:]

        return

    def run_mcmc(self, mcmc_samples,num_leapfrog_steps = 3, estimate_mixing_and_noise = False, em_iters = 400, learning_rate = 0.01, warm_up = True, step_size = 0.01):
        # Inputs:
        #   mcmc_samples := number of desired samples for the hyperparameters
        # num_leap_frog_steps = number of leap frog steps for the HMC sampler
        # estimated_noise := Boolean that indicates if the model should estimate a noise variance or keep it fixed (estimation is done wuth EM MCMC)
        # em_iters := number of EM-steps to perform if the noise variance is estimated
        # learning_rate := learning rate for optimizer if the noise variance is estimated
        # warm_up := Assuming the noise is kept fixed (i.e estimate_noise == False ), this Boolean  indicates if an adaptive step size is computed during a "warm up" phase
        # step_size := step size to use for the HMC sampler if warm_up == False
        # Output:
        #       model_info = dictionary containing
        #                      1) dictionary with samples of hyperparameters as well loss function history (if noise is estimated)
        #                      2) the value of the noise variance
        #                      3) the type of kernel used
        #
        if estimate_mixing_and_noise == False:
            print('Noise variance and mixing matrix are fixed.')
            if warm_up:
                # Execute a warmup phase to compute an adaptive step size
                burn_in  = mcmc_samples//2
                num_warmup_iters = mcmc_samples
                try:
                    print('Excecuting the warmup.')
                    step_size, next_state = self.model.warmup(num_warmup_iters = num_warmup_iters, num_leapfrog_steps = num_leapfrog_steps)
                    if step_size  < 1e-4:
                        warnings.warn("Estimated step size is low. (less than 1e-4)")
                    print('Sampling in progress.')
                    hyperpar_samples, acceptance_rate = self.model.mcmc(mcmc_samples = mcmc_samples, num_burnin_steps =burn_in,step_size = step_size,
                                                                    num_leapfrog_steps = num_leapfrog_steps, initial_state = next_state)
                    if acceptance_rate < 0.1:
                        warnings.warn("Acceptance rate was low  (less than 0.1)")
                except Exception as e:
                    traceback.print_exc()
                    print('Sampling failed. Increase the noise level or decrease the step size or the number of leap frog steps if necessary.')
            else:
                try:
                    burn_in  = mcmc_samples
                    print('Sampling in progress.')
                    hyperpar_samples, acceptance_rate = self.model.mcmc(mcmc_samples = mcmc_samples, num_burnin_steps =burn_in,step_size = step_size,
                                                                    num_leapfrog_steps = num_leapfrog_steps)
                    if acceptance_rate < 0.1:
                        warnings.warn("Acceptance rate was low  (less than 0.1)")
                except Exception as e:
                    traceback.print_exc()
                    print('Sampling failed. Increase the noise level or decrease the step size or the number of leap frog steps if necessary.')

        else:
            print('Estimating the noise variance and the mixing matrix using EMMCMC')
            try:
                num_warmup_iters = (3*mcmc_samples)//4
                hyperpar_samples, loss_history,_ = self.model.EM_with_MCMC(num_warmup_iters = num_warmup_iters, em_iters = em_iters,
                                                                    mcmc_samples = mcmc_samples, num_leapfrog_steps = num_leapfrog_steps, learning_rate = learning_rate, display_rate = 400)
                self.hyperpar_samples['loss_function_history'] = loss_history
            except Exception as e:
                traceback.print_exc()
        loc_samples, varm_samples, beta_samples, varc_samples = hyperpar_samples
        self.hyperpar_samples['kernel_variance'] = varm_samples
        self.hyperpar_samples['common_kernel_variance'] = varc_samples
        self.hyperpar_samples['kernel_inverse_lengthscales'] = beta_samples
        self.hyperpar_samples['gp_constant_mean_function'] = loc_samples

        model_info = {}
        model_info['samples'] = copy.deepcopy(self.hyperpar_samples)
        model_info['kernel_type'] = self.kernel_type
        with tf.Session() as sess:
            model_info['mixing_matrix'] = self.model.Wmix.eval()
            model_info['noise_level'] = self.model.noise.eval()

        return model_info


    def plot_chains(self, directory_path = None):
        # Function used to plot the chains from the  mcmc sampling
        # Inputs:
        #   directory_path:= directory where to save the plots. It defaults to the current directory if not specified
        #

        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)

        M = self.n_tasks
        Q = self.n_latent
        D = self.n_inputs

        # plotting the samples for loc
        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        for k in range(M):
            plt.figure(figsize=(20,10))
            plt.plot(loc_samples[:,k])
            title = 'loc_' + str(k)
            figpath = title + '.png'
            plt.title(title)
            figpath = os.path.join(directory_path, figpath)
            plt.savefig(figpath)
            plt.close()

        # plotting the samples for the the array of variances
        varm_samples = self.hyperpar_samples['kernel_variance']
        for k in range(M):
            for q in range(Q):
                plt.figure(figsize=(20,10))
                plt.plot(varm_samples[:,k,q])
                title = 'varm_' + str(k) + '_' + str(q)
                figpath = title + '.png'
                plt.title(title)
                figpath = os.path.join(directory_path, figpath)
                plt.savefig(figpath)
                plt.close()

        #plotting the samples for beta
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        for q in range(Q):
            for d in range(D):
                plt.figure(figsize=(20,10))
                plt.plot(beta_samples[:,q,d])
                title = 'beta_' + str(q) + '_' + self.input_labels[d]
                figpath = title + '.png'
                plt.title(title)
                figpath = os.path.join(directory_path, figpath)
                plt.savefig(figpath)
                plt.close()

        # plotting the samples for varc
        varc_samples = self.hyperpar_samples['common_kernel_variance']
        plt.figure(figsize=(20,10))
        plt.plot(varc_samples)
        title = 'varc'
        figpath = title + '.png'
        plt.title(title)
        figpath = os.path.join(directory_path, figpath)
        plt.savefig(figpath)
        plt.close()

        return

    def plot_loss_function(self, directory_path = None):
        # Function used to compute the loss function from the M-step executed while estimating
        # the noise variance
        # Inputs:
        #   path:= directory where to save the plots. Defaults to current directory
        #       if not specified
        if not('loss_function_history' in self.hyperpar_samples.keys()):
            print('Loss function is not available.')
            return
        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)
        plt.figure(figsize=(12,10))
        plt.plot(self.hyperpar_samples['loss_function_history'])
        title = 'loss_function'
        plt.title(title)
        figpath = title + '.png'
        figpath = os.path.join(directory_path, figpath)
        plt.savefig(figpath)
        plt.close()
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

        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')

        mean_x, std_x = self.scaling_input
        mean_y, std_y = self.scaling_output

        if len(Xtest.shape) == 1:
            Xnew = Xtest[:, None]
        else:
            Xnew = Xtest
        Xtest_norm = (Xnew - mean_x)/std_x

        mcmc_samples = len(self.hyperpar_samples['kernel_variance'])
        # Limiting the number of mcmc samples used if necessary
        if mcmc_samples > 3000:
            selected = np.random.permutation(mcmc_samples)
            loc_samples = self.hyperpar_samples['gp_constant_mean_function'][selected]
            varm_samples = self.hyperpar_samples['kernel_variance'][selected]
            beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales'][selected]
            varc_samples = self.hyperpar_samples['common_kernel_variance'][selected]
        else:
            loc_samples = self.hyperpar_samples['gp_constant_mean_function']
            varm_samples = self.hyperpar_samples['kernel_variance']
            beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
            varc_samples = self.hyperpar_samples['common_kernel_variance']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples, varc_samples]

        if with_point_samples:
            mean_pos, var_pos, samples = self.model.samples(Xtest_norm, hyperpar_samples, num_samples = 20, with_point_samples = True)
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

    # Sensitivity analysis
    def maineffect_and_interaction(self, grid_points = 30, nx_samples = None, directory_path1 = None, directory_path2 = None, create_plot = True):
        # Computes  and generate main_effect function plots
        # Inputs:
        #   grid_points:= the number of grid poinst for the plots
        #   nx_samples = the number of sample points for the Monte Carlo integration. Will default
        #       to a multiple of the number of variables if not provided
        #   directory_path1 :=  directory where to save the main effect plots if needed. Defaults to current directory
        #       if not specified
        #   directory_path2 :=  directory where to save the interaction surface plots if needed. Defaults to current directory
        #       if not specified
        #  create_plot := specifies if the plost should be generated are not
        # Outputs:
        #      main, interaction: = dictionaries containing values for the mean and interaction functions

        # Main effect
        if self.n_inputs == 1:
            print('Not enough variables to perform sensitivity analysis.')
            return
        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')
        if directory_path1 == None:
            directory_path1 = os.getcwd()
        if not(os.path.isdir(directory_path1)):
            raise Exception('Invalid directory path ', directory_path1)
        if directory_path2 == None:
            directory_path2 = os.getcwd()
        if not(os.path.isdir(directory_path2)):
            raise Exception('Invalid directory path ', directory_path2)

        # get list of gpu devices to parallelize computation if possible
        M = self.n_tasks
        D = self.n_inputs
        mean_y, std_y = self.scaling_output
        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        varm_samples = self.hyperpar_samples['kernel_variance']
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        varc_samples = self.hyperpar_samples['common_kernel_variance']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples, varc_samples]
        if nx_samples == None:
            nx_samples = 400*D
        selected_vars = [i for i in range(D)]
        if D <= 6:
            y_main = sensitivity.mainEffect(self.model, self.Rangenorm, selected_vars, nx_samples, hyperpar_samples, grid_points)
        else:
            y_main = {}
            vars_groups = np.array_split(selected_vars,6)
            completed = 0
            for group in vars_groups:
                y_group = sensitivity.mainEffect(self.model, self.Rangenorm, group, nx_samples, hyperpar_samples, grid_points)
                completed += len(group)
                progress = 100.0*completed/self.n_inputs
                print("Main effect computation: {:.2f}% complete".format(progress))
                y_main.update(y_group)

        ybase = sensitivity.allEffect(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
        z_mean = np.zeros((M,D,grid_points))
        z_std = np.zeros((M,D,grid_points))
        for i in range(D):
            for j in range(M):
                key = tuple([i])
                z_mean[j,i,:] = y_main[key][:,j,0] - ybase[0][0,j,0]
                # The next 3 lines give an approximation of the standard deviation of the normalized main effect function E[Y|xi]
                lower_app = np.sqrt(np.abs(np.sqrt(y_main[key][:,j,1]) - np.sqrt(ybase[0][0,j,1])))
                upper_app = np.sqrt(y_main[key][:,j,1]) + np.sqrt(ybase[0][0,j,1])
                z_std[j,i,:] = (lower_app + upper_app)/2.0

        # Converting to the proper scale and plotting
        main = {}
        for i in range(D):
            for j in range(M):
                y = z_mean[j,i,:]*std_y[0,j] + mean_y[j]
                y_std = z_std[j,i,:]*std_y[0,j]
                x = np.linspace(self.Range[i,0], self.Range[i,1], grid_points)
                key = self.output_labels[j] + '_vs_' + self.input_labels[i]
                main[key] = {}
                main[key]['inputs'] = x
                main[key]['output_mean']= y
                main[key]['output_std']= y_std
        if create_plot:
            fig, axes = plt.subplots(nrows=M, ncols=D, sharex = 'col', sharey='row', figsize =(15,15))
            for i in range(D):
                for j in range(M):
                    key = self.output_labels[j] + '_vs_' + self.input_labels[i]
                    x = main[key]['inputs']
                    y = main[key]['output_mean']
                    y_std = main[key]['output_std']
                    axes[j,i].plot(x,y, label= self.input_labels[i])
                    axes[j,i].fill_between(x, y-2*y_std, y + 2*y_std, alpha = 0.2, color ='orange')
                    axes[j,i].grid()
                    axes[j,i].legend()
            title = 'main_effects'
            plt.title(title)
            figpath = title + '.png'
            figpath = os.path.join(directory_path1, figpath)
            plt.savefig(figpath)
            plt.close(fig)

        #---------------------------------------------------------------------
        # Interaction effect
        selected_pairs = []
        for i in range(D-1):
            for j in range(i+1,D):
                selected_pairs.append([i,j])
        selected_pairs = np.array(selected_pairs)
        n_pairs = len(selected_pairs)
        if n_pairs <= 6:
            y_int = sensitivity.mainInteraction(self.model, self.Rangenorm, selected_pairs, nx_samples, hyperpar_samples, grid_points)
        else:
            y_int = {}
            pairs_groups = np.array_split(selected_pairs,6)
            completed = 0
            for group in pairs_groups:
                y_group = sensitivity.mainInteraction(self.model, self.Rangenorm, group, nx_samples, hyperpar_samples, grid_points)
                completed += len(group)
                progress = 100.0*completed/n_pairs
                print("Main interaction computation: {:.2f}% complete".format(progress))
                y_int.update(y_group)
        z_intmean = np.zeros((M ,n_pairs, grid_points, grid_points))
        z_intstd = np.zeros((M, n_pairs, grid_points, grid_points))
        for k in  range(n_pairs):
            key = tuple(selected_pairs[k])
            j1, j2 = selected_pairs[k]
            idx1 = selected_vars.index(j1)
            idx2 = selected_vars.index(j2)
            key1 = tuple([idx1])
            key2 = tuple([idx2])
            y_slice = np.reshape(y_int[key], (grid_points, grid_points,M,2))
            for j in range(M):
                v1 = y_main[key1][:,j,1]
                v2 = y_main[key2][:,j,0]
                p1, p2 = np.meshgrid(v1,v2)
                w1 = np.sqrt(y_main[key1][:,j,1])
                w2 = np.sqrt(y_main[key2][:,j,1])
                q1, q2 = np.meshgrid(w1,w2)
                z_intmean[j,k,:,:] = y_slice[:,:,j,0] - p1 - p2 +  ybase[0][0,j,0]
                upper_app = np.sqrt(y_slice[:,:,j,1]) + q1 + q2 + np.sqrt(ybase[0][0,j,1])
                lower_app = np.abs( np.sqrt(y_slice[:,:,j,1]) - q1 - q2 + np.sqrt(ybase[0][0,j,1]) )
                z_intstd[j,k,:,:] = (upper_app + lower_app)/2.0

        # Converting to the proper scale and storing
        interaction = {}
        for k in range(n_pairs):
            for j in range(M):
                item = selected_pairs[k]
                j1, j2 = item
                x = np.linspace(self.Range[j1,0],self.Range[j1,1],grid_points)
                y = np.linspace(self.Range[j2,0],self.Range[j2,1],grid_points)
                Z = z_intmean[j,k,:,:]*std_y[0,j] + mean_y[j]
                Zstd = z_intstd[j,k,:,:]*std_y[0,j]
                key = self.output_labels[j] + '_vs_' + self.input_labels[j1] + '_&_' + self.input_labels[j2]
                X,  Y = np.meshgrid(x,y)
                interaction[key] = {}
                interaction[key]['input1'] = X
                interaction[key]['input2'] = Y
                interaction[key]['output_mean'] = Z
                interaction[key]['output_std'] = Zstd

        if create_plot:
            # Bounds for the interaction surface plot
            zmin = np.amin(z_intmean, axis =(1,2,3))*std_y[0,:] + mean_y
            zmax = np.amax(z_intmean, axis = (1,2,3))*std_y[0,:] + mean_y
            minn = np.amin(z_intstd, axis = (1,2,3))*std_y[0,:]
            maxx = np.amax(z_intstd, axis = (1,2,3))*std_y[0,:]

            for k in range(n_pairs):
                for j in range(M):
                    item = selected_pairs[k]
                    j1, j2 = item
                    key = self.output_labels[j] + '_vs_' + self.input_labels[j1] + '_&_' + self.input_labels[j2]
                    X = interaction[key]['input1']
                    Y = interaction[key]['input2']
                    Z = interaction[key]['output_mean']
                    Zstd = interaction[key]['output_std']
                    fig = plt.figure(figsize = (20,10))
                    norm = mpl.colors.Normalize(minn[j], maxx[j])
                    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
                    m.set_array(Zstd)
                    m.set_clim(minn[j], maxx[j])
                    color_dimension = Zstd
                    fcolors = m.to_rgba(color_dimension)
                    ax = fig.gca(projection='3d')
                    ax.plot_surface(Y, X, Z, rstride=1, cstride=1,facecolors= fcolors, shade = False)
                    title = key
                    ax.set_title(title)
                    ax.set_xlabel(self.input_labels[j2])
                    ax.set_ylabel(self.input_labels[j1])
                    ax.set_zlim(zmin[j],zmax[j])
                    plt.gca().invert_xaxis()
                    plt.colorbar(m)
                    figpath = title + '.png'
                    figpath = os.path.join(directory_path2, figpath)
                    plt.savefig(figpath)
                    plt.close(fig)

        return main, interaction



    def sobol_indices(self,  max_order = 2, S = None, nx_samples = None, directory_path = None, create_plot = True):
        # Computes sobol indices and generate bar plot.
        # Inputs:
        #   Sobol_store := dictionary containing previously computed Sobol indices. The computation of
        #      Sobol indices is a recursive computation
        #   max_order := maximum order of sobol indices to compute
        #   nx_samples = the number of sample points for the Monte Carlo integration. Will default
        #           to a multiple of the number of variables if not provided
        #  directory_path :=  directory where to save the Sobol barplot if needed. Defaults to current directory
        #       if not specified
        #  create_plot := specifies if the Sobol barplot should be generated are not
        # Outputs:
        #       Sobol := dictionary containing the Sobol indices values
        if max_order >  self.n_inputs:
            raise Exception('max_order cannot be greater than the number of variables')
        if self.n_inputs == 1:
            print('Not enough variables to perform sensitivity analysis.')
            return
        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')
        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)

        # get list of gpu devices to parallelize computation if possible

        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        varm_samples = self.hyperpar_samples['kernel_variance']
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        varc_samples = self.hyperpar_samples['common_kernel_variance']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples, varc_samples]
        if nx_samples == None:
            nx_samples = 300*self.n_inputs
        selected_vars = [i for i in range(self.n_inputs)]

        initial_list  = sensitivity.powerset(selected_vars,1, max_order)
        subsets_list = []
        if S != None:
            print('Initial number of Sobol computations: ', len(initial_list))
            try:
                for item in initial_list:
                    l = sensitivity.generate_label(item, self.input_labels)
                    if not (l in S.keys()):
                        subsets_list.append(item)
                print('New number of Sobol computations: ', len(subsets_list))
            except Exception as e:
                traceback.print_exc()
                print('Invalid Sobol indices dictionary')
        else:
            subsets_list = initial_list

        n_subset = len(subsets_list)
        M = self.n_tasks
        if n_subset > 0:
            ybase = sensitivity.allEffect(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
            ey_square = sensitivity.direct_samples(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
            ey_square = np.reshape(ey_square,(M,nx_samples,2))
            if n_subset <= 10:
                y_higher_order = sensitivity.mainHigherOrder(self.model, self.Rangenorm, subsets_list, nx_samples, hyperpar_samples)
            else:
                y_higher_order = {}
                completed = 0
                n_groups = math.ceil(n_subset/10)
                for i in range(n_groups):
                    group = subsets_list[i*10:(i+1)*10]
                    y_group = sensitivity.mainHigherOrder(self.model, self.Rangenorm, group, nx_samples, hyperpar_samples)
                    completed += len(group)
                    progress = 100.0*completed/n_subset
                    print("Sobol indices computation: {:.2f}% complete".format(progress))
                    y_higher_order.update(y_group)

            e1 = np.mean(ey_square[:,:,1] + np.square(ey_square[:,:,0]), axis = 1)
            e2 = ybase[0][0,:,1] + np.square(ybase[0][0,:,0])
            # This will store the quantities E*[Vsub]/E*(Var(Y)) where Vsub = E[Y|Xsub] and Y is normalized
            quotient_variances = {}

            for idx in range(n_subset):
                key = tuple(subsets_list[idx])
                quotient_variances[key] = np.mean(y_higher_order[key][:,:,1] + np.square(y_higher_order[key][:,:,0]), axis = 0)
                quotient_variances[key] = (quotient_variances[key] - e2)/(e1-e2)
        if S != None:
            Sobol = S
        else:
            Sobol = {}
        for i in range(n_subset):
            key = tuple(subsets_list[i])
            sensitivity.compute_Sobol(Sobol, quotient_variances, key, self.input_labels)

        all_labels = list(Sobol.keys())

        # plotting
        n_selected = min(40, len(all_labels))
        y_pos = np.arange(n_selected)
        if create_plot:
            si_all = {}
            for j in range(M):
                si_all[j] = []
                for i in range(len(all_labels)):
                    key = all_labels[i]
                    si_all[j].append(Sobol[key][j])
                si_all[j] = np.array(si_all[j])
                order = np.argsort(-si_all[j])
                selected = order[:n_selected] # taking the top 40 values to plot
                plt.figure(figsize =(12,12))
                # Create bars
                plt.barh(y_pos, si_all[j][selected])
                new_labels = [all_labels[selected[i]] for i in range(n_selected)]
                title = 'top_sobol_indices_for_' + self.output_labels[j]
                plt.title(title)
                # Create names on the x-axis
                plt.yticks(y_pos, new_labels)
                figpath = title + '.png'
                figpath = os.path.join(directory_path, figpath)
                plt.savefig(figpath)
                plt.close()

        return Sobol

    def total_sobol_indices(self, nx_samples = None, directory_path = None, create_plot = True):
        # Computes total sobol indices and generate bar plot.
        # Inputs:
        #   nx_samples = the number of sample points for the Monte Carlo integration. Will default
        #           to a multiple of the number of variables if not provided
        #  directory_path :=  directory where to save the Sobol barplot if needed. Defaults to current directory
        #       if not specified
        #    create_plot := specifies if the total Sobol barplot should be generated are not
        # Outputs:
        #   Sobol_total := dictionary containining containing the total Sobol indices values


        if self.n_inputs == 1:
            print('Not enough variables to perform sensitivity analysis.')
            return
        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')
        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)

        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        varm_samples = self.hyperpar_samples['kernel_variance']
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        varc_samples = self.hyperpar_samples['common_kernel_variance']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples, varc_samples]
        if nx_samples == None:
            nx_samples = 300*self.n_inputs
        selected_vars = [i for i in range(self.n_inputs)]
        M = self.n_tasks
        D = self.n_inputs

        ybase = sensitivity.allEffect(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
        ey_square = sensitivity.direct_samples(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
        ey_square = np.reshape(ey_square,(M,nx_samples,2))
        if self.n_inputs  <= 6:
            y_remaining  = sensitivity.compute_remaining_effect(self.model, self.Rangenorm, selected_vars, nx_samples, hyperpar_samples)
        else:
            y_remaining = {}
            vars_groups = np.array_split(selected_vars,6)
            completed = 0
            for group in vars_groups:
                y_group = sensitivity.compute_remaining_effect(self.model, self.Rangenorm, group, nx_samples, hyperpar_samples)
                completed += len(group)
                progress = 100.0*completed/n_vars
                print("Total Sobol indices computation: {:.2f}% complete".format(progress))
                y_remaining.update(y_group)

        e1 = np.mean(ey_square[:,:,1] + np.square(ey_square[:,:,0]), axis = 1)
        e2 = ybase[0][0,:,1] + np.square(ybase[0][0,:,0])

        si_remaining = np.zeros((M,D))
        for i in range(D):
            key = tuple([i])
            si_remaining[:,i] = np.mean(y_remaining[key][:,:,1] + np.square(y_remaining[key][:,:,0]), axis = 0)
            si_remaining  = (si_remaining -e2[:,np.newaxis])/(e1[:,np.newaxis]-e2[:,np.newaxis])
            si_remaining = np.maximum(si_remaining,0)
        si_total = 1 - si_remaining
        si_total = np.maximum(si_total,0)

        if create_plot:
            #  generating the plot
            n_selected = min(40, D)
            y_pos = np.arange(n_selected)
            for j in range(M):
                order = np.argsort(-si_total[j,:])
                selected = order[:n_selected] # taking the top 40 values to plot
                plt.figure(figsize =(12,12))
                # Create bars
                plt.barh(y_pos, si_total[j,selected])
                new_labels = [self.input_labels[selected[i]] for i in range(n_selected)]
                title = 'top_total_sobol_indices_for_' + self.output_labels[j]
                plt.title(title)
                # Create names on the x-axis
                plt.yticks(y_pos, new_labels)
                figpath = title + '.png'
                figpath = os.path.join(directory_path, figpath)
                plt.savefig(figpath)
                plt.close()

        Sobol_total = {}
        for i in range(self.n_inputs):
            l = self.input_labels[i]
            Sobol_total[l] = si_total[:,i]

        return Sobol_total
