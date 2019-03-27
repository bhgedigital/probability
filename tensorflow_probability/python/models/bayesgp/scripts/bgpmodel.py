from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
from tensorflow_probability.python.models.bayesgp.scripts import bayesiangp
import os
import matplotlib.pyplot as plt
import traceback
import warnings
from tensorflow_probability.python.models.bayesgp.scripts import sensitivity
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
from tensorflow.python.client import device_lib
import copy






# This class is used to run the Bayesian GP model. To initialize the model, the
# user needs to provide:
#   a) the input
#   b) the output
#   c) The type of kernel to use: 'RBF', 'Matern12', 'Matern32' or 'Matern52'
#   d) A value for the variance of the Gaussian noise of the output
# The data will be normalized internally. So the specified noise variance is
# expected to be between 0 and 1

class BGP_model():

    def __init__(self, inputs, outputs, model_info = None, kernel_type = 'RBF', noise_level = 1e-3, labels = []):
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
        #   labels:= list containing labels for the input variables. A default list is
        #           generated if this is not specified


        if model_info:
            warnings.warn("Retrieving model info from previous run. The inputs and outputs arrays must be the same as the previous run.")
            try:
                self.hyperpar_samples = copy.deepcopy(model_info['samples'])
                self.kernel_type = model_info['kernel_type']
                noise_level = model_info['noise_level']
            except Exception as e:
                traceback.print_exc()
                print('Failed to retrieve model info.')
        else:
            self.hyperpar_samples = {}
            self.kernel_type = kernel_type
            noise_level = noise_level

        # Checking that the Gaussian noise variance is between 0 and 1
        if (noise_level > 1) or (noise_level < 0):
            raise Exception('Invalid value for the noise_level: ' + str(noise_level) + '. It should be between 0 and 1.')

        if len(inputs.shape) == 1:
            self.n_inputs = 1
            X = inputs[:, None]
        else:
            self.n_inputs = inputs.shape[1]
            X = inputs

        # normalizing the input
        mean_x = np.mean(X, axis = 0)
        std_x = np.std(X, axis = 0, keepdims = True)
        Xnorm = (X - mean_x)/std_x
        self.scaling_input = [mean_x, std_x]

        # Normalizing the outputs
        mean_y = np.mean(outputs)
        std_y = np.std(outputs)
        Ynorm = (outputs - mean_y)/std_y
        self.scaling_output = [mean_y, std_y]



        self.model = bayesiangp.BayesianGP(Xnorm, Ynorm, self.kernel_type, noise_level) # initializing internal GP model

        # Bounds needed for sensitivity analysis
        mins_range = np.min(X, axis = 0,  keepdims = True).T
        maxs_range = np.max(X, axis = 0,keepdims = True).T
        self.Range = np.concatenate([mins_range, maxs_range], axis = 1)

        mins_range = np.min(Xnorm, axis = 0,  keepdims = True).T
        maxs_range = np.max(Xnorm, axis = 0,keepdims = True).T
        self.Rangenorm = np.concatenate([mins_range, maxs_range], axis = 1)

        if labels == []:
            self.labels = ['x' + str(i) for i in range(self.n_inputs)]
        elif  (len(labels) != self.n_inputs) or not(all(isinstance(s, str) for s in labels)):
            raise Exception('Invalid input for labels')
        else:
            self.labels = labels

        return

    def run_mcmc(self, mcmc_samples,num_leapfrog_steps = 3, estimate_noise = False, em_iters = 400, learning_rate = 0.01, warm_up = True, step_size = 0.01):
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
        if estimate_noise == False:
            print('Noise variance is fixed.')
            if warm_up:
                # Execute a warmup phase to compute an adaptive step size
                burn_in  = mcmc_samples//2
                num_warmup_iters = burn_in
                try:
                    print('Excecuting the warmup.')
                    step_size, next_state = self.model.warmup(num_warmup_iters = num_warmup_iters, num_leapfrog_steps = num_leapfrog_steps)
                    if step_size  < 1e-4:
                        warnings.warn("Estimated step size is low. (less than 1e-4)")
                    print('Sampling in progress.')
                    hyperpar_samples, acceptance_rate = self.model.mcmc(mcmc_samples = mcmc_samples, num_burnin_steps =burn_in,step_size = 0.9*step_size,
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
            print('Estimating the noise variance using EMMCMC')
            try:
                num_warmup_iters = mcmc_samples//2
                hyperpar_samples, loss_history,_ = self.model.EM_with_MCMC(num_warmup_iters = num_warmup_iters, em_iters = em_iters,
                                                                    mcmc_samples = mcmc_samples, num_leapfrog_steps = num_leapfrog_steps, learning_rate = learning_rate)
                self.hyperpar_samples['loss_function_history'] = loss_history
            except Exception as e:
                traceback.print_exc()
        loc_samples, varm_samples, beta_samples = hyperpar_samples
        self.hyperpar_samples['kernel_variance'] = varm_samples
        self.hyperpar_samples['kernel_inverse_lengthscales'] = beta_samples
        self.hyperpar_samples['gp_constant_mean_function'] = loc_samples

        model_info = {}
        model_info['samples'] = copy.deepcopy(self.hyperpar_samples)
        model_info['kernel_type'] = self.kernel_type
        model_info['noise_level'] = self.model.noise

        return model_info

    def plot_chains(self, directory_path = None):
        # Function used to plot the chains from the  mcmc sampling
        # Inputs:
        #   directory_path:= directory where to save the plots. It defaults to the current directory if not specified
        #
        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')

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
            title =  self.labels[i] + '_inverse_lengthscale_samples'
            axes[i+2].set_title(title)

        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)
        figpath ='mcmc_chains.png'
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
        local_device_protos = device_lib.list_local_devices()
        devices_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if len(devices_list) == 0:
            devices_list = ['/cpu:0']
        mean_y, std_y = self.scaling_output
        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        varm_samples = self.hyperpar_samples['kernel_variance']
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples]
        if nx_samples == None:
            nx_samples = 300*self.n_inputs
        selected_vars = [i for i in range(self.n_inputs)]
        y_main = sensitivity.mainEffect(self.model, self.Rangenorm, selected_vars, nx_samples, hyperpar_samples, devices_list, grid_points)
        ybase = sensitivity.allEffect(self.model, self.Rangenorm, nx_samples, hyperpar_samples, devices_list)
        z_mean = y_main[:,:,0] - ybase[0] # mean value of the normalized main effect function E[Y|xi]

        # The next 3 lines give an approximation of the standard deviation of the normalized main effect function E[Y|xi]
        lower_app = np.sqrt(np.abs(np.sqrt(y_main[:,:,1]) - np.sqrt(ybase[1])))
        upper_app = np.sqrt(y_main[:,:,1]) + np.sqrt(ybase[1])
        z_std = (lower_app + upper_app)/2.0

        # Converting to the proper scale and plotting
        main = {}
        for i in range(self.n_inputs):
            y = z_mean[i,:]*std_y + mean_y
            y_std = z_std[i,:]*std_y
            x = np.linspace(self.Range[i,0], self.Range[i,1], grid_points)
            key = self.labels[i]
            main[key] = {}
            main[key]['inputs'] = x
            main[key]['output_mean']= y
            main[key]['output_std']= y_std
        if create_plot:
            fig, axes = plt.subplots(nrows=1, ncols=self.n_inputs, sharey=True, figsize =(20,10))
            for i in range(self.n_inputs):
                key = self.labels[i]
                x = main[key]['inputs']
                y = main[key]['output_mean']
                y_std = main[key]['output_std']
                axes[i].plot(x,y, label= self.labels[i])
                axes[i].fill_between(x, y-2*y_std, y + 2*y_std, alpha = 0.2, color ='orange')
                axes[i].grid()
                axes[i].legend()
            title = 'main_effects'
            plt.title(title)
            figpath = title + '.png'
            figpath = os.path.join(directory_path1, figpath)
            plt.savefig(figpath)
            plt.close(fig)

        #---------------------------------------------------------------------
        # Interaction effect
        selected_pairs = []
        for i in range(self.n_inputs-1):
            for j in range(i+1,self.n_inputs):
                selected_pairs.append([i,j])
        selected_pairs = np.array(selected_pairs)
        y_int = sensitivity.mainInteraction(self.model, self.Rangenorm, selected_pairs, nx_samples, hyperpar_samples, devices_list, grid_points)
        n_pairs = len(selected_pairs)
        z_intmean = np.zeros((n_pairs, grid_points, grid_points))
        z_intstd = np.zeros((n_pairs, grid_points, grid_points))
        for k in  range(n_pairs):
            y_slice = y_int[k,:,:]
            j1, j2 = selected_pairs[k]
            idx1 = selected_vars.index(j1)
            idx2 = selected_vars.index(j2)
            v1 = y_main[idx1,:,0]
            v2 = y_main[idx2,:,0]
            p1, p2 = np.meshgrid(v1,v2)
            w1 = np.sqrt(y_main[idx1,:,1])
            w2 = np.sqrt(y_main[idx2,:,1])
            q1, q2 = np.meshgrid(w1,w2)
            z_intmean[k,:,:] = np.reshape(y_slice[:,0], (grid_points, grid_points)) - p1 - p2 +  ybase[0]
            # The next 3 lines give an approximation of the standard deviation of the normalized interaction effect function E[Y|xi xj]
            upper_app = np.sqrt(np.reshape(y_slice[:,1], (grid_points, grid_points))) + q1 + q2 + np.sqrt(ybase[1])
            lower_app = np.abs(np.sqrt(np.reshape(y_slice[:,1], (grid_points, grid_points))) - q1 - q2 + np.sqrt(ybase[1]))
            z_intstd[k,:,:] = (upper_app + lower_app)/2.0

        # Converting to the proper scale and storing
        interaction = {}
        for k in range(n_pairs):
            item = selected_pairs[k]
            j1, j2 = item
            x = np.linspace(self.Range[j1,0],self.Range[j1,1],grid_points)
            y = np.linspace(self.Range[j2,0],self.Range[j2,1],grid_points)
            Z = z_intmean[k,:,:]*std_y + mean_y
            Zstd = z_intstd[k,:,:]*std_y
            key = self.labels[j1] + ' || ' + self.labels[j2]
            X,  Y = np.meshgrid(x,y)
            interaction[key] = {}
            interaction[key]['input1'] = X
            interaction[key]['input2'] = Y
            interaction[key]['output_mean'] = Z
            interaction[key]['output_std'] = Zstd

        if create_plot:
            # Bounds for the interaction surface plot
            zmin = np.min(z_intmean)*std_y + mean_y
            zmax = np.max(z_intmean)*std_y + mean_y
            minn = np.min(z_intstd)*std_y
            maxx = np.max(z_intstd)*std_y

            for k in range(n_pairs):
                item = selected_pairs[k]
                j1, j2 = item
                key = self.labels[j1] + ' || ' + self.labels[j2]
                X = interaction[key]['input1']
                Y = interaction[key]['input2']
                Z = interaction[key]['output_mean']
                Zstd = interaction[key]['output_std']
                fig = plt.figure(figsize = (20,10))
                norm = mpl.colors.Normalize(minn, maxx)
                m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
                m.set_array(Zstd)
                m.set_clim(minn, maxx)
                color_dimension = Zstd
                fcolors = m.to_rgba(color_dimension)
                ax = fig.gca(projection='3d')
                ax.plot_surface(Y, X, Z, rstride=1, cstride=1,facecolors= fcolors, shade = False)
                title = key
                ax.set_title(title)
                ax.set_xlabel(self.labels[j2])
                ax.set_ylabel(self.labels[j1])
                ax.set_zlim(zmin,zmax)
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
        local_device_protos = device_lib.list_local_devices()
        devices_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if len(devices_list) == 0:
            devices_list = ['/cpu:0']

        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        varm_samples = self.hyperpar_samples['kernel_variance']
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples]
        if nx_samples == None:
            nx_samples = 300*self.n_inputs
        selected_vars = [i for i in range(self.n_inputs)]
        selected_vars = [i for i in range(self.n_inputs)]

        initial_list  = sensitivity.powerset(selected_vars,1, max_order)
        subsets_list = []
        if S != None:
            print('Initial number of Sobol computations: ', len(initial_list))
            try:
                for item in initial_list:
                    l = sensitivity.generate_label(item, self.labels)
                    if not (l in S.keys()):
                        subsets_list.append(item)
                print('New number of Sobol computations: ', len(subsets_list))
            except Exception as e:
                traceback.print_exc()
                print('Invalid Sobol indices dictionary')
        else:
            subsets_list = initial_list

        n_subset = len(subsets_list)
        if n_subset > 0:
            ybase = sensitivity.allEffect(self.model, self.Rangenorm, nx_samples, hyperpar_samples, devices_list)
            ey_square = sensitivity.direct_samples(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
            y_higher_order = sensitivity.mainHigherOrder(self.model, self.Rangenorm, subsets_list, nx_samples, hyperpar_samples, devices_list)



            e1 = np.mean(ey_square[:,1] + np.square(ey_square[:,0]))
            e2 = ybase[1] + np.square(ybase[0])
            # This will store the quantities E*[Vsub]/E*(Var(Y)) where Vsub = E[Y|Xsub] and Y is normalized
            quotient_variances = {}

            for idx in range(n_subset):
                k = tuple(subsets_list[idx])
                quotient_variances[k] = np.mean(y_higher_order[idx,:,1] + np.square(y_higher_order[idx,:,0]))
                quotient_variances[k] = (quotient_variances[k] - e2)/(e1-e2)
        if S != None:
            Sobol = S
        else:
            Sobol = {}
        for i in range(n_subset):
            key = tuple(subsets_list[i])
            sensitivity.compute_Sobol(Sobol, quotient_variances, key, self.labels)

        all_labels = list(Sobol.keys())
        si_all = list(Sobol.values())

        # plotting
        si_all = np.array(si_all)
        order = np.argsort(-si_all)
        n_selected = min(40, len(si_all))
        selected = order[:n_selected] # taking the top 40 values to plot
        y_pos = np.arange(n_selected)
        if create_plot:
            plt.figure(figsize =(12,12))
            # Create bars
            plt.barh(y_pos, si_all[selected])
            new_labels = [all_labels[selected[i]] for i in range(n_selected)]
            title = 'top_sobol_indices'
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

        # get list of gpu devices to parallelize computation if possible
        local_device_protos = device_lib.list_local_devices()
        devices_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if len(devices_list) == 0:
            devices_list = ['/cpu:0']

        loc_samples = self.hyperpar_samples['gp_constant_mean_function']
        varm_samples = self.hyperpar_samples['kernel_variance']
        beta_samples = self.hyperpar_samples['kernel_inverse_lengthscales']
        hyperpar_samples =   [loc_samples, varm_samples, beta_samples]
        if nx_samples == None:
            nx_samples = 300*self.n_inputs
        selected_vars = [i for i in range(self.n_inputs)]
        selected_vars = [i for i in range(self.n_inputs)]

        ybase = sensitivity.allEffect(self.model, self.Rangenorm, nx_samples, hyperpar_samples, devices_list)
        ey_square = sensitivity.direct_samples(self.model, self.Rangenorm, nx_samples, hyperpar_samples)
        y_remaining  = sensitivity.compute_remaining_effect(self.model, self.Rangenorm, selected_vars, nx_samples, hyperpar_samples, devices_list)

        e1 = np.mean(ey_square[:,1] + np.square(ey_square[:,0]))
        e2 = ybase[1] + np.square(ybase[0])
        si_remaining = np.mean(y_remaining[:,:,1] + np.square(y_remaining[:,:,0]), axis = 1)
        si_remaining  = (si_remaining -e2)/(e1-e2)
        si_remaining = np.maximum(si_remaining,0)
        si_total = 1 - si_remaining
        si_total = np.maximum(si_total,0)

        if create_plot:
            #  generating the plot
            order = np.argsort(-si_total)
            n_selected = min(40, len(si_total))
            selected = order[:n_selected] # taking the top 40 values to plot
            y_pos = np.arange(n_selected)
            plt.figure(figsize =(12,12))
            # Create bars
            plt.barh(y_pos, si_total[selected])
            new_labels = [self.labels[selected[i]] for i in range(n_selected)]
            title = 'top_total_sobol_indices'
            plt.title(title)
            # Create names on the x-axis
            plt.yticks(y_pos, new_labels)
            figpath = title + '.png'
            figpath = os.path.join(directory_path, figpath)
            plt.savefig(figpath)
            plt.close()

        Sobol_total = {}
        for i in range(self.n_inputs):
            l = self.labels[i]
            Sobol_total[l] = si_total[i]

        return Sobol_total
