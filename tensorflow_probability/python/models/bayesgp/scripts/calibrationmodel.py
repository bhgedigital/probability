from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#-- fix for tensorflow 2.0 version ---
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import os
import matplotlib.pyplot as plt
import traceback
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
from tensorflow.python.client import device_lib
from tensorflow_probability.python.models.bayesgp.scripts import calibration
from tensorflow_probability.python.models.bayesgp.scripts import sensitivity
import copy
import math


class Calibration_model():

    def __init__(self, sim_inputs_pars, sim_outputs, exp_inputs, exp_outputs, model_info  = None,  kernel_type = 'RBF', noise_level = 1e-3, labels = []):
        # Inputs:
        #   sim_inputs_pars := N x D numpy array of simulation inputs and calibration parameter values.
        #           The first columns must correspond to the input variables and the remaining
        #           columns must correspond to the calibration parameters.
        #   sim_outputs := N-dimensional numpy vector of outputs
        #   exp_inputs := N x D numpy array of experimental input values. The variable columns
        #             must have the same order as the input variable columns of sim_inputs_pars
        #  exp_outputs := N-dimensional numpy vector of outputs
        #   model_info = dictionary containing
        #               1) a dictionary containing the hyperparameter samples.
        #               2) numpy array of samples for the calibration parameters
        #               3) the value of the noise variance
        #               4) the type of kernel used
        #  the dictionary must be generated from a previous run of the model with the same set of inputs and outputs arrays
        # if model_info is not provided, the kernel_type and noise level can be provided or will be set as default
        # kernel_type := string specifying the type of kernel to be used. Options are
		#                'RBF', 'Matern12', 'Matern32', 'Matern52'
        # noise_level := variance of the Gaussian noise for the normalized data
        #   labels:= list containing labels for the input variables and the calibration parameters. A default list is
        #       generated if this is not specified

        # Checking that the Gaussian noise variance is between 0 and 1
        if model_info:
            warnings.warn("Retrieving model info from previous run. The set of inputs and outputs arrays must be the same as the previous run.")
            try:
                self.hyperpar_samples = copy.deepcopy(model_info['hyp_samples'])
                self.par_samples = model_info['par_samples'].copy()
                self.kernel_type = model_info['kernel_type']
                noise_level = model_info['noise_level']
            except Exception as e:
                traceback.print_exc()
                print('Failed to retrieve model info.')
        else:
            self.hyperpar_samples = {}
            self.kernel_type = kernel_type
            noise_level = noise_level

        if (noise_level > 1) or (noise_level < 0):
            raise Exception('Invalid value for the noise_level: ' + str(noise_level) + '. It should be between 0 and 1.')

        if len(sim_inputs_pars.shape) == 1:
            raise Exception('Array sim_inputs_pars of simulator input and calibration parameter must have at least 2 columns')

        if len(exp_inputs.shape) == 1:
            self.n_inputs = 1
            Xexp = exp_inputs[:,None]
        else:
            self.n_inputs = exp_inputs.shape[1]
            Xexp = exp_inputs

        self.n_pars = sim_inputs_pars.shape[1] - self.n_inputs  # number of calibration parameters

        if self.n_pars <= 0:
            raise Exception('Computed number of calibration parameters is less than or equal to 0! Array sim_inputs_pars is supposed to have more columns than array exp_inputs.')


        # normalizing the data
        mean_sim = np.mean(sim_inputs_pars, axis = 0)
        std_sim = np.std(sim_inputs_pars, axis = 0, keepdims = True)
        sim_in_norm = (sim_inputs_pars - mean_sim)/std_sim
        exp_in_norm = (Xexp - mean_sim[:self.n_inputs])/std_sim[:,:self.n_inputs]
        self.scaling_input = [mean_sim, std_sim]

        # Normalizing the outputs
        mean_y = np.mean(sim_outputs)
        std_y = np.std(sim_outputs)
        sim_out_norm = (sim_outputs - mean_y)/std_y

        exp_out_norm = (exp_outputs - mean_y)/std_y

        self.scaling_output = [mean_y, std_y]

        # The normalized calibration parameters will be given uniform distributions. We need to specify the bounds for these distributions.
        lower_bounds = np.min(sim_in_norm[:,self.n_inputs:], axis = 0).tolist()
        upper_bounds = np.max(sim_in_norm[:,self.n_inputs:], axis = 0).tolist()

        # Initialize the model
        self.model = calibration.Calibration(sim_in_norm, sim_out_norm, exp_in_norm, exp_out_norm, lower_bounds, upper_bounds, self.kernel_type, noise_level)

        # Bounds needed for sensitivity analysis
        mins_range = np.min(sim_inputs_pars, axis = 0,  keepdims = True).T
        maxs_range = np.max(sim_inputs_pars, axis = 0,keepdims = True).T
        self.Range = np.concatenate([mins_range, maxs_range], axis = 1)

        mins_range = np.min(sim_in_norm, axis = 0,  keepdims = True).T
        maxs_range = np.max(sim_in_norm, axis = 0,keepdims = True).T
        self.Rangenorm = np.concatenate([mins_range, maxs_range], axis = 1)

        if labels == []:
            self.input_labels = ['x' + str(i) for i in range(self.n_inputs)]
            self.par_labels  = ['p' + str(i) for i in range(self.n_pars)]
            self.labels = self.input_labels + self.par_labels
        elif  (len(labels) != self.n_inputs + self.n_pars) or not(all(isinstance(s, str) for s in labels)):
            raise Exception('Invalid input for labels')
        else:
            self.labels = labels
            self.input_labels = labels[:self.n_inputs]
            self.par_labels = labels[self.n_inputs:]

        return



    def run_mcmc(self, mcmc_samples,num_leapfrog_steps = 3, estimate_noise = False, em_iters = 400, learning_rate = 0.05, warm_up = True, step_size = 0.01, thinning =2):
        # Inputs:
        #   mcmc_samples := number of desired samples for the hyperparameters
        # num_leap_frog_steps = number of leap frog steps for the HMC sampler
        # estimated_noise := Boolean that indicates if the model should estimate a noise variance or keep it fixed (estimation is done wuth EM MCMC)
        # em_iters := number of EM-steps to perform if the noise variance is estimated
        # learning_rate := learning rate for optimizer if the noise variance is estimated
        # warm_up := Assuming the noise is kept fixed (i.e estimate_noise == False ), this Boolean  indicates if an adaptive step size is computed during a "warm up" phase
        # step_size := step size to use for the HMC sampler if warm_up == False
        #       model_info = dictionary containing
        #                      1) dictionary with samples of hyperparameters as well loss function history (if noise is estimated)
        #                      2) numpy array of calibration parameters
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
                    self.par_samples, hyperpar_samples, acceptance_rate = self.model.mcmc(mcmc_samples = mcmc_samples, num_burnin_steps =burn_in,step_size = 0.9*step_size,
                                                                    num_leapfrog_steps = num_leapfrog_steps, initial_state = next_state, thinning = thinning)
                    if acceptance_rate < 0.1:
                        warnings.warn("Acceptance rate was low  (less than 0.1)")
                except Exception as e:
                    traceback.print_exc()
                    print('Sampling failed. Increase the noise level or decrease the step size or the number of leap frog steps if necessary.')
            else:
                try:
                    burn_in  = mcmc_samples
                    print('Sampling in progress.')
                    self.par_samples, hyperpar_samples, acceptance_rate = self.model.mcmc(mcmc_samples = mcmc_samples, num_burnin_steps =burn_in,step_size = 0.9*step_size,
                                                                    num_leapfrog_steps = num_leapfrog_steps, thinning = thinning)
                    if acceptance_rate < 0.1:
                        warnings.warn("Acceptance rate was low  (less than 0.1)")
                except Exception as e:
                    traceback.print_exc()
                    print('Sampling failed. Increase the noise level or decrease the step size or the number of leap frog steps if necessary.')

        else:
            print('Estimating the noise variance using EMMCMC')
            try:
                num_warmup_iters = mcmc_samples//2
                self.par_samples, hyperpar_samples, loss_history,_ = self.model.EM_with_MCMC(num_warmup_iters = num_warmup_iters, em_iters = em_iters,
                                                                    mcmc_samples = mcmc_samples, num_leapfrog_steps = num_leapfrog_steps, learning_rate = learning_rate)
            except Exception as e:
                traceback.print_exc()
        loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples = hyperpar_samples
        self.hyperpar_samples['sim_kernel_variance'] = varsim_samples
        self.hyperpar_samples['disc_kernel_variance'] = vard_samples
        self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales'] = betasx_samples
        self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales'] = betaspar_samples
        self.hyperpar_samples['disc_kernel_inverse_lengthscales'] = betad_samples
        self.hyperpar_samples['sim_gp_constant_mean_function'] = loc_samples

        model_info = {}
        model_info['hyp_samples'] = copy.deepcopy(self.hyperpar_samples)
        model_info['par_samples'] = self.par_samples.copy()
        model_info['kernel_type'] = self.kernel_type
        model_info['noise_level'] = self.model.noise
        return model_info

    def plot_chains(self, directory_path1 = None, directory_path2 = None):
        # Function used to plot the chains from the  mcmc sampling and scatter plot
        # for the calibration parameters
        # Inputs:
        #   directory_path1:= directory where to save the mcmc samples plot. It defaults to the current directory if not specified
        #   directory_path2:= directory where to save the scatter plot. It defaults to the current directory if not specified
        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')

        # plotting the mcmc chains
        nplots = 2*self.n_inputs + 2*self.n_pars + 3
        fig, axes = plt.subplots(nplots, 1, figsize=(20, 2.0*nplots),sharex=True)

        # 1) Plotting and saving the chains for the inverse lengthscale
        betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales']
        mcmc_samples = len(betasx_samples)
        t = np.arange(mcmc_samples)
        k = -1
        for i in range(self.n_inputs):
            k = k+1
            axes[k].plot(t,betasx_samples[:,i])
            title =  self.input_labels[i] + 'sim_inverse_lengthscale_samples'
            axes[k].set_title(title)

        betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales']
        for i in range(self.n_pars):
            k = k+1
            axes[k].plot(t,betaspar_samples[:,i])
            title =  self.par_labels[i] + 'sim_inverse_lengthscale_samples'
            axes[k].set_title(title)

        betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales']
        for i in range(self.n_inputs):
            k = k+1
            axes[k].plot(t,betad_samples[:,i])
            title =  self.input_labels[i] + 'disc_inverse_lengthscale_samples'
            axes[k].set_title(title)

        #  2) Plotting and saving the chains for the variances
        varsim_samples = self.hyperpar_samples['sim_kernel_variance']
        k = k+1
        axes[k].plot(t,varsim_samples)
        title = 'sim_kernel_variance_samples'
        axes[k].set_title(title)

        vard_samples = self.hyperpar_samples['disc_kernel_variance']
        k = k+1
        axes[k].plot(t,vard_samples)
        title = 'disc_kern_variance_samples'
        axes[k].set_title(title)

        #  3) Plotting and saving the chain for the simulator mean function
        loc_samples = self.hyperpar_samples['sim_gp_constant_mean_function']

        k = k+1
        axes[k].plot(t,loc_samples)
        title = 'sim_constant_mean_function_samples'
        axes[k].set_title(title)

        # 4) Plotting and saving the chains for the calibration parameters
        # We first need to convert to the proper scale

        mean_sim, std_sim = self.scaling_input

        par_samples_right_scale = self.par_samples*std_sim[0,self.n_inputs:] + mean_sim[self.n_inputs:]
        for i in range(self.n_pars):
            k = k+1
            axes[k].plot(t,par_samples_right_scale[:,i])
            title = self.par_labels[i] + '_samples'
            axes[k].set_title(title)

        if directory_path1 == None:
            directory_path1 = os.getcwd()
        if not(os.path.isdir(directory_path1)):
            raise Exception('Invalid directory path ', directory_path1)
        figpath ='mcmc_chains.png'
        figpath = os.path.join(directory_path1, figpath)
        plt.savefig(figpath)
        plt.close()

        # 4) scatter plot for the calibration parameters
        if directory_path2 == None:
            directory_path2 = os.getcwd()
        if not(os.path.isdir(directory_path2)):
            raise Exception('Invalid directory path ', directory_path2)
        df = pd.DataFrame(par_samples_right_scale, columns = self.par_labels)
        figpath = 'par_scatter.png'
        figpath = os.path.join(directory_path2, figpath)
        plt.figure(figsize = (12,12))
        pd.plotting.scatter_matrix(df)
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


    def plot_local_sensitivity(self, directory_path = None):
        # Function used to plot the local sensitivty  boxplot

        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')

        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)

        betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales']
        betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales']
        betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales']
        beta_samples_list = [betasx_samples, betaspar_samples, betad_samples]
        figpath = 'local_sensitivity.png'
        figpath = os.path.join(directory_path, figpath)
        sensitivity.generateBetaBoxPlots(bounds=self.Rangenorm, beta_samples_list=beta_samples_list, labels=self.labels, figpath = figpath, calibration_type=True)
        return

    def predict(self, Xtest, with_point_samples = False):
        # Computes approximate values of the full posterior mean and variance of the Gaussian process
		# by using samples of the posterior distribution of the hyperparameters
		#Inputs:
		#	Xtest :=  N x D input array
		#	with_point_samples = Boolean specifying if we sample from the posterior Gaussian
		#                       for each input and for each sample of the hyperparameters
		# Outputs:
		#	main_results, sim_results, disc_results := lists containing prediction results for
        #                the full model, simulation model and discrepancy respectively
        #                Each list consist of an array containing the mean values, an array with
        #               the standard deviation values and if with_point_samples == True, the list
        #               also contains an array of ouput samples
        mean_sim, std_sim = self.scaling_input
        mean_x = mean_sim[:self.n_inputs]
        std_x = std_sim[:,:self.n_inputs]
        mean_y, std_y = self.scaling_output

        if len(Xtest.shape) == 1:
            Xnew = Xtest[:, None]
        else:
            Xnew = Xtest

        Xtest_norm = (Xnew - mean_x)/std_x

        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')

        mcmc_samples = len(self.hyperpar_samples['sim_kernel_variance'])
        # Limiting the number of mcmc samples used if necessary
        if mcmc_samples > 3000:
            selected = np.random.permutation(mcmc_samples)
            varsim_samples = self.hyperpar_samples['sim_kernel_variance'][selected]
            vard_samples = self.hyperpar_samples['disc_kernel_variance'][selected]
            betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales'][selected]
            betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales'][selected]
            betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales'][selected]
            loc_samples = self.hyperpar_samples['sim_gp_constant_mean_function'][selected]
            par_samples = self.par_samples[selected]

        else:
            varsim_samples = self.hyperpar_samples['sim_kernel_variance']
            vard_samples = self.hyperpar_samples['disc_kernel_variance']
            betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales']
            betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales']
            betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales']
            loc_samples = self.hyperpar_samples['sim_gp_constant_mean_function']
            par_samples = self.par_samples

        hyperpar_samples = [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples]

        if with_point_samples:
            mean_pos, var_pos, samples, mean_posim, var_posim, samplesim, mean_poserr, var_poserr, sampleserr = self.model.samples_withpar(Xtest_norm, hyperpar_samples,
                                                                                                            par_samples, num_samples = 30, with_point_samples = True)
            std_pos = np.sqrt(var_pos)
            std_posim = np.sqrt(var_posim)
            std_poserr = np.sqrt(var_poserr)
            # Converting to the proper scale
            mean_pos = mean_pos*std_y + mean_y
            mean_posim = mean_posim*std_y + mean_y
            mean_poserr= mean_poserr*std_y
            std_pos = std_pos*std_y
            std_posim = std_posim*std_y
            std_poserr = std_poserr*std_y
            samples = samples*std_y + mean_y
            samplesim = samplesim*std_y + mean_y
            sampleserr = sampleserr*std_y

            main_results = [mean_pos, std_pos, samples ]
            sim_results = [mean_posim, std_posim, samplesim]
            disc_results = [mean_poserr, std_poserr, sampleserr]

            return main_results, sim_results, disc_results

        else:
            mean_pos, var_pos, mean_posim, var_posim, mean_poserr, var_poserr = self.model.samples_withpar(Xtest_norm, hyperpar_samples,
                                                                                                            par_samples, num_samples = 30, with_point_samples = False)

            std_pos = np.sqrt(var_pos)
            std_posim = np.sqrt(var_posim)
            std_poserr = np.sqrt(var_poserr)
            # Converting to the proper scale
            mean_pos = mean_pos*std_y + mean_y
            mean_posim = mean_posim*std_y + mean_y
            mean_poserr= mean_poserr*std_y
            std_pos = std_pos*std_y
            std_posim = std_posim*std_y
            std_poserr = std_poserr*std_y

            main_results = [mean_pos, std_pos]
            sim_results = [mean_posim, std_posim]
            disc_results = [mean_poserr, std_poserr]

            return main_results, sim_results, disc_results

    # Sensitivity analysis
    def maineffect_and_interaction(self, type = 'simulator', grid_points = 30, nx_samples = None, directory_path1 = None, directory_path2 = None, create_plot = True, batch_size=10):
        # Computes  and generate main_effect function plots
        # Inputs:
        #   type := string that specifies for which gaussian process we are performing the sensitivity analysis.
        #           allowed values = 'simulator', 'discrepancy'
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


        if len(self.hyperpar_samples) == 0:
            raise Exception('Execute run_mcmc first.')
        if directory_path1 == None:
            directory_path1 = os.getcwd()
        if not(os.path.isdir(directory_path1)):
            raise Exception('Invalid directory path ', directory_path1)
        if directory_path2 == None:
            directory_path2 = os.getcwd()
        if not(os.path.isdir(directory_path2)):
            raise Exception('Invalid directory path ', directory_path2)

        mean_sim, std_sim = self.scaling_input
        mean_y, std_y = self.scaling_output

        if type == 'simulator':
            used_labels = self.labels
            used_Range = self.Range
            used_Rangenorm = self.Rangenorm
            n_vars = self.n_inputs + self.n_pars
        elif type == 'discrepancy':
            used_labels = self.input_labels
            used_Range = self.Range[:self.n_inputs,:]
            used_Rangenorm = self.Rangenorm [:self.n_inputs,:]
            n_vars = self.n_inputs
        else:
            raise Exception('Invalid type')

        if n_vars  == 1:
            print('Not enough variables to perform sensitivity analysis.')
            return
        varsim_samples = self.hyperpar_samples['sim_kernel_variance']
        vard_samples = self.hyperpar_samples['disc_kernel_variance']
        betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales']
        betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales']
        betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales']
        loc_samples = self.hyperpar_samples['sim_gp_constant_mean_function']
        par_samples = self.par_samples

        hyperpar_samples = [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples]
        # Main effect
        if nx_samples == None:
            nx_samples = 300*n_vars
        selected_vars = [i for i in range(n_vars)]
        ybase = sensitivity.allEffect(self.model, used_Rangenorm, nx_samples, hyperpar_samples, par_samples, type)

        if n_vars <= batch_size:
            y_main = sensitivity.mainEffect(self.model, used_Rangenorm, selected_vars, nx_samples, hyperpar_samples, grid_points, par_samples, type)
        else:
            y_main = {}
            n_batches = n_vars//batch_size
            vars_groups = np.array_split(selected_vars,n_batches)
            completed = 0
            for group in var_groups:
                y_group = sensitivity.mainEffect(self.model, used_Rangenorm, group, nx_samples, hyperpar_samples, grid_points, par_samples, type)
                completed += len(group)
                progress = 100.0*completed/n_vars
                print("Main effect computation: {:.2f}% complete".format(progress))
                y_main.update(y_group)

        z_mean = np.zeros((n_vars, grid_points))
        z_std = np.zeros((n_vars, grid_points))
        for i in range(n_vars):
            key = tuple([i])
            z_mean[i,:] = y_main[key][:,0] - ybase[0][0,0]
            # The next 3 lines give an approximation of the standard deviation of the normalized main effect function E[Y|xi] - E[Y]
            lower_app = np.sqrt(np.abs(np.sqrt(y_main[key][:,1]) - np.sqrt(ybase[0][0,1])))
            upper_app = np.sqrt(y_main[key][:,1]) + np.sqrt(ybase[0][0,1])
            z_std[i,:] = (lower_app + upper_app)/2.0

        # Converting to the proper scale and storing
        main = {}
        for i in range(n_vars):
            y = z_mean[i,:]*std_y + mean_y
            y_std = z_std[i,:]*std_y
            x = np.linspace(used_Range[i,0], used_Range[i,1], grid_points)
            key = used_labels[i]
            main[key] = {}
            main[key]['inputs'] = x
            main[key]['output_mean']= y
            main[key]['output_std']= y_std
        if create_plot:
            print('Generating main effect plots.')
            if n_vars <= 6:
                fig, axes = plt.subplots(nrows=1, ncols=n_vars, sharey=True, figsize =(20,10))
                for i in range(n_vars):
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
            else:
                plot_rows = math.ceil(self.n_inputs/6)
                fig, axes = plt.subplots(nrows=plot_rows, ncols=6, sharey=True, figsize =(20,15))
                for i in range(n_vars):
                    row_idx = i//6
                    col_idx = i%6
                    key = self.labels[i]
                    x = main[key]['inputs']
                    y = main[key]['output_mean']
                    y_std = main[key]['output_std']
                    axes[row_idx, col_idx].plot(x,y, label= self.labels[i])
                    axes[row_idx, col_idx].fill_between(x, y-2*y_std, y + 2*y_std, alpha = 0.2, color ='orange')
                    axes[row_idx, col_idx].grid()
                    axes[row_idx, col_idx].legend()
                title = 'main_effects'
                plt.title(title)
                figpath = title + '.png'
                figpath = os.path.join(directory_path1, figpath)
                plt.savefig(figpath)
                plt.close(fig)

        #---------------------------------------------------------------------
        # Interaction effect
        print("Starting interaction computations.")
        selected_pairs = []
        for i in range(n_vars-1):
            for j in range(i+1,n_vars):
                selected_pairs.append([i,j])
        selected_pairs = np.array(selected_pairs)
        n_pairs = len(selected_pairs)
        if n_pairs <= batch_size:
            y_int = sensitivity.mainInteraction(self.model, used_Rangenorm, selected_pairs, nx_samples, hyperpar_samples, grid_points, par_samples, type)
        else:
            y_int = {}
            n_batches = n_pairs//batch_size
            pairs_groups = np.array_split(selected_pairs,n_batches,axis=0)
            completed = 0
            for group in pairs_groups:
                y_group = sensitivity.mainInteraction(self.model, used_Rangenorm, group, nx_samples, hyperpar_samples, grid_points, par_samples, type)
                completed += len(group)
                progress = 100.0*completed/n_pairs
                print("Interaction effect computation: {:.2f}% complete".format(progress))
                y_int.update(y_group)
        z_intmean = np.zeros((n_pairs, grid_points, grid_points))
        z_intstd = np.zeros((n_pairs, grid_points, grid_points))
        for k in  range(n_pairs):
            key = tuple(selected_pairs[k])
            y_slice = np.reshape(y_int[key],(grid_points,grid_points,2))
            j1, j2 = selected_pairs[k]
            key1 = tuple([j1])
            key2 = tuple([j2])
            v1 = y_main[key1][:,0]
            v2 = y_main[key2][:,0]
            p1, p2 = np.meshgrid(v1,v2)
            w1 = np.sqrt(y_main[key1][:,1])
            w2 = np.sqrt(y_main[key2][:,1])
            q1, q2 = np.meshgrid(w1,w2)
            z_intmean[k,:,:] = y_slice[:,:,0] - p1 - p2 +  ybase[0][0,0]
            upper_app = np.sqrt(y_slice[:,:,1]) + q1 + q2 + np.sqrt(ybase[0][0,1])
            lower_app = np.abs(np.sqrt(y_slice[:,:,1]) - q1 - q2 + np.sqrt(ybase[0][0,1]))
            z_intstd[k,:,:] = (upper_app + lower_app)/2.0

        # Converting to the proper scale and storing
        interaction = {}
        for k in range(n_pairs):
            item = selected_pairs[k]
            j1, j2 = item
            x = np.linspace(used_Range[j1,0],used_Range[j1,1],grid_points)
            y = np.linspace(used_Range[j2,0],used_Range[j2,1],grid_points)
            Z = z_intmean[k,:,:]*std_y + mean_y
            Zstd = z_intstd[k,:,:]*std_y
            X,  Y = np.meshgrid(x,y)
            key = used_labels[j1] + '_&_' + used_labels[j2]
            X,  Y = np.meshgrid(x,y)
            interaction[key] = {}
            interaction[key]['input1'] = X
            interaction[key]['input2'] = Y
            interaction[key]['output_mean'] = Z
            interaction[key]['output_std'] = Zstd

        if create_plot:
            print('Generating interaction surfaces plots.')
            # Bounds for the interaction surface plot
            zmin = np.min(z_intmean)*std_y + mean_y
            zmax = np.max(z_intmean)*std_y + mean_y
            minn = np.min(z_intstd)*std_y
            maxx = np.max(z_intstd)*std_y

            for k in range(n_pairs):
                item = selected_pairs[k]
                j1, j2 = item
                key = used_labels[j1] + '_&_' + used_labels[j2]
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
                ax.set_xlabel(used_labels[j2])
                ax.set_ylabel(used_labels[j1])
                ax.set_zlim(zmin,zmax)
                plt.gca().invert_xaxis()
                plt.colorbar(m)
                figpath = title + '.png'
                figpath = os.path.join(directory_path2, figpath)
                plt.savefig(figpath)
                plt.close(fig)


        return main, interaction

    def sobol_indices(self, type = 'simulator', max_order = 2, S = None, nx_samples = None, directory_path = None, create_plot = True, batch_size=10):
        # Computes sobol indices and generate bar plot.
        # Inputs:
        #   Sobol_store := dictionary containing previously computed Sobol indices. The computation of
        #      Sobol indices is a recursive computation
        #   max_order := maximum order of sobol indices to compute
        #   nx_samples = the number of sample points for the Monte Carlo integration. Will default
        #           to a multiple of the number of variables if not provided
        #  directory_path :=  directory where to save the Sobol barplot if needed. Defaults to current directory
        #       if not specified
        #    create_plot := specifies if the Sobol barplot should be generated are not
        # Outputs:
        #   Sobol := dictionary containining containing the  Sobol indices values

        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')
        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)


        mean_sim, std_sim = self.scaling_input
        mean_y, std_y = self.scaling_output

        if type == 'simulator':
            used_labels = self.labels
            used_Range = self.Range
            used_Rangenorm = self.Rangenorm
            n_vars = self.n_inputs + self.n_pars
        elif type == 'discrepancy':
            used_labels = self.input_labels
            used_Range = self.Range[:self.n_inputs,:]
            used_Rangenorm = self.Rangenorm [:self.n_inputs,:]
            n_vars = self.n_inputs
        else:
            raise Exception('Invalid type')

        if n_vars  == 1:
            print('Not enough variables to perform sensitivity analysis.')
            return
        if max_order >  n_vars:
            raise Exception('max_order cannot be greater than the number of variables', n_vars)

        varsim_samples = self.hyperpar_samples['sim_kernel_variance']
        vard_samples = self.hyperpar_samples['disc_kernel_variance']
        betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales']
        betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales']
        betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales']
        loc_samples = self.hyperpar_samples['sim_gp_constant_mean_function']
        par_samples = self.par_samples

        hyperpar_samples = [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples]

        if nx_samples == None:
            nx_samples = 300*n_vars
        selected_vars = [i for i in range(n_vars)]

        initial_list  = sensitivity.powerset(selected_vars,1, max_order)
        subsets_list = []
        if S != None:
            print('Initial number of Sobol computations: ', len(initial_list))
            try:
                for item in initial_list:
                    l = sensitivity.generate_label(item, used_labels)
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
            ybase = sensitivity.allEffect(self.model, used_Rangenorm, nx_samples, hyperpar_samples, par_samples, type)
            ey_square = sensitivity.direct_samples(self.model, used_Rangenorm, nx_samples, hyperpar_samples, par_samples, type)
            if n_subset <= batch_size:
                y_higher_order = sensitivity.mainHigherOrder(self.model, used_Rangenorm, subsets_list, nx_samples, hyperpar_samples, par_samples, type)
            else:
                y_higher_order = {}
                completed = 0
                n_groups = math.ceil(n_subset/batch_size)
                for i in range(n_groups):
                    group = subsets_list[i*batch_size:(i+1)*batch_size]
                    y_group = sensitivity.mainHigherOrder(self.model, used_Rangenorm, group, nx_samples, hyperpar_samples, par_samples, type)
                    completed += len(group)
                    progress = 100.0*completed/n_subset
                    print("Sobol indices computation: {:.2f}% complete".format(progress))
                    y_higher_order.update(y_group)

            e1 = np.mean(ey_square[:,1] + np.square(ey_square[:,0]))
            e2 = ybase[0][0,1] + np.square(ybase[0][0,0])
            # This will store the quantities E*[Vsub]/E*(Var(Y)) where Vsub = E[Y|Xsub] and Y is normalized
            quotient_variances = {}

            for idx in range(n_subset):
                k = tuple(subsets_list[idx])
                quotient_variances[k] = np.mean(y_higher_order[k][:,1] + np.square(y_higher_order[k][:,0]))
                quotient_variances[k] = (quotient_variances[k] - e2)/(e1-e2)
        if S != None:
            Sobol = S
        else:
            Sobol = {}
        for i in range(n_subset):
            key = tuple(subsets_list[i])
            sensitivity.compute_Sobol(Sobol, quotient_variances, key, used_labels)

        all_labels = list(Sobol.keys())
        si_all = list(Sobol.values())

        # plotting
        if create_plot:
            print('Generating Sobol indices barplot.')
            si_all = np.array(si_all)
            order = np.argsort(-si_all)
            n_selected = min(40, len(si_all))
            selected = order[:n_selected] # taking the top 40 values to plot
            y_pos = np.arange(n_selected)
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

    def total_sobol_indices(self, type = 'simulator', nx_samples = None, directory_path = None, create_plot = True, batch_size=10):
        # Computes total sobol indices and generate bar plot.
        # Inputs:
        #   nx_samples = the number of sample points for the Monte Carlo integration. Will default
        #           to a multiple of the number of variables if not provided
        #  directory_path :=  directory where to save the Sobol barplot if needed. Defaults to current directory
        #       if not specified
        #    create_plot := specifies if the Sobol barplot should be generated are not
        # Outputs:
        #   Sobol_total := dictionary containining containing the total Sobol indices values


        if len(self.hyperpar_samples) == 0:
            raise Exception('Hyperparameter samples must be generated or retrieved first.')
        if directory_path == None:
            directory_path = os.getcwd()
        if not(os.path.isdir(directory_path)):
            raise Exception('Invalid directory path ', directory_path)


        mean_sim, std_sim = self.scaling_input
        mean_y, std_y = self.scaling_output

        if type == 'simulator':
            used_labels = self.labels
            used_Range = self.Range
            used_Rangenorm = self.Rangenorm
            n_vars = self.n_inputs + self.n_pars
        elif type == 'discrepancy':
            used_labels = self.input_labels
            used_Range = self.Range[:self.n_inputs,:]
            used_Rangenorm = self.Rangenorm [:self.n_inputs,:]
            n_vars = self.n_inputs
        else:
            raise Exception('Invalid type')

        if n_vars  == 1:
            print('Not enough variables to perform sensitivity analysis.')
            return

        varsim_samples = self.hyperpar_samples['sim_kernel_variance']
        vard_samples = self.hyperpar_samples['disc_kernel_variance']
        betasx_samples = self.hyperpar_samples['sim_inputs_kernel_inverse_lengthscales']
        betaspar_samples = self.hyperpar_samples['sim_pars_kernel_inverse_lengthscales']
        betad_samples = self.hyperpar_samples['disc_kernel_inverse_lengthscales']
        loc_samples = self.hyperpar_samples['sim_gp_constant_mean_function']
        par_samples = self.par_samples

        hyperpar_samples = [loc_samples, varsim_samples, betaspar_samples, betasx_samples, betad_samples, vard_samples]

        if nx_samples == None:
            nx_samples = 300*n_vars
        selected_vars = [i for i in range(n_vars)]

        ybase = sensitivity.allEffect(self.model, used_Rangenorm, nx_samples, hyperpar_samples, par_samples, type)
        ey_square = sensitivity.direct_samples(self.model, used_Rangenorm, nx_samples, hyperpar_samples, par_samples, type)
        # y_remaining  = sensitivity.compute_remaining_effect(self.model, used_Rangenorm, selected_vars, nx_samples, hyperpar_samples, devices_list, par_samples, type)

        if n_vars  <= batch_size:
            y_remaining  = sensitivity.compute_remaining_effect(self.model, used_Rangenorm, selected_vars, nx_samples, hyperpar_samples,60, par_samples, type)
        else:
            y_remaining = {}
            n_batches = n_vars//batch_size
            vars_groups = np.array_split(selected_vars,n_batches)
            completed = 0
            for group in vars_groups:
                y_group = sensitivity.compute_remaining_effect(self.model, used_Rangenorm, group, nx_samples, hyperpar_samples,60,par_samples, type)
                completed += len(group)
                progress = 100.0*completed/n_vars
                print("Total Sobol indices computation: {:.2f}% complete".format(progress))
                y_remaining.update(y_group)

        e1 = np.mean(ey_square[:,1] + np.square(ey_square[:,0]))
        e2 = ybase[0][0,1] + np.square(ybase[0][0,0])
        si_remaining  = np.zeros(n_vars)
        for i in range(n_vars):
            key = tuple([i])
            si_remaining[i] = np.mean(y_remaining[key][:,1] + np.square(y_remaining[key][:,0]))
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
            new_labels = [used_labels[selected[i]] for i in range(n_selected)]
            title = 'top_total_sobol_indices'
            plt.title(title)
            # Create names on the x-axis
            plt.yticks(y_pos, new_labels)
            figpath = title + '.png'
            figpath = os.path.join(directory_path, figpath)
            plt.savefig(figpath)
            plt.close()

        Sobol_total = {}
        for i in range(n_vars):
            l = used_labels[i]
            Sobol_total[l] = si_total[i]

        return Sobol_total
