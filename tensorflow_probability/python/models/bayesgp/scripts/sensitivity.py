from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from itertools import chain, combinations
import functools
import multiprocessing as mp
import time
#-------------------------------------------------------------------------------
#---- Functions for performing sensitivity analysis ----------------------------

#-------------------------------------------------------------------------------


#------------ References -------------------------------------------------------
# - [Rasmussen] "Gaussian process for Machine Learning", C. E. Rasmussen and C. K. I. Williams, MIT press
# - [Farah]   "Bayesian Inference for Sensitivity Analysis of Computer Simulators, with an Application to
#   Radiative Transfer Models",
# - [Janon] " Asynptotoc normality and efficiency of two Sobol index estimators",  A. Janon,
# T. Klein, A. Lagnoux-Renaudie, M. Nodet and C. Prieur, https://hal.inria.fr/hal-00665048v2
# - [Higdon]   "Combining Field Data and Computer Simulation for Calibration and Prediction",
#
#-------------------------------------------------------------------------------

def affineTransform(X, b, a):
    return X*a + b

def powerset(S,min_size, max_size):
    p_set = list(chain.from_iterable(combinations(S,n) for n in range(min_size,max_size+1)))
    out = []
    for item in p_set:
        y = list(item)
        y.sort()
        out.append(y)
    return out

def uniform_to_triangular(u):
    # function to convert a standard uniform random variable to a random variable
    # with a triangular distribution
    if u < 0.5:
        return (2*u)**0.5 -1.0
    else:
        return 1.0 - (2*(1-u))**0.5

# vectorizing the function
vuniform_to_triangular = np.vectorize(uniform_to_triangular)

def allEffect(model, bounds, nx_samples, hyperpar_samples, *args):
    # Computes mean and variance of the Gp E[Y] which is obtained from integrating
    # out all the variables
    # The integration is done using Monte Carlo integration
    n_var = bounds.shape[0]
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]
    diff_samples = lhs(n_var, samples = 4*nx_samples)
    diff_samples = vuniform_to_triangular(diff_samples)
    M = lhs(n_var, samples = nx_samples)
    sampling_dict = {}
    sampling_dict[0] = {}
    sampling_dict[0]['fixed_indices_list'] = []
    sampling_dict[0]['diff_samples'] = diff_samples*a
    sampling_dict[0]['X_sampling'] = np.zeros((1, nx_samples, n_var))
    sampling_dict[0]['X_sampling'][0,:,:] = affineTransform(M, b, a)
    if len(args): # calibration case
        par_samples = args[0]
        type = args[1]
        ybase = model.expected_predict_posterior(sampling_dict, hyperpar_samples, par_samples, type)
    else:
        ybase = model.expected_predict_posterior(sampling_dict, hyperpar_samples)
    return ybase

def mainEffect(model, bounds, selected_vars, nx_samples, hyperpar_samples,n_points = 30, *args):
    # Computes mean and variance of the Gp E[Y | Xp]
    # The integration is done using Monte Carlo integration
    n_var = bounds.shape[0]
    n_selected = len(selected_vars)
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]
    points = np.linspace(0,1,n_points)
    M = lhs(n_var-1, samples = nx_samples)
    diff_samples = lhs(n_var-1, 4*nx_samples)
    if diff_samples.shape[1] > 0:
        diff_samples = vuniform_to_triangular(diff_samples)
    sampling_dict = {}
    vars = [i for i in range(n_var)]
    for idx in range(n_selected):
        j = selected_vars[idx]
        key  = tuple([j])
        sampling_dict[key] ={}
        sampling_dict[key]['fixed_indices_list'] = [j]
        sampling_dict[key]['X_sampling'] = np.zeros((n_points,nx_samples, n_var))
        vars_left = vars[:]
        vars_left.remove(j)
        range_slice = a[:,vars_left].copy()
        sampling_dict[key]['diff_samples'] = diff_samples*range_slice
        for k in range(n_points):
            v = points[k]
            N = np.insert(M, j, v*np.ones(nx_samples), axis = 1)
            sampling_dict[key]['X_sampling'][k,:,:] = affineTransform(N, b, a)
    if len(args): # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(sampling_dict, hyperpar_samples, par_samples, type)
    else:
        y= model.expected_predict_posterior(sampling_dict, hyperpar_samples)
    return y

def mainInteraction(model, bounds, selected_pairs, nx_samples, hyperpar_samples, n_points = 30, *args):
    # Computes mean and variance of the Gp E[Y | Xpq]
    # using Monte Carlo integration
    n_var = bounds.shape[0]
    n_pairs = len(selected_pairs)
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]
    M = lhs(n_var, samples = nx_samples)
    diff_samples = lhs(n_var-2, 4*nx_samples)
    if diff_samples.shape[1] > 0:
        diff_samples = vuniform_to_triangular(diff_samples)
    points = np.linspace(0,1,n_points)
    p1, p2 = np.meshgrid(points,points)
    q1 = p1.ravel()
    q2 = p2.ravel()
    sampling_dict = {}
    vars = set([i for i in range(n_var)])
    M = lhs(n_var-2, samples = nx_samples)
    # Computing samples for E(Y|xi xj)
    for idx in range(n_pairs):
        j1, j2 = selected_pairs[idx]
        key = tuple([j1,j2])
        sampling_dict[key] ={}
        sampling_dict[key]['fixed_indices_list'] = [j1, j2]
        sampling_dict[key]['X_sampling'] = np.zeros((n_points**2,nx_samples, n_var))
        vars_left = list(vars - set([j1,j2]))
        vars_left.sort()
        range_slice = a[:,vars_left].copy()
        sampling_dict[key]['diff_samples'] = diff_samples*range_slice
        for k in range(n_points**2):
            v1 = q1[k]
            v2 = q2[k]
            N = np.insert(M, j1, v1*np.ones(nx_samples), axis = 1)
            N = np.insert(N, j2, v2*np.ones(nx_samples), axis = 1)
            sampling_dict[key]['X_sampling'][k,:,:] = affineTransform(N, b, a)
    if len(args): # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(sampling_dict, hyperpar_samples, par_samples, type)
    else:
        y= model.expected_predict_posterior(sampling_dict, hyperpar_samples)
    return y



def mainHigherOrder(model, bounds, subsets_list, nx_samples, hyperpar_samples, *args):
    # Computes mean and variance of the Gp E[Y|Xsub] where Xsub denotes a subset
    # of the variables that are kept fixed
    n_var = bounds.shape[0]
    n_subset = len(subsets_list)
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]
    a_aug = a[np.newaxis, :]
    b_aug = b[np.newaxis, :]
    sampling_dict ={}
    vars = set([i for i in range(n_var)])
    for idx in range(n_subset):
        subset = subsets_list[idx]
        key = tuple(subset)
        size_subset = len(subset)
        size_left = n_var - size_subset
        M_fixed = lhs(size_subset, samples = nx_samples//2)
        M_varying = lhs(size_left, samples = nx_samples)
        M_varying = np.tile(M_varying[np.newaxis,:,:],[nx_samples//2,1,1])
        for jdx in range(size_subset):
            j = subset[jdx]
            points = M_fixed[:,jdx]
            M_varying = np.insert(M_varying,j, points[:, None], axis = -1)
        sampling_dict[key] ={}
        sampling_dict[key]['fixed_indices_list'] = subset[:]
        sampling_dict[key]['X_sampling'] = affineTransform(M_varying,b_aug, a_aug)
        vars_left = list(vars - set(subset))
        diff_samples = lhs(len(vars_left), 4*nx_samples)
        if diff_samples.shape[1] > 0:
            diff_samples = vuniform_to_triangular(diff_samples)
        range_slice = a[:,vars_left].copy()
        sampling_dict[key]['diff_samples'] = diff_samples*range_slice

    if len(args): # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(sampling_dict, hyperpar_samples, par_samples, type)
    else:
        y= model.expected_predict_posterior(sampling_dict, hyperpar_samples)
    return y




def direct_samples(model, bounds, nx_samples, hyperpar_samples,*args):
    # computes samples for the Gaussian process directly.
    n_var = bounds.shape[0]
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]

    M = lhs(n_var, samples = nx_samples)
    Xtest = affineTransform(M, b, a)
    if len(args)> 0: # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.predict_posterior(Xtest, hyperpar_samples, par_samples, type)
    else:
        y = model.predict_posterior(Xtest, hyperpar_samples)
    return y


def compute_remaining_effect(model, bounds, selected_vars, nx_samples, hyperpar_samples,n_points = 60, *args):
    # Computes mean and variance of the Gp E[Y | X_p]
    # using Monte Carlo integration
    n_var = bounds.shape[0]
    n_selected = len(selected_vars)
    a = bounds[:,1] -bounds[:,0]
    a_aug = a[np.newaxis, np.newaxis, :]
    b = bounds[:,0]
    b_aug = b[np.newaxis, np.newaxis, :]
    M = lhs(n_var-1, samples = nx_samples)
    M = np.tile(M[:,np.newaxis,:],[1,n_points,1])
    points = np.linspace(0,1,n_points)
    diff_samples = lhs(1, 4*nx_samples)
    diff_samples = vuniform_to_triangular(diff_samples)
    sampling_dict = {}
    vars = [i for i in range(n_var)]

    for idx in range(n_selected):
        j = selected_vars[idx]
        key  = tuple([j])
        sampling_dict[key] ={}
        vars_left = vars[:]
        vars_left.remove(j)
        sampling_dict[key]['fixed_indices_list'] = vars_left
        Xm = np.insert(M,j,points[None, :], axis = -1)
        # Perform the affine transformation
        sampling_dict[key]['X_sampling']  = affineTransform(Xm,b_aug, a_aug)
        sampling_dict[key]['diff_samples'] = diff_samples*a[j]
    if len(args)> 0: # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(sampling_dict, hyperpar_samples, par_samples, type)
    else:
        y = model.expected_predict_posterior(sampling_dict, hyperpar_samples)
    return y





def generateBetaBoxPlots(bounds, beta_samples_list, labels, figpath = None, calibration_type = False):
	# Generate Box plots for a metric defined in terms of the inverse lengthscale
    if calibration_type:
        betasx_samples, betaspar_samples, betad_samples = beta_samples_list
    	# For the simulator
        Range = (bounds[:,1] - bounds[:,0])[:,None]
        n_inputs = betasx_samples.shape[1]
        n_pars = betaspar_samples.shape[1]
        metric_x = 1 - np.exp(-np.square(betasx_samples*(Range[:n_inputs,0]))/8.0)
        metric_par = 1 - np.exp(-np.square(betaspar_samples*(Range[n_inputs:,0]))/8.0)
        data_to_plot = []
        for i in range(n_inputs):
            data_to_plot.append(metric_x[:,i])
        for i in range(n_pars):
    	    data_to_plot.append(metric_par[:,i])
        plt.figure(figsize=(20, 10))
        plt.subplot(2,1,1)
        # Create the boxplot
        plt.boxplot(data_to_plot,  showfliers=False)
        locs, _ = plt.xticks()
        plt.xticks(locs, labels)
        plt.title('Simulator model')

        # Fo the inadequacy
        metric_x = 1 - np.exp(-np.square(betad_samples*(Range[:n_inputs,0]))/8.0)
        data_to_plot = []
        for i in range(n_inputs):
            data_to_plot.append(metric_x[:,i])
        plt.subplot(2,1,2)
        # Create the boxplot
        plt.boxplot(data_to_plot,  showfliers=False)

        plt.xticks(locs, labels[:n_inputs])
        plt.title('Inadequacy model')
        if figpath:
            plt.savefig(figpath)
            plt.close()
    else:
        beta_samples = beta_samples_list[0]
        Range = (bounds[:,1] - bounds[:,0])[:,None]
        n_inputs = beta_samples.shape[1]
        metric_x = 1 - np.exp(-np.square(beta_samples*(Range[:n_inputs,0]))/8.0)
        data_to_plot = []
        for i in range(n_inputs):
            data_to_plot.append(metric_x[:,i])

        plt.figure(figsize=(20, 10))
        plt.subplot(1,1,1)
        # Create the boxplot
        plt.boxplot(data_to_plot,  showfliers=False)
        locs, _ = plt.xticks()
        plt.xticks(locs, labels)

        if figpath:
            plt.savefig(figpath)
            plt.close()
    return

def generate_label(subset, labels):
    # function used to generate labels for sobol indices
    if len(subset) > 1:
        target = [labels[i] for i in subset]
        return " || ".join(target)
    else:
        return labels[subset[0]]

def compute_Sobol(S,Q,key, labels):
    # function used to compute the sobol indices recursively given a dictionary
    # Q containing the values for the quotient variances
    l = generate_label(list(key), labels)
    S[l] = Q[key]
    if len(key) > 1:
        subsets = powerset(list(key),1,len(key)-1)
        for item in subsets:
            l_item = generate_label(item, labels)
            S[l] -= S[l_item]
    S[l] = max(S[l],0)
    return
