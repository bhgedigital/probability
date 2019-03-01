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

def allEffect(model, bounds, nx_samples, hyperpar_samples, devices_list, *args):
    # Computes mean and variance of the Gp E[Y] which is obtained from integrating
    # out all the variables
    # The integration is done using Monte Carlo integration
    n_var = bounds.shape[0]
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]

    M = lhs(n_var, samples = nx_samples)
    Mp = lhs(2*n_var, samples = nx_samples)
    Mp1 = Mp[:, :n_var].copy()
    Mp2 = Mp[:,n_var:].copy()
    Mt= affineTransform(M, b, a)
    Mpt1 = affineTransform(Mp1,b,a)
    Mpt2 = affineTransform(Mp2,b,a)
    Xtest = np.zeros((1,1, nx_samples, 3*n_var))
    Xtest[0,0,:,:] = np.concatenate([Mt, Mpt1, Mpt2], axis = 1)
    if len(args): # calibration case
        par_samples = args[0]
        type = args[1]
        ybase = model.expected_predict_posterior(Xtest, hyperpar_samples, par_samples, devices_list, type)[0,0,:]
    else:
        ybase = model.expected_predict_posterior(Xtest, hyperpar_samples,devices_list)[0,0,:]
    return ybase

def mainEffect(model, bounds, selected_vars, nx_samples, hyperpar_samples,devices_list, n_points = 30, *args):
    # Computes mean and variance of the Gp E[Y | Xp]
    # The integration is done using Monte Carlo integration
    n_var = bounds.shape[0]
    n_selected = len(selected_vars)
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]

    points = np.linspace(0,1,n_points)
    Xtest = np.zeros((n_points, n_selected, nx_samples, 3*n_var))
    M = lhs(n_var-1, samples = nx_samples)
    Mp = lhs(2*(n_var-1), samples = nx_samples)
    for idx in range(n_var):
        j = selected_vars[idx]
        for k in range(n_points):
            v = points[k]
            N = M.copy()
            N = np.insert(N, j, v*np.ones(nx_samples), axis = 1)
            Nt = affineTransform(N, b, a)
            Np1 = Mp[:, :n_var-1].copy()
            Np1 = np.insert(Np1, j, v*np.ones(nx_samples), axis = 1)
            Npt1 = affineTransform(Np1, b, a)
            Np2 = Mp[:,n_var-1:].copy()
            Np2 = np.insert(Np2, j, v*np.ones(nx_samples), axis = 1)
            Npt2 = affineTransform(Np2, b, a)
            Xtest[k,idx,:,:] = np.concatenate([Nt, Npt1, Npt2], axis = 1)
    if len(args)> 0: # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, par_samples, devices_list, type)
    else:
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, devices_list)

    y = np.swapaxes(y,0,1)

    return y

def mainInteraction(model, bounds, selected_pairs, nx_samples, hyperpar_samples, devices_list, n_points = 30, *args):
    # Computes mean and variance of the Gp E[Y | Xpq]
    # using Monte Carlo integration

    n_var = bounds.shape[0]
    n_pairs = len(selected_pairs)
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]
    M = lhs(n_var, samples = nx_samples)
    Mp = lhs(2*n_var, samples = nx_samples)
    Mp1 = Mp[:, :n_var].copy()
    Mp2 = Mp[:,n_var:].copy()
    points = np.linspace(0,1,n_points)
    p1, p2 = np.meshgrid(points,points)
    q1 = p1.ravel()
    q2 = p2.ravel()
    Xtest = np.zeros((n_points**2, n_pairs, nx_samples, 3*n_var))
    M = lhs(n_var-2, samples = nx_samples)
    Mp = lhs(2*(n_var-2), samples = nx_samples)
    Mp1 = Mp[:, :n_var-2].copy()
    Mp2 = Mp[:,n_var-2:].copy()
    # Computing samples for E(Y|xi xj)
    for idx in range(n_pairs):
        j1, j2 = selected_pairs[idx]
        for k in range(n_points**2):
            v1 = q1[k]
            v2 = q2[k]
            N = M.copy()
            N = np.insert(N, j1, v1*np.ones(nx_samples), axis = 1)
            N = np.insert(N, j2, v2*np.ones(nx_samples), axis = 1)
            Nt = affineTransform(N, b, a)
            Np1 = Mp1.copy()
            Np1 = np.insert(Np1, j1, v1*np.ones(nx_samples), axis = 1)
            Np1 = np.insert(Np1, j2, v2*np.ones(nx_samples), axis = 1)
            Npt1 = affineTransform(Np1, b, a)
            Np2 = Mp2.copy()
            Np2 = np.insert(Np2, j1, v1*np.ones(nx_samples), axis = 1)
            Np2 = np.insert(Np2, j2, v2*np.ones(nx_samples), axis = 1)
            Npt2 = affineTransform(Np2, b, a)
            Xtest[k,idx,:,:] = np.concatenate([Nt, Npt1, Npt2], axis = 1)
    if len(args)> 0: # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, par_samples, devices_list, type)
    else:
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, devices_list)
    y = np.swapaxes(y, 0,1)

    return y

def sampling_Matrix(L, values):
    subset, b, a, n_var, nx_samples = L
    size_subset  = len(subset)
    size_left = n_var - size_subset
    size_left = n_var - size_subset
    N = lhs(size_left, samples = nx_samples)
    Np = lhs(2*size_left, samples = nx_samples)
    Np1 = Np[:, :size_left]
    Np2 = Np[:,size_left:]
    for jdx in range(size_subset):
        j = subset[jdx]
        value = values[jdx]
        N = np.insert(N, j, value*np.ones(nx_samples), axis = 1)
        Np1 = np.insert(Np1, j, value*np.ones(nx_samples), axis = 1)
        Np2 = np.insert(Np2, j, value*np.ones(nx_samples), axis = 1)
    Nt = affineTransform(N, b, a)
    Npt1 = affineTransform(Np1, b, a)
    Npt2 = affineTransform(Np2, b, a)
    out = np.concatenate([Nt, Npt1, Npt2], axis = 1)
    return out[np.newaxis,:,:]



def mainHigherOrder(model, bounds, subsets_list, nx_samples, hyperpar_samples, devices_list, *args):
    # Computes mean and variance of the Gp E[Y|Xsub] where Xsub denotes a subset
    # of the variables that are kept fixed
    p = mp.Pool(processes = 10)
    n_var = bounds.shape[0]
    n_subset = len(subsets_list)
    a = bounds[:,1] -bounds[:,0]
    a = a[None,:]
    b = bounds[:,0][None,:]
    Xtest = np.zeros((nx_samples//3, n_subset, nx_samples, 3*n_var))
    for idx in range(n_subset):
        subset = subsets_list[idx]
        size_subset = len(subset)
        L = [subset, b, a, n_var, nx_samples]
        f = functools.partial(sampling_Matrix,L)
        Sampling = lhs(size_subset, samples = nx_samples//3)
        results = [p.apply(f,args=(x,)) for x in Sampling]
        Xtest[:,idx,:,:] = np.concatenate(results, axis=0)
    print('Sampling matrix generated')
    if len(args)> 0: # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, par_samples, devices_list, type)
    else:
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, devices_list)

    y = np.swapaxes(y, 0,1)
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


def compute_remaining_effect(model, bounds, selected_vars, nx_samples, hyperpar_samples, devices_list, *args):
    # Computes mean and variance of the Gp E[Y | X_p]
    # using Monte Carlo integration
    n_var = bounds.shape[0]
    n_selected = len(selected_vars)
    a = bounds[:,1] -bounds[:,0]
    a = a[np.newaxis, np.newaxis, :]
    b = bounds[:,0]
    b = b[np.newaxis, np.newaxis, :]
    M = lhs(n_var-1, samples = nx_samples)
    n_points = nx_samples//3
    M = np.tile(M[:,np.newaxis,:],[1,n_points,1])
    points = np.linspace(0,1,n_points)
    pairs  = lhs(2, samples = n_points)
    Xtest  = np.zeros((nx_samples,n_selected,n_points, 3*n_var))

    for idx in range(n_selected):
        i = selected_vars[idx]
        Xm = np.insert(M,i,points[None, :], axis = -1)
        Xp1 = np.insert(M,i,pairs[:,0][None, :], axis = -1)
        Xp2 = np.insert(M,i,pairs[:,1][None, :], axis = -1)
        # Perform the affine trnsformation
        Xm = Xm*a + b
        Xp1 = Xp1*a + b
        Xp2 = Xp2*a + b
        Xtest[:,idx,:,:] = np.concatenate([Xm, Xp1, Xp2], axis = -1)
    if len(args)> 0: # calibration case
        par_samples = args[0]
        type = args[1]
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, par_samples, devices_list, type)
    else:
        y = model.expected_predict_posterior(Xtest, hyperpar_samples, devices_list)
    y = np.swapaxes(y, 0,1)
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
