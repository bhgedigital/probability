import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import tensorflow_probability as tfp
tfm = tfp.models

#-------- test 1---------------------------------------------------------


data = pd.read_csv('./test_Data/training_data.csv').values

Xtr = data[:,:-1]
Ytr = data[:,-1]

# initialize
bgp = tfm.BGP_model(Xtr, Ytr, 'RBF', 1e-3)



# perform mcmc sampling
bgp.run_mcmc(5000,num_leapfrog_steps = 3, estimate_noise = True)

results_path = './test_results/test1/'

bgp.plot_loss_function(display = False, save = True, path = results_path)

bgp.plot_chains(display = False, save = True, path = results_path)

# get predictions on training data
mean_pos, std_pos, samples = bgp.predict(Xtr, with_point_samples = True)

# Computing percentiles
lower = np.percentile(samples,2.5, axis = 0)
upper = np.percentile(samples,97.5, axis = 0)


# generating predicted vs actual plot
# Predicted vs actual
plt.figure(figsize =(10,10))
plt.plot(Ytr, Ytr, color = 'red', label ='actual')
plt.scatter(Ytr, mean_pos, color = 'blue', label = 'predicted')
plt.vlines(Ytr, lower, upper, color = 'green', label = '95 % confidence region')
plt.legend()
figpath = 'predicted_vs_actual.png'
figpath = results_path + figpath
plt.savefig(figpath)
plt.grid()
plt.close()


#--------------------------------------------------------------------------------
#------------------------- test 2 -----------------------------------------------

data = pd.read_csv('./test_Data/training_data2.csv').values

Xtr = data[:,:-1]
Ytr = data[:,-1]

# initialize
bgp = tfm.BGP_model(Xtr, Ytr, 'Matern32', 1e-2)



# perform mcmc sampling
bgp.run_mcmc(5000,num_leapfrog_steps = 3, estimate_noise = False, warm_up = False, step_size = 0.30)

results_path = './test_results/test2/'

# plot the samples from the mcmc sampling
bgp.plot_chains(labels =['u1','u2','u3','u4','u5'], display = False, save = True,  path = results_path)

# get predictions on training data
mean_pos, std_pos, samples = bgp.predict(Xtr, with_point_samples = True)

# Computing percentiles
lower = np.percentile(samples,2.5, axis = 0)
upper = np.percentile(samples,97.5, axis = 0)


# generating predicted vs actual plot
# Predicted vs actual
plt.figure(figsize =(10,10))
plt.plot(Ytr, Ytr, color = 'red', label ='actual')
plt.scatter(Ytr, mean_pos, color = 'blue', label = 'predicted')
plt.vlines(Ytr, lower, upper, color = 'green', label = '95 % confidence region')
plt.legend()
figpath = 'predicted_vs_actual.png'
figpath = results_path + figpath
plt.savefig(figpath)
plt.grid()
plt.close()


#-----------------------------------------------------------------------------
#------------------- test 3 --------------------------------------------

data = pd.read_csv('./test_Data/training_data2.csv').values


Xtr = data[:,:-1]
Ytr = data[:,-1]

# initialize
bgp = tfm.BGP_model(Xtr, Ytr,'Matern52', 1e-3)



# perform mcmc sampling
bgp.run_mcmc(5000,  num_leapfrog_steps = 3, estimate_noise = False, warm_up = True)

results_path = './test_results/test3/'

# plot the samples from the mcmc sampling
bgp.plot_chains(display = False, save = True, path = results_path)

# get predictions on training data
mean_pos, std_pos = bgp.predict(Xtr)

