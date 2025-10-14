#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:05:04 2019
@author: cemma
#Predicitng single current profile from E327 expeirmental data
"""
# 1. Importing data from Lucretia sim
import time
import sys
sys.path.insert(0,'/Users/cemma/Documents/Work/FACET-II/Lucretia_sims/ML_Two_bunch/')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from plotting_functions_twobunch import plot_2bunch_prediction_vs_lucretia
import sklearn.neural_network as nn
from sklearn import preprocessing
import extractDAQBSAScalars
path = "data/raw/TEST/2023/20230901/TEST_03748/"
bsaScalarData, bsaVars = extractDAQBSAScalars.extractDAQBSAScalars(sio.loadmat(path+'TEST_03748.mat', squeeze_me=True, struct_as_record=False)['data_struct'])
lps = sio.loadmat(path+'lpsFlattened_TEST_03748.mat',squeeze_me=True)
# Scandata is in the following columns [L1p,L2p,L1v,L2v,Qi,bc11pkI,bc14pkI,IPpkI,bc11.centroidx,bc14.centroidx]
bsaScalarData = np.transpose(bsaScalarData)
lps = lps['lpsFlattened']; 

def trunc_norm(mu,sigma,ntrunc,nsamples):
    import scipy.stats as stats
    lower, upper = -ntrunc*sigma, ntrunc*sigma
    X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    out = X.rvs(nsamples)
    return out
#%% Add noise in physical units to the predictors before pre-processing data 
# Variables ordering is [L1s phase, L2 phase, L1 amp, L2 amp, BC11 pkI, BC14 pkI, IP pkI, BC11 energy, BC14 energy]
# Noise values are the sigma of the noise you want to add in physical units
nsims=bsaScalarData.shape[0]
# No need to add noise since this is experimental data
#noise_values = [0.0,0.0,0.00,0.00,0.0,0.0,0.0,0.0,0.0]
#X = np.empty([nsims,])
# for i in range(X.shape[1]):
#     #noise = trunc_norm(0,noise_values[i],2,nsims);
#     noise = 0;
#     X[:,i]=bsaScalarData[:,i]+noise;     
X = bsaScalarData
# Now choose a number of random training, validation and test shots
ntrain = int(np.round(nsims*0.8))
ntest = int(np.round(nsims*0.2))
# Randomly index your shots for traning and test sets
idx = np.random.permutation(nsims)
idxtrain = idx[0:ntrain]
idxtest = idx[ntrain:ntrain+ntest]
# Normalize the current
Iz_scaled = lps/np.max(lps) 
Iz_train_scaled = Iz_scaled[idxtrain,:]
Iz_test_scaled = Iz_scaled[idxtest,:]

# Scale the input data between 0 and 1 
X_train_scaled = np.zeros((ntrain,X.shape[1]))
X_test_scaled = np.zeros((ntest,X.shape[1]))
scale_x = preprocessing.MinMaxScaler(feature_range=(0,1))
for i in range(X.shape[1]):
    x1 = X[:,i]
    x2 = x1.reshape(-1,1)
    X_pv = scale_x.fit_transform(x2)
    X_train_scaled[:,i] = X_pv[idxtrain,0]
    X_test_scaled[:,i] = X_pv[idxtest,0]
#%% Initialize and train the NN
# Initialize the neural network model - good for April scan data
# Relu activation function works better than tanh - is it cause the data is more sparse/nonlinear? idk...
nn_model_curprof = nn.MLPRegressor(
    activation = 'relu',
    alpha = 1.0e-4,
    batch_size = 24,
    tol = 1e-5,# default 1e-4
#    hidden_layer_sizes = (500,200,100),
    hidden_layer_sizes = (1000,500,500),
#    hidden_layer_sizes = (1000,500,200,100),#98% accuracy 5e-5 learning rate
    solver = 'adam',
    learning_rate = 'adaptive',# Only for sgd solver
    learning_rate_init = 5.0e-5,
    max_iter = 5000,
    beta_1 = 0.9,beta_2=0.999,# Only for adam solver
    shuffle = True,
    early_stopping = True,
    validation_fraction = 0.2,
    verbose = False,
    momentum = 0.7,# Only used for sgd solver
    warm_start = False,
    random_state = None
)
t0 = time.time()
# Fit the nn model on the training set
nn_model_curprof.fit(X_train_scaled,Iz_train_scaled)
elapsed = time.time() - t0
print("Elapsed time [mins] = 	{:.1f} ".format(elapsed/60))
# Predict on training and validation set
predict_Iz_train = nn_model_curprof.predict(X_train_scaled)
predict_Iz_test = nn_model_curprof.predict(X_test_scaled)
#%% Print results and plot score
print("Score on training set = {0:.3f} ".format(nn_model_curprof.score(X_train_scaled,Iz_train_scaled)*100),"%")
print("Score on test set = {0:.3f}".format(nn_model_curprof.score(X_test_scaled,Iz_test_scaled) * 100),"%")
if nn_model_curprof.solver == 'adam':
    import pandas as pd
    pd.DataFrame(nn_model_curprof.loss_curve_).plot()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

for i in range(50):
   plot_2bunch_prediction_vs_lucretia(Iz_test_scaled,predict_Iz_test,np.max(lps))
#%% Make a histogram of the score
score = np.zeros(Iz_train_scaled.shape[0])
for n in range(Iz_train_scaled.shape[0]):
     trueval = Iz_train_scaled[n,:]
     predval = predict_Iz_train[n,:]
     rmse = ((trueval-predval)**2).sum()
     norm = ((trueval-trueval.mean())**2).sum()
     score[n]= 1-rmse/norm
idx = score>0 & np.isfinite(score);     
plt.hist(score[idx])
plt.xlabel("Score")
#plt.xlim([0,1])
plt.show()
print(np.mean(score[np.isfinite(score)]))    
#%% Save stuff if you want    
import joblib
dirr = 'PredictLPSWithHeaterOn/'
# Save to file in the current working directory
joblib_file = dirr+"nn_model_singleBunch_lps_E327_.pkl"  
joblib.dump(nn_model_curprof, joblib_file)
  # #Save stuff to matlab variables so I can put them into my figure of merit function
sio.savemat(dirr+'lps_test_scaled.mat',mdict={'Iz_test_scaled': Iz_test_scaled*np.max(lps)})
sio.savemat(dirr+'predict_lps_test_scaled.mat',mdict={'predict_Iz_test': predict_Iz_test*np.max(lps)})
sio.savemat(dirr+'lps_test_shots.mat',mdict={'idxtest': idxtest})

#%% Load nn model from file
# joblib_model = joblib.load(joblib_file)

# # Calculate the accuracy and predictions
# score = joblib_model.score(X_test_scaled,Iz_test_scaled)  
# print("Test score: {0:.3f} %".format(100 * score))  


