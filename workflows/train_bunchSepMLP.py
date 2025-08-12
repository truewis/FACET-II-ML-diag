import numpy as np
import h5py
from scipy.io import loadmat
import pandas as pd
import re
import matplotlib.pyplot as plt
from types import SimpleNamespace
import scipy
import warnings
from scipy.ndimage import median_filter, gaussian_filter
import os

# import Python functions 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('./Python_Functions')
from Python_Functions.functions import cropProfmonImg, matstruct_to_dict, extractDAQBSAScalars, segment_centroids_and_com, plot2DbunchseparationVsCollimatorAndBLEN

# Select Experiment and runname
while True:
    answer = input("Please provide experiment and runname (ex: E338, 12710) ").strip()
    if re.match('^E[0-9]+, [0-9]+', answer) is None:
        print("Invalid response, please try again")
        continue
    else: 
        experiment = re.match('^E[0-9]+', answer)[0]
        runname = re.search('[0-9]+$', answer)[0]
        try: 
            dataloc = 'data/raw/' + experiment + '/' + experiment + '_' + runname + '/' + experiment + '_'  +runname + '.mat'
            mat = loadmat(dataloc,struct_as_record=False, squeeze_me=True)
            data_struct = mat['data_struct']
            print('Experiment loaded successfully.')
            break
        except FileNotFoundError: 
            print("Error: The specified data was not found in '/data/raw/'.")
            continue


# Define XTCAV calibration
krf = 239.26
cal = 1167 # um/deg  http://physics-elog.slac.stanford.edu/facetelog/show.jsp?dir=/2025/11/13.03&pos=2025-$
streakFromGUI = cal*krf*180/np.pi*1e-6#um/um

# Sets the main beam energy
mainbeamE_eV = 10e9
# Sets the dnom value for CHER
dnom = 59.8e-3

# Sets the calibration value for SYAG in eV/m
SYAG_cal = 64.4e9

# Extracts number of steps
stepsAll = data_struct.params.stepsAll
if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
    stepsAll = 1

# calculate xt calibration factor
xtcalibrationfactor = data_struct.metadata.DTOTR2.RESOLUTION*1e-6/streakFromGUI/3e8

# cropping aspect ration 
xrange = 100 
yrange = xrange

# degree of Gaussian blur 
sigma = 5 

# Extract current profiles and 2D LPS images 
xtcavImages_list = []
horz_proj_list = []

for a in range(len(stepsAll)):
    raw_path = data_struct.images.DTOTR2.loc[a]
    match = re.search(rf'({experiment}_\d+/images/DTOTR2/DTOTR2_data_step\d+\.h5)', raw_path)
    if not match:
        raise ValueError(f"Path format invalid or not matched: {raw_path}")

    DTOTR2datalocation = 'data/raw/'+ experiment + '/' + match.group(0)

    with h5py.File(DTOTR2datalocation, 'r') as f:
        data_raw = f['entry']['data']['data'][:].astype(np.float64)  # shape: (N, H, W)
    
    # Transpose to shape: (H, W, N)
    DTOTR2data_step = np.transpose(data_raw, (2, 1, 0))
    xtcavImages_step = DTOTR2data_step - data_struct.backgrounds.DTOTR2[:,:,np.newaxis].astype(np.float64)
    
    for idx in range(DTOTR2data_step.shape[2]):
        if idx is None:
            continue
        image = xtcavImages_step[:,:,idx]
        
        # crop images 
        image_cropped, _ = cropProfmonImg(image, xrange, yrange, plot_flag=False)
        Nrows = np.array(image_cropped).shape[0]
        img = median_filter(image_cropped, size=3)
        processed_image = gaussian_filter(img, sigma=sigma)
        [centroidIndices, centers_of_mass] = segment_centroids_and_com(processed_image, Nrows,1)

        # centroid correcion
        centroid_corrections = np.round((centers_of_mass / np.abs(centers_of_mass)) * np.abs(centers_of_mass) - centers_of_mass.shape[0] / 2)
        centroid_corrections[np.isnan(centroid_corrections)] = 0

        # shift images
        image_shifted = np.empty_like(image_cropped)
        for row in range(image_cropped.shape[0]):
            shift = int(-centroid_corrections[row])
            image_shifted[row] = np.roll(image_cropped[row], shift)
        # calcualte current profiles 
        horz_proj_idx = np.sum(image_cropped, axis=0)
        horz_proj_idx = horz_proj_idx[:,np.newaxis]
        image_shifted = image_shifted[:, :, np.newaxis]
        
        # combine current profiles into one array 
        horz_proj_list.append(horz_proj_idx)

        # combine images into one array 
        xtcavImages_list.append(image_shifted)

xtcavImages = np.concatenate(xtcavImages_list, axis=2)
horz_proj = np.concatenate(horz_proj_list, axis=1)

# Keeps only the data with a common index
DTOTR2commonind = data_struct.images.DTOTR2.common_index -1 
horz_proj = horz_proj[:,DTOTR2commonind]
xtcavImages = xtcavImages[:,:,DTOTR2commonind]

# extract scalars (predictors) 
bsaScalarData, bsaVars = extractDAQBSAScalars(data_struct)

ampl_idx = next(i for i, var in enumerate(bsaVars) if 'TCAV_LI20_2400_A' in var)
xtcavAmpl = bsaScalarData[ampl_idx, :]

phase_idx = next(i for i, var in enumerate(bsaVars) if 'TCAV_LI20_2400_P' in var)
xtcavPhase = bsaScalarData[phase_idx, :]

xtcavOffShots = xtcavAmpl<0.1
xtcavPhase[xtcavOffShots] = 0 # Set this for ease of plotting

# extract current profiles
isChargePV = [bool(re.search(r'TORO_LI20_2452_TMIT', pv)) for pv in bsaVars]
pvidx = [i for i, val in enumerate(isChargePV) if val]
charge = bsaScalarData[pvidx, :] * 1.6e-19  # in C 

minus_90_idx = np.where((xtcavPhase >= -91) & (xtcavPhase <= -89))[0]
plus_90_idx = np.where((xtcavPhase >= 89) & (xtcavPhase <= 91))[0]
all_idx = np.append(minus_90_idx,plus_90_idx)

currentProfile_all = [] 

# Process all streaked shots
for ij in range(len(all_idx)):
    idx = all_idx[ij]
    streakedProfile = horz_proj[:,idx]

    tvar = np.arange(1, len(streakedProfile) + 1) * xtcalibrationfactor
    tvar = tvar - np.median(tvar)  # Center around zero

    prefactor = charge[0, idx] / np.trapz(streakedProfile, tvar)

    currentProfile = 1e-3 * streakedProfile * prefactor  # Convert to kA
    currentProfile_all.append(currentProfile)
currentProfile_all = np.array(currentProfile_all)

separationCutoff = 0.05

# calculate bunch separation using Gaussian Mixture Model (GMM)
from sklearn.mixture import GaussianMixture
mean1 = []
mean2 = []
amp2 = []

for ij in range(len(all_idx)):
    y = currentProfile_all[ij, :]
    x = np.arange(len(y))
    cp = np.column_stack((x,y))
    if all_idx[ij] < 0.5*np.max(all_idx):
        gm = GaussianMixture(n_components=6, random_state=0).fit(cp)
    
        m1 = gm.means_[0][0]
        m2 = gm.means_[1][0]
        mean1.append(m1)
        mean2.append(m2)
        amp2.append(currentProfile_all[ij,int(m2)])
    else:
        gm = GaussianMixture(n_components=4, random_state=0).fit(cp) 
        
        m1 = gm.means_[0][0]
        m2 = gm.means_[3][0]
        mean1.append(m1)
        mean2.append(m2)
        amp2.append(currentProfile_all[ij,int(m2)-10])

mean1 = np.array(mean1)
mean2 = np.array(mean2)
amp2 = np.array(amp2 )

# define good shots where clear bunch separation
goodShots = np.where(amp2 > 0)[0]
bunchSeparation_all_fit = (mean1[goodShots] - mean2[goodShots]) * xtcalibrationfactor * 1e6 * 3e8 # in microns 
steps = data_struct.scalars.steps[DTOTR2commonind]

#################################################################################
# MODEL PREPARATION 
#################################################################################

# order parameters by correlation 
c = []

for n in range(bsaScalarData.shape[0]):
    X = np.array(bsaScalarData)[:,goodShots][n,:]
    Y = bunchSeparation_all_fit
    R = np.corrcoef(X, Y)
    c.append(R[0, 1])

c = np.array(c)
c = c[~np.isnan(c)]
absc = np.abs(c)
idx = np.argsort(absc)

# N-best parameter fit (bsaScalar PVs + step number)
N = 20 
X = np.vstack((bsaScalarData[idx[-N:],:][:,all_idx[goodShots]],steps[all_idx[goodShots]])).T
bunchSep = bunchSeparation_all_fit


# save data to files 
answer = input("Do you want to save preprocessed data? (y/n): ").strip().lower() 
 
# Check the response 
for _ in range(1):
    if answer == 'y': 
        print("Saving data to '/data/processed/'....")
        np.save('data/processed/predictors' + experiment + '_' + runname +'.npy', X)
        np.save('data/processed/bunchSep' + experiment + '_' + runname + '.npy', bunchSep)
    elif answer == 'n': 
        print('Data will not be saved.')
        break
    else: 
        print("Please answer with 'y' or 'n'.") 

print('Training model...')

nsims=X.shape[0]

# 80/20 train-test split
ntrain = int(np.round(nsims*0.8))
ntest = int(np.round(nsims*0.2))

idx = np.random.permutation(nsims)
idxtrain = idx[0:ntrain]
idxtest = idx[ntrain:ntrain+ntest]

# scale feature data for improved model accuracy 
bs = bunchSep
bs_train = bs[idxtrain]
bs_test = bs[idxtest] 

# scale feature data for improved model accuracy
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
x_train_scaled = x_scaled[idxtrain]
x_test_scaled = x_scaled[idxtest]

#################################################################################
# MODEL TRAINING 
#################################################################################

import sklearn.neural_network as nn
import time

nn_model_bunchsep = nn.MLPRegressor(
    activation = 'relu',
    alpha = 1.0e-4,
    batch_size = 24,
    tol = 1e-4,# default 1e-4
#    hidden_layer_sizes = (500,200,100),
#    hidden_layer_sizes = (1000,500,500),
    hidden_layer_sizes = (500, 200,100),#98% accuracy 5e-5 learning rate
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
nn_model_bunchsep.fit(x_train_scaled,bs_train)
elapsed = time.time() - t0
print("Elapsed time [mins] = {:.1f} ".format(elapsed/60))
# Predict on training and validation set
predict_bs_train = nn_model_bunchsep.predict(x_train_scaled)
predict_bs_test = nn_model_bunchsep.predict(x_test_scaled)
#%% Print results and plot score
print("Train R²: {0:.2f} ".format(nn_model_bunchsep.score(x_train_scaled,bs_train)*100),"%")
print("Test R²: {0:.2f}".format(nn_model_bunchsep.score(x_test_scaled,bs_test) * 100),"%")

# save model 
answer = input("Do you want to save model? (y/n): ").strip().lower() 
 
# Check the response 
if answer == 'y': 
    import joblib
    joblib_file = 'model/MLP_bunchSep_'+experiment+'_'+runname+'.pkl'    
    joblib.dump(nn_model_bunchsep, joblib_file)
elif answer == 'n': 
    exit() 
else: 
    print("Please answer with 'y' or 'n'.") 
