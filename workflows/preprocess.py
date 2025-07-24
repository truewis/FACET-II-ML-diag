import numpy as np
import h5py
import pandas as pd 
from scipy.io import loadmat
import re
import matplotlib.pyplot as plt
from types import SimpleNamespace
import scipy
import warnings
import os

# import Python functions 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('./Python_Functions')
from Python_Functions.functions import cropProfmonImg, matstruct_to_dict, extractDAQBSAScalars, segment_centroids_and_com, plot2DbunchseparationVsCollimatorAndBLEN

# Sets data location
## make this an argument for the user to input
experiment = 'E338'
runname = '12710'

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

# Loads dataset
dataloc = 'data/' + experiment + '/' + experiment + '_' + runname + '/' + experiment + '_'  +runname + '.mat'
mat = loadmat(dataloc,struct_as_record=False, squeeze_me=True)
data_struct = mat['data_struct']

# Extracts number of steps
stepsAll = data_struct.params.stepsAll
if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
    stepsAll = 1

# calculate xt calibration factor
xtcalibrationfactor = data_struct.metadata.DTOTR2.RESOLUTION*1e-6/streakFromGUI/3e8

xrange = 100 
yrange = xrange
sigma = 5 # degree of Gaussian blur 

xtcavImages = None
horz_proj = []

from scipy.ndimage import median_filter, gaussian_filter

# Extract current profiles and 2D LPS images 
xtcavImages_list = []
horz_proj_list = []

for a in range(len(stepsAll)):
    raw_path = data_struct.images.DTOTR2.loc[a]
    match = re.search(rf'({experiment}_\d+/images/DTOTR2/DTOTR2_data_step\d+\.h5)', raw_path)
    if not match:
        raise ValueError(f"Path format invalid or not matched: {raw_path}")

    DTOTR2datalocation = 'data/'+ experiment + '/' + match.group(0)

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
        horz_proj_idx = np.sum(image_shifted, axis=0)
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

minus_90_idx = np.where((xtcavPhase >= -91) & (xtcavPhase <= -89))[0]
plus_90_idx = np.where((xtcavPhase >= 89) & (xtcavPhase <= 91))[0]

# extract charge scalar to normalize current profile 
isChargePV = [bool(re.search(r'TORO_LI20_2452_TMIT', pv)) for pv in bsaVars]
pvidx = [i for i, val in enumerate(isChargePV) if val]
charge = bsaScalarData[pvidx, :] * 1.6e-19

# Process +90 deg shots 
currentProfile_plus_90 = []

for idx in plus_90_idx: 
    tvar = np.arange(1, len(horz_proj[:,idx]) + 1) * xtcalibrationfactor
    tvar = tvar - np.median(tvar)  # Center around zero
    prefactor = charge[0, idx] / np.trapz(horz_proj[:,idx], tvar)
    currentProfile = 1e-3 * horz_proj[:,idx] * prefactor  # Convert to kA
    currentProfile_plus_90.append(currentProfile)

currentProfile_plus_90 = np.array(currentProfile_plus_90)

# bi-Gaussian fit to determine good shots (large bunch-witness separation)
from scipy.optimize import curve_fit

def bi_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
            A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

amp2 = [] 
for ij in range(len(plus_90_idx)):
    y = currentProfile_plus_90[ij, :]
    x = np.arange(len(y))

    # Initial guess: [A1, mu1, sigma1, A2, mu2, sigma2]
    initial_guess = [np.max(y), 100, 4, np.max(y)*0.2, 50 + ij * 0.05, 4]

    try:
        popt, pcov = curve_fit(bi_gaussian, x, y, p0=initial_guess, maxfev=5000)
    except RuntimeError:
        print(f"Fit failed at index {ij}")
        amp2.append(np.nan)
        continue

    # Extract parameters
    A1, mu1_val, sig1, A2, mu2_val, sig2 = popt
    amp2.append(A2)

# Convert results to arrays
amp2 = np.array(amp2)
good_shots = np.where((amp2 > 0.5) & (amp2 < 100))[0]


driverPeak = np.max(currentProfile_plus_90, axis = 1)
predictor = bsaScalarData[:,plus_90_idx]

# removing outliers 
y = []
x = []
for n in range(driverPeak.shape[0]): 
    if driverPeak[n] > 3 * np.std(driverPeak) + np.average(driverPeak):
        continue
    else: 
        y.append(driverPeak[n])
        x.append([predictor[:,n]])
y = np.array(y)
x = np.concatenate(x, axis = 0 ).T
        

# Calculate correlation coefficient between image data and all bsa scalar PVs
c = []

for n in range(predictor.shape[0]):
    X = predictor[n, :]
    Y = driverPeak
    R = np.corrcoef(X, Y)
    c.append(R[0, 1])

c = np.array(c)
c = c[~np.isnan(c)]
absc = np.abs(c)
idx = np.argsort(absc)

# save preprocessed data for ML training
bsaScalarData = np.array(x)[idx,:][-20:].T # selected best 20 scalar PVs
feature_names = np.array(bsaVars)[idx][-20:]
bsaScalarData = pd.DataFrame(bsaScalarData, columns = feature_names)

bsaScalarData.to_pickle('data/processed/bsaScalarData' + experiment + '_' + runname +'.pkl')
np.save('data/processed/driverPeak' + experiment + '_' + runname + '.npy', y)

