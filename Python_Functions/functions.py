import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy
import warnings
from scipy.io.matlab.mio5_params import mat_struct
from scipy.ndimage import median_filter, gaussian_filter
import re
import h5py
from tqdm import tqdm # Import tqdm
import torch.nn as nn

# Define MLP structure
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, out_dim)
        )
    def forward(self, x):
        return self.model(x)

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import find_objects
from typing import Tuple, Dict


# ----------------- 1D Gaussian Fitting Models -----------------

def gaussian_1d(x, mu, sigma, amplitude):
    """Single 1D Gaussian PDF for fitting slices/projections."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)

def bigaussian_1d(x, mu1, sigma1, pi1, mu2, sigma2, pi2):
    """Sum of two 1D Gaussians for vertical projection fitting."""
    return (
        gaussian_1d(x, mu1, sigma1, pi1) + 
        gaussian_1d(x, mu2, sigma2, pi2)
    )

# ----------------- 2D Gaussian Model (for Covariance Calculation) -----------------

def gaussian_2d_pdf(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Calculates the 2D Gaussian PDF evaluated at coordinates.
    Args:
        mu (np.ndarray): Mean vector of shape (2,).
        Sigma (np.ndarray): Covariance matrix of shape (2, 2).

    """
    xrange = 100
    yrange = 100
    
    # Check for singularity (shouldn't happen with robust fitting but for safety)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return np.zeros((2*yrange, 2*xrange))

    # Coordinates centered relative to mu
    xy_coords = np.array(np.meshgrid(np.arange(0, 2* xrange), np.arange(0, 2*yrange))).T.reshape(-1, 2)  # (N, 2)

    print("Shape of xy_coords:", xy_coords.shape)
    centered_coords = (xy_coords - mu)
    
   
    # Step A: V = D @ Sigma_inv  (Shape: (N, 2) @ (2, 2) -> (N, 2))
    V = centered_coords @ Sigma_inv 
    
    # Step B: Sum(V * D, axis=1) (Element-wise product, then sum along D=2 axis -> (N,))
    # This correctly computes the quadratic form for all N points in parallel.
    quad_form = np.sum(V * centered_coords, axis=1) # (N,)
    
    # PDF formula: 1 / ( (2*pi)^D * |Sigma| )^0.5 * exp(-0.5 * quad_form)
    # D=2 for 2D
    numerator = np.exp(-0.5 * quad_form)

    return numerator

# ----------------- Main Conversion Function -----------------

def image_to_bigaussian_params(target_image: np.ndarray) -> Dict:
    # Constants for a 200x200 image normalized to a range of [-1, 1]
    IMG_SIZE = target_image.shape[0]  # Assuming square image
    K = 2 # Fixed number of components
    """
    Analyzes a 2D target density map to extract K=2 GMM parameters 
    using sequential 1D Gaussian fitting based on structural guarantees.
    Args:
        target_image (np.ndarray): 2D array of shape (200, 200) representing the target density map.
    """
    
    # 0. Preprocessing: Normalize image mass (required for projection fitting)
    target_density = target_image.copy()
    total_mass = target_density.sum()
    if total_mass > 0:
        target_density /= total_mass

    # Convert target density logarithmically
    target_density = np.log1p(target_density * 1e3)  # log(1 + density * scale)

    y_coords = np.arange(IMG_SIZE)
    x_coords = np.arange(IMG_SIZE)

    # --- Step 1: Vertical Projection and Y-Mean/Weight Estimation ---

    vertical_projection = target_density.max(axis=1) # Max across X-axis (200,)

    # TODO: Improve initial guess based on peak finding
    # Plot the whole image for debugging
    plt.figure()
    plt.imshow(target_density, cmap='viridis', aspect='auto')
    plt.title('Target Density Map')
    plt.colorbar(label='Density')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    try:
        peaks, properties = find_peaks(vertical_projection, distance=15, height=np.max(vertical_projection)*0.1, width = 5)

        # Extract fitted parameters
        mu_y = np.array([y_coords[peaks[0]], y_coords[peaks[1]]])
        pi_y_raw = np.array([properties['peak_heights'][0], properties['peak_heights'][1]])
        sigma_y = np.array([properties['widths'][0]/2, properties['widths'][1]/2]) # Ensure positive stddev

        
        print("Shape of vertical_projection:", vertical_projection.shape)
        plt.figure()
        plt.plot(y_coords, vertical_projection, label='Vertical Projection', color='blue')
        plt.plot(y_coords, bigaussian_1d(y_coords, mu_y[0], sigma_y[0], pi_y_raw[0], mu_y[1], sigma_y[1], pi_y_raw[1]), label='Fitted Bi-Gaussian', color='red')
        plt.legend()
        plt.title('Vertical Projection and Fitted Bi-Gaussian')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Density')
        plt.show()


    except RuntimeError:
        print("Warning: 1D vertical fit failed, throwing error.")
        raise RuntimeError("1D vertical Gaussian fitting failed.")
        
    # --- Step 2 & 3: Horizontal Slice and X-Mean Estimation ---

    means = []
    sigma_x = []

    # Find the nearest row indices corresponding to the fitted mu_y locations
    y_indices = np.round(mu_y).astype(int)
    
    for k in range(K):
        y_k = mu_y[k]
        
        # Extract the horizontal slice at the determined row index
        slice_k = target_density[y_indices[k], :]
        
        # Robust initial guess for 1D horizontal fit
        x_peak_idx = np.argmax(slice_k)
        p0_x = [x_coords[x_peak_idx], 5, slice_k[x_peak_idx]]
        print (f"Component {k+1}: Fitting horizontal slice at y={y_k} (row index {y_indices[k]}) with initial guess {p0_x}")
        plt.figure()
        plt.plot(x_coords, slice_k, label='Horizontal Slice', color='blue')
        plt.legend()
        plt.title('Horizontal Slice and Fitted Gaussian')
        plt.xlabel('X Coordinate')
        plt.ylabel('Density')
        plt.show()
        try:
            # Fit 1D Gaussian to the horizontal slice to find x_k
            popt_x, _ = curve_fit(gaussian_1d, x_coords, slice_k, p0=p0_x, maxfev=5000)
            mu_x_k = popt_x[0]
            sigma_x_k = np.abs(popt_x[1])  # Not used further but could be stored if needed
        except RuntimeError:
            print(f"Warning: 1D horizontal fit for component {k+1} failed.")
            raise RuntimeError(f"1D horizontal Gaussian fitting failed for component {k+1}.")

        means.append(np.array([mu_x_k, y_k]))
        sigma_x.append(sigma_x_k)
    print("Fitted Means (Mu):", means)
    # --- Step 4: Full 2D Parameter Estimation (Fixed Means & Weights) ---

    # Covariance Matrix (Sigma_k) calculation uses the analytical M-step equivalent
    # for a fixed assignment, which is the robust way to find Sigma given fixed Mu and Pi.

    # This is not fitting any function but calculating the weighted scatter matrix.
    
    sigmas = []
    
    X, Y = np.meshgrid(x_coords, y_coords)
    coordinates = np.stack([X.ravel(), Y.ravel()], axis=1) # (N, 2) where N=IMG_SIZE*IMG_SIZE
    flat_density = target_density.ravel() # (N,)
    
    # 1. Calculate Mahalanobis-like Distance (incorporating estimated sigma_y)
    distances = np.zeros((coordinates.shape[0], K))
    
    for k in range(K):
        mu_k = means[k]
        sigma_yk = sigma_y[k] 
        sigma_xk = sigma_x[k]

        diff = coordinates - mu_k # (N, 2)
        
        # Use a distance weighted by the estimated vertical variance (sigma_x, sigma_y)
        # This is a heuristic to better assign points based on the known horizontal and vertical spread.
        # It's a simplified Mahalanobis distance assuming a diagonal covariance 
        # (a diagonal matrix with a large constant for sigma_x^2 and sigma_yk^2) 
        # For simplicity, we'll use Euclidean distance but weight the y-dimension 
        # by the inverse of the variance, a common simplification:
        
        # D_i,k^2 = (x_i - mu_x,k)^2 + (y_i - mu_y,k)^2 / sigma_y,k^2
        dist_sq_x_weighted = diff[:, 0]**2 / (sigma_xk**2 + 1e-6) # (x_i - mu_x,k)^2 / sigma_x,k^2
        # Use a small constant floor for sigma_yk to prevent division by zero/overscaling
        dist_sq_y_weighted = diff[:, 1]**2 / (sigma_yk**2 + 1e-6) # (y_i - mu_y,k)^2 / sigma_y,k^2

        distances[:, k] = np.sqrt(dist_sq_x_weighted + dist_sq_y_weighted)

    # 2. Hard Assignment
    assigned_component = np.argmin(distances, axis=1) # (N,) indices 0 or 1
    # If distances are greater than a threshold, we could assign to no component, which is -1.
    dist_threshold = 4 # This threshold can be tuned based on expected spread
    assigned_component[distances.min(axis=1) > dist_threshold] = -1

    # Plot the assignment for debugging
    plt.figure()
    # Overlay original density as background
    plt.imshow(target_density, cmap='gray', alpha=0.3, extent=(0, IMG_SIZE, 0, IMG_SIZE), origin='lower')
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=assigned_component, s=1, cmap='jet', alpha=0.2)
    plt.title('Pixel Assignment to GMM Components')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Assigned Component')
    plt.show()
    # 3. Calculate Sigma via Weighted Scatter Matrix
    for k in range(K):
        mu_k = means[k]
        
        # Responsibility/Weight for component k (mass assigned to nearest mean)
        r_ik = (assigned_component == k).astype(float) * flat_density
        
        # Total mass for component k
        r_k_sum = r_ik.sum()
        
        # Calculate the scatter matrix for component k
        # Sigma_k = (1/r_k_sum) * Sum_i [ r_ik * (x_i - mu_k) * (x_i - mu_k)^T ]
        
        diff = coordinates - mu_k # (N, 2)
        weighted_diff = r_ik[:, np.newaxis] * diff # (N, 2), element-wise product
        
        # Matrix multiplication: (N, 2).T @ (N, 2) -> (2, 2)
        Sigma_k = (weighted_diff.T @ diff) / r_k_sum
        
        # Ensure symmetry (for numerical stability)
        Sigma_k = (Sigma_k + Sigma_k.T) / 2
        sigmas.append(Sigma_k)

    # Plot 1 sigma countours overlaided on the original density for verification
    plt.figure()
    plt.imshow(target_density, cmap='viridis', aspect='auto')
    theta = np.linspace(0, 2 * np.pi, 100)
    for k in range(K):
        mu_k = means[k]
        Sigma_k = sigmas[k]
        
        # Eigen-decomposition for ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(Sigma_k)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        # 1-sigma ellipse
        a = np.sqrt(eigvals[0])  # Semi-major axis
        b = np.sqrt(eigvals[1])  # Semi-minor axis
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        
        ellipse_x = mu_k[0] + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
        ellipse_y = mu_k[1] + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
        
        plt.plot(ellipse_x, ellipse_y, label=f'Component {k+1} 1-sigma', linewidth=2)
    plt.title('Fitted GMM 1-Sigma Contours')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Density')
    plt.show()

    return {
        'pi': torch.tensor(pi_y_raw, dtype=torch.float32), 
        'mu': torch.tensor(np.stack(means), dtype=torch.float32), 
        'Sigma': torch.tensor(np.stack(sigmas), dtype=torch.float32)
    }

def matstruct_to_dict(obj):
    """
    Recursively convert MATLAB structs (loaded via scipy.io.loadmat) to Python dictionaries.
    """
    if isinstance(obj, mat_struct):
        result = {}
        for fieldname in obj._fieldnames:
            result[fieldname] = matstruct_to_dict(getattr(obj, fieldname))
        return result
    elif isinstance(obj, np.ndarray):
        if obj.dtype == 'object':
            return [matstruct_to_dict(o) for o in obj]
        else:
            return obj
    else:
        return obj

def commonIndexFromSteps(data_struct, steps_to_process=None):
    """
    Get common indices for DTOTR2 images based on specified steps to process.
    Args:
        data_struct: Data structure containing image and scalar information.
        steps_to_process: List of steps to filter common indices. If None, all steps are used.

        data_struct.scalars.steps is indexed by common index.

    """
    DTOTR2commonind_all = data_struct.images.DTOTR2.common_index - 1
    # If there are any same values in DTOTR2commonind_all, raise error
    print("Min and Max of DTOTR2 common indices:", np.min(DTOTR2commonind_all), np.max(DTOTR2commonind_all))
    if len(DTOTR2commonind_all) != len(set(DTOTR2commonind_all)):
        raise ValueError("DTOTR2 common indices contain duplicates.")
    print("Total number of DTOTR2 common indices:", len(DTOTR2commonind_all))
    step_comp = data_struct.scalars.steps[DTOTR2commonind_all]
    # print the number of 0, 1, 2, ... steps in step_comp
    unique, counts = np.unique(step_comp, return_counts=True)
    step_count_dict = dict(zip(unique, counts))
    print("Step counts in DTOTR2 common indices:", step_count_dict)
    if steps_to_process is not None:
        DTOTR2commonind = [i for i in DTOTR2commonind_all if data_struct.scalars.steps[i] in steps_to_process]
    else:
        DTOTR2commonind = DTOTR2commonind_all
    return DTOTR2commonind


def extractDAQBSAScalars(data_struct, step_list=None, filter_index= True):
    """
    Extracts BSA scalar data from the provided data structure, filtered with scalar common index, and also with step list if provided.
    Args:
        data_struct: Data structure containing scalar information.
        common_index: Optional list of indices to filter the scalar data.
    """
    data_struct = matstruct_to_dict(data_struct)
    dataScalars = data_struct['scalars']
    if step_list is not None:
        idx = [i for i in dataScalars['common_index'].astype(int).flatten()
               if dataScalars['steps'][i] in step_list ]
    else:
        idx = dataScalars['common_index'].astype(int).flatten()
        
    fNames = list(dataScalars.keys())
    isBSA = [name for name in fNames if name.startswith('BSA_List_')]

    bsaScalarData = []
    bsaVarPVs = []

    for bsaName in isBSA:
        bsaList = dataScalars[bsaName]
        varNames = list(bsaList.keys())
        bsaListData = []

        for varName in varNames:
            varData = np.array(bsaList[varName]).squeeze()
            if varData.size == 0:
                continue
            if filter_index:
                varData = varData[idx]  # apply common index
            varData = np.nan_to_num(varData)  # replace NaN with 0
            bsaListData.append(varData)

        # Add to output arrays
        bsaVarPVs.extend([vn for vn in varNames if bsaList[vn].size != 0])
        if bsaListData:
            bsaScalarData.extend(bsaListData)

    bsaScalarData = np.array(bsaScalarData)
    return bsaScalarData, bsaVarPVs

def apply_tcav_zeroing_filter(bsaScalarData, bsaVarPVs):
    """
    Modifies the bsaScalarData array in-place by zeroing out the 
    TCAV_LI20_2400_P (phase) data points for any index where TCAV_LI20_2400_A (amplitude) is less than 1.

    This enhances Principal Component Analysis if your dataset includes all of TCAV phase -90, 0, and 90 degrees images.
    
    The function assumes:
    1. bsaScalarData has shape (N_variables, N_samples).
    2. The variable order in bsaScalarData corresponds to bsaVarPVs.
    3. Both TCAV_LI20_2400_P and TCAV_LI20_2400_A are present in bsaVarPVs.

    Args:
        bsaScalarData (np.ndarray): The 2D array of BSA scalar data (N_vars x N_samples).
        bsaVarPVs (list): List of PV names corresponding to the rows of bsaScalarData.
        
    Returns:
        np.ndarray: The modified bsaScalarData array.
    """
    
    pv_p_name = 'TCAV_LI20_2400_P'
    pv_a_name = 'TCAV_LI20_2400_A'
    
    # --- 1. Locate the indices of the required PVs ---
    try:
        # Find the row index corresponding to the Phase (P) and Amplitude (A) PVs
        idx_p = bsaVarPVs.index(pv_p_name)
        idx_a = bsaVarPVs.index(pv_a_name)
    except ValueError as e:
        print(f"Error: Required PV not found in bsaVarPVs. Missing: {e}")
        # Return unmodified data if the required PVs are not available
        return bsaScalarData

    # --- 2. Extract the Amplitude data ---
    # Since bsaScalarData is (N_vars, N_samples), we take the row corresponding to idx_a
    tcav_a_data = bsaScalarData[idx_a, :]

    # --- 3. Create the Boolean Mask ---
    # Identify all column indices (samples) where the Amplitude is less than 1.0
    zeroing_mask = tcav_a_data < 1.0

    # --- 4. Apply the Zeroing Filter ---
    # Use the mask to set the corresponding samples in the Phase (P) row to zero
    print(f"Applying filter: Setting {np.sum(zeroing_mask)} samples of {pv_p_name} to 0 where {pv_a_name} < 1.0")
    bsaScalarData[idx_p, zeroing_mask] = 0.0

    return bsaScalarData


def cropProfmonImg(img, xrange, yrange, plot_flag=False):
    """
    Crop a 2D image around the peak of horizontal and vertical projections,
    returning a region of shape (2*yrange, 2*xrange), with padding if needed.
    Center of mass is always at the center of the cropped image.
    Justification: For TCAV images, center of x represents zeta = 0, which can be arbitrarily chosen.
    center of y represents E=10.0GeV, whose variation seems much smaller than the 'orbit' effect of quadropoles which causes unwanted image shifting in y.

    Args:
        img (2D np.ndarray): Input image.
        xrange (int): Half-width in x-direction.
        yrange (int): Half-height in y-direction.
        plot_flag (bool): If True, show plots.

    Returns:
        cropped_img (2D np.ndarray): Cropped image (always 2*yrange by 2*xrange).
        error_flag (int): 0 if successful, 1 if fallback used.
    """
    img = img.astype(float)
    img_h, img_w = img.shape


    # Use max projections to find ROI center
    x_peak = int(np.argmax(np.sum(img, axis=0)))
    y_peak = int(np.argmax(np.sum(img, axis=1)))

    # Desired window size, larger to allow for centering later
    x_start = x_peak - xrange*2
    x_end   = x_peak + xrange*2
    y_start = y_peak - yrange*2
    y_end   = y_peak + yrange*2

    # Compute required padding if indices go out of bounds
    pad_left   = max(0, -x_start)
    pad_right  = max(0, x_end - img_w)
    pad_top    = max(0, -y_start)
    pad_bottom = max(0, y_end - img_h)

    # Pad image as needed
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        img = np.pad(img, 
                     ((pad_top, pad_bottom), (pad_left, pad_right)), 
                     mode='constant', constant_values=0)
        error_flag = 1  # Fallback was needed
    else:
        error_flag = 0

    # Recalculate indices after padding
    x_start += pad_left
    x_end   += pad_left
    y_start += pad_top
    y_end   += pad_top

    # Crop
    cropped_img = img[y_start:y_end, x_start:x_end]

    # Calculate center of mass for cropped image
    cropped_h, cropped_w = cropped_img.shape
    x_com_cropped = int(np.round(np.sum(np.arange(cropped_w) * np.sum(cropped_img, axis=0)) / np.sum(cropped_img)))
    y_com_cropped = int(np.round(np.sum(np.arange(cropped_h) * np.sum(cropped_img, axis=1)) / np.sum(cropped_img)))

    # Shift cropped image to center the COM
    shift_x = (cropped_w // 2) - x_com_cropped
    shift_y = (cropped_h // 2) - y_com_cropped
    cropped_img = np.roll(cropped_img, shift_x, axis=1)
    cropped_img = np.roll(cropped_img, shift_y, axis=0)

    # Crop again to ensure final size
    cropped_img = cropped_img[(cropped_h//2 - yrange):(cropped_h//2 + yrange),
                              (cropped_w//2 - xrange):(cropped_w//2 + xrange)]

    # Sanity check
    if cropped_img.shape != (2*yrange, 2*xrange):
        warnings.warn(f"Cropped image shape mismatch: got {cropped_img.shape}")
        error_flag = 1

    if plot_flag:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].imshow(img, cmap='viridis', aspect='auto')
        axs[0].set_title("Original / Padded Image")
        axs[1].imshow(cropped_img, cmap='viridis', aspect='auto')
        axs[1].set_title("Cropped Image")
        plt.tight_layout()
        plt.show()

    return cropped_img, error_flag

    
def segment_centroids_and_com(image, nrows, return_com=True):
    """
    Segments the image row-wise and computes the center of mass for each row.

    Parameters:
    - image: 2D numpy array (cropped and processed XTCAV image)
    - nrows: int, number of rows (typically image.shape[0])
    - return_com: if True, returns center of mass positions; else returns only row indices

    Returns:
    - centroid_indices: numpy array of row indices (0 to nrows-1)
    - centers_of_mass: numpy array of COM along the horizontal axis for each row
    """

    height, width = image.shape
    centroid_indices = np.arange(nrows)
    centers_of_mass = np.zeros(nrows)

    for i in range(nrows):
        row = image[i, :]
        total_mass = np.sum(row)
        if total_mass > 0:
            x_coords = np.arange(width)
            com = np.sum(row * x_coords) / total_mass
        else:
            com = np.nan
        centers_of_mass[i] = com

    if return_com:
        return centroid_indices, centers_of_mass
    else:
        return centroid_indices

def plot2DbunchseparationVsCollimatorAndBLEN(bc14BLEN, step_vals, bunchSeparation, idx_list, subplot_index, title, N, caxis_limits):
    plt.subplot(1, 2, subplot_index)
    
    # Build 2D histogram grid
    hist2d, xedges, yedges = np.histogram2d(
        bc14BLEN[0][idx_list], step_vals[idx_list],
        bins=N,
        weights=bunchSeparation[idx_list] * 3e8 * 1e6  # Convert to microns
    )
    
    # Count for normalization
    counts, _, _ = np.histogram2d(
        bc14BLEN[0][idx_list], step_vals[idx_list], bins=[xedges, yedges]
    )
    
    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_hist2d = np.where(counts > 0, hist2d / counts, np.nan)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = plt.imshow(avg_hist2d.T, origin='lower', aspect='auto', extent=extent, cmap='jet', vmin=caxis_limits[0], vmax=caxis_limits[1])
    plt.xlabel('BC14 BLEN')
    plt.ylabel('Notch Position')
    plt.title(title)
    plt.colorbar(im, label='Bunch Separation [μm]')

def extract_processed_images(data_struct, experiment, xrange=100, yrange=100, hotPixThreshold=1e3, sigma=1, threshold=5, step_list=None,
                             roi_xrange=None, roi_yrange=None
                             ):
    """
    Processes DTOTR2 images from HDF5 files, applying module-defined cropping, 
    filtering, and calculating horizontal projections (current profiles).

    Assumes the following are available in the current module's scope:
    - cropProfmonImg, median_filter, gaussian_filter (functions)
    - xrange, yrange, hotPixThreshold, sigma, threshold (constants)

    Args:
        data_struct (object): Structure containing image locations and backgrounds 
                              (e.g., data_struct.images.DTOTR2.loc, data_struct.backgrounds.DTOTR2).
        experiment (str): The name of the experiment, used in the file path pattern.
        xrange (int): Half-width for cropping in x-direction.
        yrange (int): Half-height for cropping in y-direction.
        hotPixThreshold (float): Threshold for hot pixel masking.
        sigma (float): Sigma for Gaussian smoothing.
        threshold (float): Threshold below which pixel values are set to zero.
        step_list (list): List of steps to process. If None, process all steps.
        roi_xrange (tuple): Optional custom cropping range in x-direction (min, max), applied before peak finding. Both roi_xrange and roi_yrange must be provided to apply.
        roi_yrange (tuple): Optional custom cropping range in y-direction (min, max), applied before peak finding. Both roi_xrange and roi_yrange must be provided to apply.

    Returns:
        tuple: A tuple containing the processed image arrays:
               (xtcavImages, xtcavImages_raw, horz_proj, LPSImage)
    """
    xtcavImages_list = []
    xtcavImages_list_raw = []
    horz_proj_list = []
    LPSImage = []
    stepsAll = data_struct.params.stepsAll
    if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
        stepsAll = [1]
    steps_to_process = stepsAll if step_list is None else [s for s in stepsAll if s in step_list]
    for a in range(len(steps_to_process)):
        # --- Determine File Path ---
        if len(steps_to_process) == 1:
            raw_path = data_struct.images.DTOTR2.loc
        else:
            raw_path = data_struct.images.DTOTR2.loc[a]
        
        # Search for the expected file name pattern
        match = re.search(rf'({experiment}_\d+/images/DTOTR2/DTOTR2_data_step\d+\.h5)', raw_path)
        if not match:
            raise ValueError(f"Path format invalid or not matched: {raw_path}")

        DTOTR2datalocation = '../../data/raw/' + experiment + '/' + match.group(0)

        # --- Read and Prepare Data ---
        with h5py.File(DTOTR2datalocation, 'r') as f:
            data_raw = f['entry']['data']['data'][:].astype(np.float64) # shape: (N, H, W)
        
        # Transpose to shape: (H, W, N) - Height, Width, Shots
        DTOTR2data_step = np.transpose(data_raw, (2, 1, 0))
        # Subtract background (H, W) from all shots (H, W, N)
        xtcavImages_step = DTOTR2data_step - data_struct.backgrounds.DTOTR2[:,:,np.newaxis].astype(np.float64)
        
        # --- Process Individual Shots ---
        for idx in tqdm(range(DTOTR2data_step.shape[2]), desc="Processing Shots Step {}, {} samples".format(steps_to_process[a], DTOTR2data_step.shape[2])):
            if idx is None:
                continue
                
            image = xtcavImages_step[:,:,idx] # Single shot (H, W)
            
            # Store Raw (background-subtracted) Image
            xtcavImages_list_raw.append(image[:,:,np.newaxis])

            # Apply ROI cropping if specified

            if roi_xrange is not None and roi_yrange is not None:
                x_min, x_max = roi_xrange
                y_min, y_max = roi_yrange
                image = image[y_min:y_max, x_min:x_max]
            
            # Crop image
            image_cropped, _ = cropProfmonImg(image, xrange, yrange, plot_flag=False)
            
            # Filter and mask hot pixels
            img_filtered = median_filter(image_cropped, size=3)
            hotPixels = img_filtered > hotPixThreshold
            img_filtered = np.ma.masked_array(img_filtered, hotPixels)
            
            # Gaussian smoothing and thresholding
            processed_image = gaussian_filter(img_filtered, sigma=sigma, radius = 6*sigma + 1)
            processed_image[processed_image < threshold] = 0.0
            
            # Calculate current profiles (Horizontal Projection)
            horz_proj_idx = np.sum(processed_image, axis=0)
            horz_proj_idx = horz_proj_idx[:,np.newaxis]
            
            # Prepare for collection
            processed_image = processed_image[:,:,np.newaxis]
            image_ravel = processed_image.ravel()
            
            horz_proj_list.append(horz_proj_idx)
            xtcavImages_list.append(processed_image)
            LPSImage.append([image_ravel]) 

    # --- Concatenate Results ---
    xtcavImages = np.concatenate(xtcavImages_list, axis=2)
    del xtcavImages_list  # Free memory
    xtcavImages_raw = np.concatenate(xtcavImages_list_raw, axis=2)
    del xtcavImages_list_raw  # Free memory
    horz_proj = np.concatenate(horz_proj_list, axis=1)
    LPSImage = np.concatenate(LPSImage, axis = 0)

    print("Processed XTCAV Images shape:", xtcavImages.shape)

    # --- Apply Common Indexing ---
    print("StepsToProcess:"+str(steps_to_process))
    image_common_index = [data_struct.images.DTOTR2.common_index[i] - 1 for i in range(len(data_struct.images.DTOTR2.common_index)) if data_struct.scalars.steps[data_struct.scalars.common_index[i]] in steps_to_process]
    horz_proj = horz_proj[:, image_common_index]
    xtcavImages = xtcavImages[:, :, image_common_index]
    xtcavImages_raw = xtcavImages_raw[:, :, image_common_index]
    # Final Arrays
    LPSImage = LPSImage[image_common_index,:] 
    # This way, xtcavImages[some_common_index] corresponds to the shot with that common index.

    return xtcavImages, xtcavImages_raw, horz_proj, LPSImage

def construct_centroid_function(images, off_idx, smoothing_window_size=5, max_degree=1):
    """
    Constructs a smoothed centroid function with localized quadratic extrapolation.

    This function calculates the mean horizontal center of mass (COM) for each
    row across a selection of images. It handles unreliable data with advanced logic:
    1.  **Interpolation**: Fills gaps for rows with high COM variance using linear
        interpolation between stable rows.
    2.  **Smoothing**: Applies a moving average filter to the entire COM profile.
    3.  **Local Extrapolation**: For rows far from any stable data, it performs
        two separate polynomial fits:
        -   **Top Extrapolation**: Fits a degree-2 polynomial to the top-most
            stable rows (up to 7 points) to project the trend upwards.
        -   **Bottom Extrapolation**: Fits a separate degree-2 polynomial to the
            bottom-most stable rows to project the trend downwards.
        This local approach correctly handles non-uniform trends (e.g., S-curves).

    Args:
        images (list or np.ndarray): A list or 3D NumPy array of 2D image arrays.
                                     All images must have the same dimensions.
        off_idx (list or np.ndarray): Indices of images to use for the calculation.
        smoothing_window_size (int, optional): The size of the moving average window
                                               for smoothing. Must be an odd number.
                                               Defaults to 5.

    Returns:
        np.ndarray: A 1D array where each value is the integer horizontal shift
                    required to center the content of that row.
    """
    if not isinstance(off_idx, (list, np.ndarray)):
        raise ValueError("off_idx must be a non-empty list or array of indices.")
    if smoothing_window_size % 2 != 1:
        raise ValueError("smoothing_window_size must be a positive odd number.")

    # 1. Select images and get dimensions
    selected_images = np.array([images[:,:,i] for i in off_idx])
    num_images, num_rows, num_cols = selected_images.shape
    image_center = num_cols / 2.0

    # 2. Calculate Center of Mass (COM) for each row
    col_indices = np.arange(num_cols)
    epsilon = 1e-9
    row_sums = selected_images.sum(axis=2)
    all_row_coms = np.sum(selected_images * col_indices, axis=2) / (row_sums + epsilon)
    all_row_coms[row_sums == 0] = np.nan

    # 3. Identify stable ("good") rows
    mean_coms = np.nanmean(all_row_coms, axis=0)
    std_dev_coms = np.nanstd(all_row_coms, axis=0)
    variance_threshold = 0.15 * num_cols
    good_rows_indices = np.where(std_dev_coms <= variance_threshold)[0]
    all_row_indices = np.arange(num_rows)
    
    # Handle cases with insufficient good data
    if len(good_rows_indices) < 2:
        print("Warning: Fewer than 2 stable rows found. Cannot perform reliable analysis.")
        return np.zeros(num_rows, dtype=int)
        
    # 4. Interpolate and Smooth
    good_com_values = mean_coms[good_rows_indices]
    interpolated_coms = np.interp(all_row_indices, good_rows_indices, good_com_values)
    
    if smoothing_window_size > 1:
        kernel = np.ones(smoothing_window_size) / smoothing_window_size
        smoothed_coms = np.convolve(interpolated_coms, kernel, mode='same')
    else:
        smoothed_coms = interpolated_coms
        
    # 5. Perform LOCAL EXTRAPOLATION
    final_coms = np.copy(smoothed_coms) # Start with interpolated/smoothed data
    min_good_idx, max_good_idx = good_rows_indices[0], good_rows_indices[-1]

    # --- Top Extrapolation ---
    top_extrap_indices = np.arange(0, min_good_idx)
    if top_extrap_indices.size > 0:
        # Select up to degree+5 (7) points from the top of the stable region
        fit_indices = good_rows_indices[:max_degree + 5]
        fit_values = good_com_values[:max_degree + 5]
        
        # Need at least degree+1 points to fit. We require 2 for linear, 3 for quadratic.
        if len(fit_indices) >= 2:
            degree = min(max_degree, len(fit_indices) - 1)
            coeffs = np.polyfit(fit_indices, fit_values, degree)
            poly_func = np.poly1d(coeffs)
            final_coms[top_extrap_indices] = poly_func(top_extrap_indices)
            
    # --- Bottom Extrapolation ---
    bottom_extrap_indices = np.arange(max_good_idx + 1, num_rows)
    if bottom_extrap_indices.size > 0:
        # Select up to degree+5 (7) points from the bottom of the stable region
        fit_indices = good_rows_indices[-(max_degree + 5):]
        fit_values = good_com_values[-(max_degree + 5):]

        if len(fit_indices) >= 2:
            degree = min(max_degree, len(fit_indices) - 1)
            coeffs = np.polyfit(fit_indices, fit_values, degree)
            poly_func = np.poly1d(coeffs)
            final_coms[bottom_extrap_indices] = poly_func(bottom_extrap_indices)

    # 6. Calculate the final correction shift
    horizontal_correction = image_center - final_coms
    return np.round(horizontal_correction).astype(int)

def apply_centroid_correction(xtcavImages, off_idx):
    """
    Applies centroid correction to a set of XTCAV images based on a constructed centroid function.
    Args:
        xtcavImages (np.ndarray): 3D array of shape (H, W, N) containing XTCAV images.
        off_idx (list or np.ndarray): Indices of images to use for constructing the centroid function.
    Returns:
        tuple: Corrected xtcavImages, horizontal projections (horz_proj), and flattened LPSImage, along with the centroid corrections applied.
    """

    # Get the number of shots (N)
    N_shots = xtcavImages.shape[2]
    Nrows = xtcavImages.shape[0]
    centroid_corrections = construct_centroid_function(xtcavImages, off_idx)

    # Prepare lists to collect results
    horz_proj_list_new = []
    xtcavImages_list_new = []
    LPSImage_new = []
    # Iterate over all shots in the concatenated array
    for idx in range(N_shots):
        # Get the processed image for the current shot
        processed_image = xtcavImages[:, :, idx]

        # --- Apply centroid correction ---
        corrected_image = np.zeros_like(processed_image)
        for row in range(Nrows):
            # The centroid correction function should provide the 'shift' for each row
            shift = centroid_corrections[row]
            corrected_image[row, :] = np.roll(processed_image[row, :], shift)

        # --- Calculate current profiles ---
        # Sum along the horizontal (time) axis (axis=0) to get the vertical (energy) profile
        horz_proj_idx = np.sum(corrected_image, axis=0)
        horz_proj_idx = horz_proj_idx[:, np.newaxis] # Reshape to (W, 1)
        
        # Reshape corrected_image for concatenation
        corrected_image = corrected_image[:, :, np.newaxis] # Reshape to (H, W, 1)

        # --- Prepare LPS Image (flattened) ---
        image_ravel = corrected_image.ravel()

        # --- Combine results into lists ---
        horz_proj_list_new.append(horz_proj_idx)
        xtcavImages_list_new.append(corrected_image)
        LPSImage_new.append([image_ravel])
    print(N_shots)
    # --- Concatenate all shots ---
    # Recreate the final arrays
    xtcavImages = np.concatenate(xtcavImages_list_new, axis=2)
    horz_proj = np.concatenate(horz_proj_list_new, axis=1)
    LPSImage = np.concatenate(LPSImage_new, axis=0)
    return xtcavImages, horz_proj, LPSImage, centroid_corrections