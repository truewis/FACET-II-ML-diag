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
    if steps_to_process is not None:
        DTOTR2commonind = [i for i in DTOTR2commonind_all if data_struct.scalars.steps[i] in steps_to_process]
    else:
        DTOTR2commonind = DTOTR2commonind_all
    return DTOTR2commonind


def extractDAQBSAScalars(data_struct, common_index=None):
    data_struct = matstruct_to_dict(data_struct)
    dataScalars = data_struct['scalars']
    if common_index is not None:
        idx = common_index
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
            varData = varData[idx]  # apply common index
            varData = np.nan_to_num(varData)  # replace NaN with 0
            bsaListData.append(varData)

        # Add to output arrays
        bsaVarPVs.extend([vn for vn in varNames if bsaList[vn].size != 0])
        if bsaListData:
            bsaScalarData.extend(bsaListData)

    bsaScalarData = np.array(bsaScalarData)
    return bsaScalarData, bsaVarPVs


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
    DTOTR2commonind = commonIndexFromSteps(data_struct, steps_to_process)
    print("Number of shots after applying common index and step range:", len(DTOTR2commonind))
    # Example: If DTOTR2commonind = [0,1,4,6],  new index is [0,1,0,0,2,0,3,...]
    new_index_list = np.full(np.max(DTOTR2commonind) + 1, -1, dtype=int)
    new_index_list[DTOTR2commonind] = np.arange(len(DTOTR2commonind))
    new_index_list[new_index_list == -1] = 0
    horz_proj = horz_proj[:, new_index_list]
    xtcavImages = xtcavImages[:, :, new_index_list]
    xtcavImages_raw = xtcavImages_raw[:, :, new_index_list]
    # Final Arrays
    LPSImage = LPSImage[new_index_list,:] 
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