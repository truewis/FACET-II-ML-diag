import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy
import warnings
from scipy.io.matlab.mio5_params import mat_struct

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


def extractDAQBSAScalars(data_struct):
    data_struct = matstruct_to_dict(data_struct)
    dataScalars = data_struct['scalars']
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