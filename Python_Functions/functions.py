import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy
import warnings
from scipy.io.matlab.mio5_params import mat_struct
from scipy.ndimage import median_filter, gaussian_filter
from scipy.optimize import curve_fit
import re
import h5py
from tqdm import tqdm # Import tqdm
import torch.nn as nn
import torch
from scipy.signal import find_peaks
import h5py
import re
import os
from pathlib import Path
from Python_Functions.gmm import bigaussian_1d
from scipy.ndimage import rotate
from sklearn.linear_model import LinearRegression # Need this for fitting the lines in the LPS images
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
        
def find_experiment_file(experiment: str, runname: str, base_nas_path: str = '/nas/nas-li20-pm00') -> Path | None:
    """
    Finds the full path to a specific experiment file, searching recursively
    under the base NAS directory structure.

    The expected file structure is:
    /nas/nas-li20-pm00/{experiment}/{year}/{date}/{experiment}_{runname}/{experiment}_{runname}.mat

    Args:
        experiment: The name of the experiment (e.g., 'E300').
        runname: The unique run identifier (e.g., '12431').
        base_nas_path: The root directory for the NAS storage.

    Returns:
        A pathlib.Path object of the found file, or None if the file is not found.
    """
    # 1. Define the part of the filename that is constant
    target_filename = f"{experiment}_{runname}.mat"
    
    # 2. Start the search from the experiment directory within the base NAS path
    search_root = Path(base_nas_path) / experiment
    print(f"Searching in: {search_root.resolve()}")

    if not search_root.exists():
        print(f"Error: The base experiment directory '{search_root}' does not exist.")
        return None

    # 3. Use rglob (recursive glob) to find the file anywhere under the search_root.
    # The glob pattern '**/' means 'any directory or sub-directory'.
    # We are looking for the specific filename.
    try:
        # Note: rglob returns a generator, so we convert it to a list and take the first item.
        # Since you stated there should only be one match, this is safe.
        found_files = list(search_root.rglob(target_filename))

        if found_files:
            return found_files[0]
        else:
            print(f"File not found: {target_filename} for experiment '{experiment}' and runname '{runname}'.")
            return None

    except PermissionError:
        print(f"Error: Permission denied while accessing files in {search_root}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during search: {e}")
        return None

def analyze_eos_and_cher(data_struct, experiment='', runname='', 
                         EOS2ymin=200, EOS2ymax=None,
                         CHERymin=1000, CHERymax=None,
                         xmin=700, SYAGxmax=964,
                         tenGeVCHERpixel=380,
                         mindels=40e-6, maxdels=85e-6,
                         skipEOSanalysis=True,
                         goosing=False,
                         debug = True,
                         EOS_cal=17.94e-15):
    """
    Port of the Claudio Emma's MATLAB EOS/CHER analysis.
    args:
        data_struct: data structure containing images, backgrounds, params, scalars
        experiment: string, experiment name for path construction
        runname: string, run name for plot titles
        EOS2ymin, EOS2ymax: vertical ROI for EOS2 analysis
        CHERymin, CHERymax: vertical ROI for CHER analysis
        SYAGxmin, SYAGxmax: horizontal ROI for SYAG analysis
        tenGeVCHERpixel: pixel position corresponding to 10 GeV in CHER image
        mindels, maxdels: min and max acceptable EOS separations (meters)
        skipEOSanalysis: if True, skip EOS separation calculation (still compute projections)
        goosing: if True, process every other shot for EOS analysis
        EOS_cal: EOS calibration in seconds/pixel

    Returns a dict with keys:
      - dels: array of EOS separations (meters) (or zeros if skipped)
      - bc14BLEN: array of BC14 BLEN values (shape: (nshots,))
      - EOS2horzProj: 2D array (rows x shots) of vertical projections used for waterfall
      - fig: matplotlib Figure (waterfall + separation/BLEN plot) or None if plotting suppressed
      - sorted_idx: indices starting from 0 

    Notes:
      - The function expects data_struct to contain images.EOS2.loc (list or single path),
        backgrounds.EOS2, and params.stepsAll. It will attempt to read HDF5 with h5py.
      - Common index handling assumes MATLAB-style 1-based common_index values and converts to 0-based.
    """
    import matplotlib.pyplot as plt

    c = 3e8

    # defaults for ROI extents if not provided
    if EOS2ymax is None:
        EOS2ymax = EOS2ymin + 100
    if CHERymax is None:
        CHERymax = CHERymin + 1000

    # --- Collect BC14 BLEN from BSA scalars ---
    bsaScalarData, bsaVars = extractDAQBSAScalars(data_struct)
    try:
        pvidx = bsaVars.index('BLEN_LI14_888_BRAW')
    except ValueError:
        raise ValueError("BLEN_LI14_888_BRAW not found among BSA PVs.")
    # bsaScalarData shape is (N_vars, N_samples)
    bc14BLEN = bsaScalarData[pvidx, :].copy()

    # --- Load/concatenate EOS2 HDF5 data across steps ---
    stepsAll = data_struct.params.stepsAll
    if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
        stepsAll = [1]

    EOSdata = None
    # Handle loc being list-like or single
    locs = data_struct.images.EOS2.loc
    # ensure iterable
    if not isinstance(locs, (list, tuple, np.ndarray)):
        locs = [locs]

    for a in range(len(stepsAll)):
        # pick appropriate location entry
        try:
            loc_entry = locs[a]
        except Exception:
            loc_entry = locs[0]
        raw_path = loc_entry
        # Search for the expected file name pattern
        match = re.search(rf'({experiment}_\d+/images/EOS2/EOS2_data_step\d+\.h5)', raw_path)
        if not match:
            raise ValueError(f"Path format invalid or not matched: {raw_path}")

        loc_entry = '../../data/raw/' + experiment + '/' + match.group(0)
        # try to open directly, otherwise attempt fallback construction similar to other functions
        print(f"Loading EOS2 data from {loc_entry} ...")
        try:
            with h5py.File(loc_entry, 'r') as f:
                stepdata = f['entry']['data']['data'][:].astype(np.float64)
                print(f"Loaded stepdata from {loc_entry}")
                print(f"stepdata shape: {stepdata.shape}")
        except Exception:
            # fallback: attempt to extract a trailing path fragment containing experiment and build relative path
            m = re.search(rf'({experiment}_\d+/images/EOS2/EOS2_data_step\d+\.h5)', str(loc_entry))
            if m:
                candidate = '../../data/raw/' + experiment + '/' + m.group(0)
                with h5py.File(candidate, 'r') as f:
                    stepdata = f['entry']['data']['data'][:].astype(np.float64)
            else:
                raise

        # concatenate along 3rd axis (MATLAB's 3rd dim is shots)
        if EOSdata is None:
            EOSdata = stepdata.copy()
        else:
            # stepdata shape (H, W, Nstep); append along third axis
            EOSdata = np.concatenate([EOSdata, stepdata], axis=2)

    if EOSdata is None:
        raise RuntimeError("No EOS data loaded.")

    # --- Keep only shots with common index and subtract background ---
    # common_index in data_struct likely 1-based
    eos_common = data_struct.images.EOS2.common_index
    eos_common_idx = [int(i) - 1 for i in eos_common]
    EOSdata = EOSdata.transpose(2, 1, 0)[:, :, eos_common_idx]  # to (H, W, Shots)

    # subtract background if available (convert to float)
    try:
        bg = np.array(data_struct.backgrounds.EOS2).astype(np.float64)
        # MATLAB code used double and a - -background pattern; here subtract background
        EOSdata = EOSdata - bg[:, :]
    except Exception:
        # if background missing, proceed without subtraction
        pass
    print(f"EOSdata shape after loading and bg subtraction: {EOSdata.shape}")

    nshots = EOSdata.shape[2]
    # prepare projection matrix (rows x shots)
    EOS2horzProj = np.zeros((EOSdata.shape[0], nshots), dtype=float)
    shotROIs = np.zeros((EOSdata.shape[0], EOSdata.shape[1], nshots), dtype=float)

    dels = np.zeros(nshots, dtype=float)
    print("Starting EOS2 analysis...")
    plot_frequency = 20 if not goosing else 21
    if not skipEOSanalysis:
        if goosing:
            rng = range(1, nshots, 2)
        else:
            rng = range(nshots)
        for a in rng:
            shot = EOSdata[:, :, a]
            # take vertical ROI
            shot_roi = shot[EOS2ymin:EOS2ymax, :]
            shotROIs[EOS2ymin:EOS2ymax, :, a] = shot_roi
            # vertical projection (sum across columns)
            proj = np.sum(shot_roi, axis=1)
            # Plot projection for debugging
            if debug and a % plot_frequency == 1:
                plt.imshow(shot_roi , cmap='viridis', aspect='auto')
                plt.show()
            EOS2horzProj[EOS2ymin:EOS2ymax, a] = proj
            # find peaks on the projection
            # height and prominence thresholds may need adjustment based on signal levels
            log_proj = np.log(1+proj)
            peaks, props = find_peaks(gaussian_filter(log_proj, sigma=0.8), height=9.2, prominence=0.05)
            if peaks.size < 2:
                dels[a] = 0.0
                continue

            #%% ######################################################################################
            # Option: Extra bigaussian fit to refine peak positions
            try:
                x_coords = np.arange(len(proj))
                # Initial guess for bigaussian fit: [amp1, sigma1, mean1, amp2, sigma2, mean2]
                # Sigma is chosen as 10 pixels arbitrarily. Amplitudes are peak heights.
                p0_x = [x_coords[peaks[0]], 10 ,log_proj[peaks[0]], x_coords[peaks[1]], 10 ,log_proj[peaks[1]]]
                popt_x, _ = curve_fit(bigaussian_1d, x_coords, log_proj, p0=p0_x, maxfev=5000)
                peak_sep_pix = popt_x[3] - popt_x[0]
            except Exception as e:
                peak_sep_pix = abs(peaks[0] - peaks[1])


            # Plot projection and peaks for debugging
            if debug and a % plot_frequency == 1:
                plt.plot(log_proj)
                plt.plot(peaks, log_proj[peaks], "x")
                plt.show()
            peak_vals = props['peak_heights']
            #%% ######################################################################################
            # Option: Just take the top 2 peaks by height
            top2 = np.argsort(peak_vals)[-2:]
            # map to actual peak positions
            pos = peaks[top2]
            # pick heights in descending order
            heights_sorted = peak_vals[top2][np.argsort(peak_vals[top2])[::-1]]
            # compute separation in pixels
            #peak_sep_pix = abs(int(pos[0]) - int(pos[1]))
            #%% ######################################################################################
            # Transpose horizontal projection by peak position.
            # EOS2horzProj[:, a] = np.roll(EOS2horzProj[:, a], -int(pos[1]))
            # reliability check: if largest*0.08 > second then unreliable
            # if heights_sorted[0] * 0.08 > heights_sorted[1]:
            #     dels[a] = 0.0
            # another reliability check: if tallest peak < 10000 counts, likely no signal
            # if heights_sorted[0] < 10000:
            #     dels[a] = 0.0
            # else:
            #%%
            dels[a] = peak_sep_pix * EOS_cal * c
        print("Done with EOS2 analysis!")
    else:
        # still compute vertical projections for plotting but leave dels zeros
        for a in range(nshots):
            shot = EOSdata[:, :, a]
            shot_roi = shot[EOS2ymin:EOS2ymax, :]
            EOS2horzProj[:, a] = np.sum(shot_roi, axis=1)
            shotROIs.append(None)

    # --- Prepare plotting arrays ---
    #sorted_idx = np.arange(nshots)
    sorted_idx = np.argsort(bc14BLEN)

    # From sorted_idx, remove entries where dels <50 or >200 microns
    valid_indices = np.where((dels[sorted_idx] >= mindels) & (dels[sorted_idx] <= maxdels))[0]
    sorted_idx = sorted_idx[valid_indices]
    sorted_bc14BLEN = bc14BLEN[sorted_idx]
    eos2sep_for_plot = dels[sorted_idx] * 1e6  # microns
    eos2sep_for_plot[eos2sep_for_plot == 0] = np.nan
    # --- Plot waterfall and separation vs BLEN ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].imshow(EOS2horzProj[:, sorted_idx], cmap='jet', aspect='auto', origin='lower')
    axs[0].set(ylim=(0, 100))
    axs[0].set_title(f"EOS2 waterfall. DAQ {experiment} {runname}")

    ax2 = axs[1]
    ax2.plot(eos2sep_for_plot, '-k', label='EOS separation [um]')
    ax2.set_ylabel('EOS bunch separation [um]')
    ax2_right = ax2.twinx()
    ax2_right.plot(sorted_bc14BLEN, '-r', label='bc14 blen')
    ax2_right.set_ylabel('bc14 blen')
    ax2.set_title(f"EOS2 calibration = {1e6 * c * EOS_cal:.3g} um/pix")
    axs[1].legend(loc='upper left')
    ax2_right.legend(loc='upper right')
    plt.tight_layout()

    return {
        'dels': dels,
        'bc14BLEN': bc14BLEN,
        'EOS2horzProj': EOS2horzProj,
        'fig': fig,
        'shotROI': shotROIs,
        'sorted_idx': sorted_idx
    }
def analyze_SYAG(data_struct, experiment='', runname='', 
                         SYAGxmin=200, SYAGxmax=None,
                         CHERymin=1000, CHERymax=None,
                         tenGeVCHERpixel=380,
                         mindels=40e-6, maxdels=85e-6,
                         skipEOSanalysis=True,
                         goosing=False,
                         EOS_cal=17.94e-15,
                         debug=False,
                         step_selector=None,
                         min_prom = 2000
                         ):
    """
    Port of the Claudio Emma's MATLAB EOS/CHER analysis.
    args:
        data_struct: data structure containing images, backgrounds, params, scalars
        experiment: string, experiment name for path construction
        runname: string, run name for plot titles
        SYAGymin, SYAGymax: vertical ROI for SYAG analysis
        CHERymin, CHERymax: vertical ROI for CHER analysis
        SYAGxmin, SYAGxmax: horizontal ROI for SYAG analysis
        tenGeVCHERpixel: pixel position corresponding to 10 GeV in CHER image
        mindels, maxdels: min and max acceptable EOS separations (meters)
        skipEOSanalysis: if True, skip EOS separation calculation (still compute projections)
        goosing: if True, process every other shot for EOS analysis
        EOS_cal: EOS calibration in seconds/pixel
        debug: if True, show debug plots during analysis
        step_selector: list of steps to include in analysis (if None, include all)
        min_prom: minimum prominence for peak detection in SYAG analysis

    Returns a dict with keys:
      - dels: array of EOS separations (meters) (or zeros if skipped)
      - bc14BLEN: array of BC14 BLEN values (shape: (nshots,))
      - SYAGhorzProj: 2D array (rows x shots) of vertical projections used for waterfall
      - fig: matplotlib Figure (waterfall + separation/BLEN plot) or None if plotting suppressed
      - sorted_idx: indices starting from 0 

    Notes:
      - The function expects data_struct to contain images.SYAG.loc (list or single path),
        backgrounds.SYAG, and params.stepsAll. It will attempt to read HDF5 with h5py.
      - Common index handling assumes MATLAB-style 1-based common_index values and converts to 0-based.
    """
    import matplotlib.pyplot as plt

    c = 3e8

    # defaults for ROI extents if not provided
    if SYAGxmax is None:
        SYAGxmax = SYAGxmin + 100
    if CHERymax is None:
        CHERymax = CHERymin + 1000

    # --- Collect BC14 BLEN from BSA scalars ---
    bsaScalarData, bsaVars = extractDAQBSAScalars(data_struct)
    try:
        pvidx = bsaVars.index('BLEN_LI14_888_BRAW')
    except ValueError:
        raise ValueError("BLEN_LI14_888_BRAW not found among BSA PVs.")
    # bsaScalarData shape is (N_vars, N_samples)
    bc14BLEN = bsaScalarData[pvidx, :].copy()

    # --- Load/concatenate SYAG HDF5 data across steps ---
    stepsAll = data_struct.params.stepsAll
    if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
        stepsAll = [1]
    
    # If step_selector is provided, filter stepsAll accordingly
    if step_selector is not None:
        stepsAll = [step for step in stepsAll if step in step_selector]

    SYAGdata = None
    # Handle loc being list-like or single
    locs = data_struct.images.SYAG.loc
    # ensure iterable
    if not isinstance(locs, (list, tuple, np.ndarray)):
        locs = [locs]

    for a in range(len(stepsAll)):
        # pick appropriate location entry
        try:
            loc_entry = locs[a]
        except Exception:
            loc_entry = locs[0]
        raw_path = loc_entry
        # Search for the expected file name pattern
        match = re.search(rf'({experiment}_\d+/images/SYAG/SYAG_data_step\d+\.h5)', raw_path)
        if not match:
            raise ValueError(f"Path format invalid or not matched: {raw_path}")

        loc_entry = '../../data/raw/' + experiment + '/' + match.group(0)
        # try to open directly, otherwise attempt fallback construction similar to other functions
        print(f"Loading SYAG data from {loc_entry} ...")
        try:
            with h5py.File(loc_entry, 'r') as f:
                stepdata = f['entry']['data']['data'][:].astype(np.int16)
                print(f"Loaded stepdata from {loc_entry}")
                print(f"stepdata shape: {stepdata.shape}")

        except Exception:
            # fallback: attempt to extract a trailing path fragment containing experiment and build relative path
            m = re.search(rf'({experiment}_\d+/images/SYAG/SYAG_data_step\d+\.h5)', str(loc_entry))
            if m:
                candidate = '../../data/raw/' + experiment + '/' + m.group(0)
                with h5py.File(candidate, 'r') as f:
                    stepdata = f['entry']['data']['data'][:].astype(np.int16)
            else:
                raise

        # concatenate along 3rd axis (MATLAB's 3rd dim is shots)
        if SYAGdata is None:
            SYAGdata = stepdata.copy()
        else:
            # stepdata shape (Nstep, W, H); append along third axis
            SYAGdata = np.concatenate([SYAGdata, stepdata], axis=0)

    if SYAGdata is None:
        raise RuntimeError("No SYAG data loaded.")

    # --- Keep only shots with common index and subtract background ---
    # common_index in data_struct likely 1-based
    syag_common_idx = data_struct.images.SYAG.common_index - 1
    # If step_selector is provided, filter syag_common_idx accordingly
    if step_selector is not None:
        syag_common_idx = [syag_common_idx[idx] for idx in range(len(syag_common_idx)) if data_struct.scalars.steps[idx] in step_selector]
    SYAGdata = SYAGdata.transpose(2, 1, 0)[:, :, syag_common_idx]  # to (H, W, Shots)

    # subtract background if available (convert to float)
    try:
        bg = np.array(data_struct.backgrounds.SYAG).astype(np.float64)
        # MATLAB code used double and a - -background pattern; here subtract background
        SYAGdata = SYAGdata - bg[:, :]
    except Exception:
        # if background missing, proceed without subtraction
        pass
    print(f"EOSdata shape after loading and bg subtraction: {SYAGdata.shape}")

    nshots = SYAGdata.shape[2]
    # prepare projection matrix (rows x shots)
    SYAGhorzProj = np.zeros((SYAGdata.shape[0], nshots), dtype=float)

    dels = np.zeros(nshots, dtype=float)
    print("Starting SYAG analysis...")
    plot_frequency = 80 if not goosing else 81
    if not skipEOSanalysis:
        if goosing:
            rng = range(1, nshots, 2)
        else:
            rng = range(nshots)
        for a in rng:
            shot = SYAGdata[:, :, a]
            # take vertical ROI
            shot_roi = shot[:, SYAGxmin:SYAGxmax]

            # horizontal projection (sum across rows)
            proj = np.sum(shot_roi, axis=1)
            # Apply smoothing to projection
            proj = gaussian_filter(proj, sigma=5)
            # Plot projection for debugging
            if debug and a % plot_frequency == 1:
                # Maximum color is twice the median of the shot ROI
                plt.imshow(shot_roi , cmap='viridis', aspect='auto', vmax=2*np.median(shot_roi))
                plt.show()
            SYAGhorzProj[:,a] = proj
            # find peaks on the projection
            # Here we measure the distance between centroid of each two peak, rather than the positions of the peaks themselves.
            # This is because the distance width is more stable against local intensity fluctuations around the peaks.
            peaks, props = find_peaks(proj, prominence=min_prom)
            # exclude any peaks too close to edges, along with the associated properties
            valid_peaks_mask = (peaks > 20) & (peaks < len(proj) - 20)
            peaks = peaks[valid_peaks_mask]
            for key in props:
                props[key] = props[key][valid_peaks_mask]
            # Plot projection and peaks for debugging
            if debug and a % plot_frequency == 1:
                plt.plot(proj)
                plt.plot(peaks, proj[peaks], "x")
                plt.show()
            if peaks.size is not 2:
                dels[a] = 0.0
                continue
            # Centroid analysis is optional.
            # dip_index = np.argmin(proj[peaks[0]:peaks[1]]) + peaks[0]
            # #print(f"Shot {a}: Found dip at pixel {peaks[0]} with prominence {props['prominences'][0]}") 
            # # peak_sep_pix is full width half max of the dip.
            # # The 'height' of the dip is given by the prominence of the peak in -proj
            # prom = props['prominences']
            # # find left and right bases for FWHM
            # left_bases = props['left_bases']
            # right_bases = props['right_bases']
            # # tread the slope from left to right to find FWHM positions
            # def find_fwhm_binary_search(func, start_idx, end_idx, half_height, search_left):
            #     while start_idx < end_idx:
            #         mid_idx = (start_idx + end_idx) // 2
            #         if not search_left:
            #             if func[mid_idx] < half_height:
            #                 end_idx = mid_idx
            #             else:
            #                 start_idx = mid_idx + 1
            #         else:
            #             if func[mid_idx] < half_height:
            #                 start_idx = mid_idx + 1
            #             else:
            #                 end_idx = mid_idx
            #     return start_idx
            # fwhm_left_0 = find_fwhm_binary_search(proj-proj[left_bases[0]], left_bases[0], peaks[0] , prom[0]/2, search_left=True)
            # fwhm_right_0 = find_fwhm_binary_search(proj-proj[dip_index], peaks[0], dip_index , prom[0]/2, search_left=False)

            # #Flip the projection and tread the slope from right to left to find FWHM positions.
            # #This works but is a bit hacky.
            # fwhm_left_1 = find_fwhm_binary_search(proj-proj[dip_index], dip_index, peaks[1] , prom[1]/2, search_left=True)
            # fwhm_right_1 = find_fwhm_binary_search(proj-proj[right_bases[1]], peaks[1], right_bases[1] , prom[1]/2, search_left=False)
            # # Plot projection and cliffs for debugging
            # if debug and a % plot_frequency == 1:
            #     plt.plot(proj)
            #     plt.axvline(fwhm_left_0, color='r')
            #     plt.axvline(fwhm_right_0, color='r')
            #     plt.axvline(fwhm_left_1, color='g')
            #     plt.axvline(fwhm_right_1, color='g')
            #     # dip position
            #     plt.axvline(dip_index, color='k')
            #     plt.show()
            # # Transpose horizontal projection by first cliff position.
            # SYAGhorzProj[:, a] = np.roll(SYAGhorzProj[:, a], -int(fwhm_left_0))
            # # centroids are center of mass of each peak FWHM region
            # centroid_0 = np.sum(np.arange(fwhm_left_0, fwhm_right_0) * proj[fwhm_left_0:fwhm_right_0]) / np.sum(proj[fwhm_left_0:fwhm_right_0])
            # centroid_1 = np.sum(np.arange(fwhm_left_1, fwhm_right_1) * proj[fwhm_left_1:fwhm_right_1]) / np.sum(proj[fwhm_left_1:fwhm_right_1])
            # print(f"Shot {a}: Centroid positions {centroid_0}, {centroid_1}")
            # centroid_sep_pix = np.abs(centroid_0 - centroid_1)
            # Alternative: simply use peak positions
            centroid_sep_pix = abs(int(peaks[0]) - int(peaks[1]))
            dels[a] = centroid_sep_pix
        print("Done with SYAG analysis!")
    else:
        # still compute vertical projections for plotting but leave dels zeros
        for a in range(nshots):
            shot = SYAGdata[:, :, a]
            shot_roi = shot[:, SYAGxmin:SYAGxmax]
            SYAGhorzProj[:, a] = np.sum(shot_roi, axis=1)

    # --- Prepare plotting arrays ---
    #sorted_idx = np.arange(nshots)
    sorted_idx = np.argsort(bc14BLEN)
    # Only take indices where dels is not zero or nan
    valid_indices = np.where((dels[sorted_idx] >= mindels) & (dels[sorted_idx] <= maxdels))[0]
    sorted_idx = sorted_idx[valid_indices]
    sorted_bc14BLEN = bc14BLEN[sorted_idx]
    SYAGsep_for_plot = dels[sorted_idx] * 1e6  # microns
    SYAGsep_for_plot[SYAGsep_for_plot == 0] = np.nan
    # --- Plot waterfall and separation vs BLEN ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].imshow(SYAGhorzProj[:, sorted_idx], cmap='jet', aspect='auto', origin='lower')
    axs[0].set_title(f"SYAG waterfall. DAQ {experiment} {runname}")

    ax2 = axs[1]
    ax2.plot(SYAGsep_for_plot, '-k', label='SYAG separation [um]')
    ax2.set_ylabel('SYAG separation [um]')
    ax2_right = ax2.twinx()
    ax2_right.plot(sorted_bc14BLEN, '-r', label='bc14 blen')
    ax2_right.set_ylabel('bc14 blen')
    ax2.set_title(f"SYAG calibration = {1e6 * c * EOS_cal:.3g} um/pix")
    axs[1].legend(loc='upper left')
    ax2_right.legend(loc='upper right')
    plt.tight_layout()

    return {
        'dels': dels,
        'bc14BLEN': bc14BLEN,
        'SYAGhorzProj': SYAGhorzProj,
        'fig': fig,
        'sorted_idx': sorted_idx
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
        step_list: List of steps to filter the data. If None, all steps are used. Starts from 1.
        filter_index: If True, apply common index filtering.
    """
    data_struct = matstruct_to_dict(data_struct)
    dataScalars = data_struct['scalars']
    if step_list is not None:
        idx = [i for i in dataScalars['common_index'].astype(int).flatten() - 1
               if dataScalars['steps'][i] in step_list ]
    else:
        idx = dataScalars['common_index'].astype(int).flatten() - 1

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
def exclude_bsa_vars(bsaVars):
    """
    Identify indices of BSA variables to exclude based on predefined names and numeric patterns.
    This removes dependence on variables that are affected by the XTCAV settings.
    Args:
        bsaVars: List of BSA variable names.
    Returns:
        excluded_var_idx: List of indices of variables to exclude."""
        # List of BSA variable names to exclude
    exclude_bsa_vars = [
        'TCAV_LI20_2400_A',  # XTCAV Amplitude
        'TCAV_LI20_2400_P',  # XTCAV Phase
        
    ]
    # Search for variable names that contain four digit number larger than 3100, this is after the XTCAV.
    for var in bsaVars:
        match = re.search(r'(\d{4})', var)
        if match:
            number = int(match.group(1))
            if number >= 3100:
                exclude_bsa_vars.append(var)

    print("Excluding BSA Variables:", exclude_bsa_vars)
    excluded_var_idx = [i for i, var in enumerate(bsaVars) if var in exclude_bsa_vars]
    print("Excluded variable indices:", excluded_var_idx)
    return excluded_var_idx
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
    # --- Modification: Replace NaN values with zero ---
    cropped_img = np.nan_to_num(cropped_img, nan=0.0)
    # Calculate center of mass for cropped image
    cropped_h, cropped_w = cropped_img.shape
    # If sum is zero, set to center
    # Strongly smoothing the image to avoid issues with sparse hot pixels
    smoothed_img = gaussian_filter(cropped_img, sigma=5)
    if (np.sum(cropped_img) == 0):
        x_com_cropped = (x_start+x_end)//2
        y_com_cropped = (y_start+y_end)//2
    else:
        x_com_cropped = int(np.round(np.sum(np.arange(cropped_w) * np.sum(smoothed_img, axis=0)) / np.sum(smoothed_img)))
        y_com_cropped = int(np.round(np.sum(np.arange(cropped_h) * np.sum(smoothed_img, axis=1)) / np.sum(smoothed_img)))
        

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
                             roi_xrange=None, roi_yrange=None, do_load_raw = False, directory_path = '/nas/nas-li20-pm00/'
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
        step_list (list): List of steps to process. If None, process all steps. Steps start from 1.
        roi_xrange (tuple): Optional custom cropping range in x-direction (min, max), applied before peak finding. Both roi_xrange and roi_yrange must be provided to apply.
        roi_yrange (tuple): Optional custom cropping range in y-direction (min, max), applied before peak finding. Both roi_xrange and roi_yrange must be provided to apply.
        do_load_raw (bool): Load raw image for debugging. Uses a lot of memory.

    Returns:
        tuple: A tuple containing the processed image arrays:
               (xtcavImages, xtcavImages_raw, horz_proj, LPSImage)
    """
    xtcavImages_list = []
    xtcavImages_list_raw = []
    LPSImage = []
    stepsAll = data_struct.params.stepsAll
    step_size = None
    raw_width = None
    raw_height = None
    if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
        stepsAll = [1]
    steps_to_process = stepsAll if step_list is None else [s for s in stepsAll if s in step_list]
    for a in range(1, np.max(steps_to_process)+1):
        # --- Determine File Path ---
        if a in steps_to_process:
            raw_path = data_struct.images.DTOTR2.loc[a-1]
            # Search for the expected file name pattern
            match = re.search(rf'(images/DTOTR2/DTOTR2_data_step\d+\.h5)', raw_path)
            if not match:
                raise ValueError(f"Path format invalid or not matched: {raw_path}")

            DTOTR2datalocation = str(directory_path) + '/' + match.group(0)

            # --- Read and Prepare Data ---
            with h5py.File(DTOTR2datalocation, 'r') as f:
                data_raw = f['entry']['data']['data'][:].astype(np.float64) # shape: (N, H, W)
            
            # Transpose to shape: (H, W, N) - Height, Width, Shots
            DTOTR2data_step = np.transpose(data_raw, (2, 1, 0))
            # Subtract background (H, W) from all shots (H, W, N)
            try:
                # If there is background data
                xtcavImages_step = DTOTR2data_step - data_struct.backgrounds.DTOTR2[:,:,np.newaxis].astype(np.float64)
            except:
                xtcavImages_step = DTOTR2data_step
            step_size = DTOTR2data_step.shape[2]
            # --- Process Individual Shots ---
            for idx in tqdm(range(step_size), desc="Processing Shots Step {}, {} samples".format(a, step_size)):
                if idx is None:
                    continue
                    
                image = xtcavImages_step[:,:,idx] # Single shot (H, W)
                raw_width = image.shape[1]
                raw_height = image.shape[0]
                if do_load_raw:
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
                
                # Prepare for collection
                processed_image = processed_image[:,:,np.newaxis]
                image_ravel = processed_image.ravel()
                xtcavImages_list.append(processed_image)
                LPSImage.append([image_ravel]) 
        else:
            print(f"Skipping step {a} as it's not in steps_to_process.")

    # --- Concatenate Results ---
    xtcavImages = np.concatenate(xtcavImages_list, axis=2)
    del xtcavImages_list  # Free memory
    if do_load_raw:
        xtcavImages_raw = np.concatenate(xtcavImages_list_raw, axis=2)
        del xtcavImages_list_raw  # Free memory
    LPSImage = np.concatenate(LPSImage, axis = 0)

    print("Processed XTCAV Images shape:", xtcavImages.shape)

    # --- Apply Common Indexing ---
    print("StepsToProcess:"+str(steps_to_process))
    def steps_skipped_before(i):
        current_step = data_struct.scalars.steps[data_struct.scalars.common_index[i]]
        # Count how many steps less than current_step are not in steps_to_process
        return sum(1 for s in stepsAll if s < current_step and s not in steps_to_process)
    for i in range(len(data_struct.images.DTOTR2.common_index)):
        if data_struct.scalars.steps[data_struct.scalars.common_index[i]] in steps_to_process:
            print("Including common index {} where i is {}, step {}, steps skipped before: {}".format(data_struct.images.DTOTR2.common_index[i], i, data_struct.scalars.steps[data_struct.scalars.common_index[i]], steps_skipped_before(i)))
    image_common_index = [data_struct.images.DTOTR2.common_index[i] - 1 - steps_skipped_before(i) * step_size for i in range(len(data_struct.images.DTOTR2.common_index)) if data_struct.scalars.steps[data_struct.scalars.common_index[i]] in steps_to_process]
    xtcavImages = xtcavImages[:, :, image_common_index]
    if do_load_raw:
        xtcavImages_raw = xtcavImages_raw[:, :, image_common_index]
    # Final Arrays
    LPSImage = LPSImage[image_common_index,:] 
    # This way, xtcavImages[some_common_index] corresponds to the shot with that common index.
    if do_load_raw:
        return xtcavImages, xtcavImages_raw, None, LPSImage
    else:
        return xtcavImages, None, None, LPSImage

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
def get_image_angle_from_line_fit(image):
    """
    Fits a line (e.g., using linear regression on the image's center of mass per row)
    to the main feature and returns the angle of that line in degrees.
    """
    Nrows, Ncols = image.shape
    row_indices = np.arange(Nrows)
    
    # 1. Find the center of mass (COM) for each row (y-coordinate)
    # This is a simplification; a more robust method would be needed for real data.
    com_y = np.sum(image * np.arange(Ncols), axis=1) / np.sum(image, axis=1)
    
    # Filter out rows with zero sum (i.e., no signal) to avoid NaN/inf
    valid_mask = np.isfinite(com_y) & (np.sum(image, axis=1) > 0)
    
    if np.sum(valid_mask) < 2:
        return 0.0 # Cannot fit a line with less than 2 points

    # 2. Fit a line: row_index = m * com_x + c
    X = com_y[valid_mask].reshape(-1, 1)
    Y = row_indices[valid_mask]
    
    # Check if the shape of X is correct for LinearRegression
    if X.shape[0] < 2:
        return 0.0
    reg = LinearRegression().fit(X, Y)
    m = reg.coef_[0]
    
    # 3. Calculate the angle from the slope (m = tan(90 - theta) if X is horizontal axis)
    # Angle (theta) is measured from the vertical axis (y-axis/row_index)
    # m = d(row_index) / d(com_x) = dy/dx
    # Angle w.r.t the horizontal (x) axis is atan(m)
    angle_rad = np.arctan(m)
    angle_deg = np.degrees(angle_rad)
    # Debug: 
    print(f"Fitted line slope: {m:.4f}, angle (deg): {angle_deg:.2f}")
    # We want the angle w.r.t the vertical (y) axis for a standard image rotation
    # A positive slope (m>0) means the line is tilted to the right. 
    # An angle of 0 means a vertical line (m=inf).
    # Since we typically want to rotate the image *back* to vertical, we use -angle_deg.
    # The actual angle may need sign inversion depending on the coordinate system (image rows vs y-axis).
    # For now, let's return the angle of the line w.r.t the x-axis, which is often used.
    # We'll rotate by -angle_deg.
    return angle_deg
def apply_centroid_correction(xtcavImages, off_idx, steps_if_stepwise=None, do_rotate=False):
    """
    Applies centroid correction or rigid rotation to a set of XTCAV images.
    
    (Docstring truncated for brevity)
    """
    N_shots = xtcavImages.shape[2]
    Nrows = xtcavImages.shape[0]
    xtcavImages_list_new = [None] * N_shots # Initialize lists to maintain order
    LPSImage_new = [None] * N_shots
    
    # --- 1. ROTATION MODE (Updated for Stepwise) ---
    if do_rotate:
        print("--- Entering Rigid Rotation Mode ---")
        
        # Dictionary to store the correction angle calculated for each unique step
        step_angles_map = {}
        
        if steps_if_stepwise is None:
            # 1A. NON-STEPWISE ROTATION (Original logic)
            print("Calculating global rotation angle...")
            rotation_angles = []
            
            # Calculate angle for all OFF images
            for idx in off_idx:
                angle = get_image_angle_from_line_fit(xtcavImages[:, :, idx])
                rotation_angles.append(90 + angle)

            avg_angle = np.mean(rotation_angles) if rotation_angles else 0.0
            correction_angle = -avg_angle
            
            # Map this single angle to a 'None' step key for uniform processing later
            step_angles_map[None] = correction_angle
            
            print(f"Applying uniform rotation of: {correction_angle:.2f} deg.")
            
            # Prepare the final return array of corrections (all shots get the same angle)
            correction_return_value = np.full(N_shots, correction_angle)

        else:
            # 1B. STEPWISE ROTATION
            print("Calculating stepwise rotation angles...")
            unique_steps = np.unique(steps_if_stepwise)
            
            correction_return_value = np.zeros(N_shots) # Angle applied to each shot
            
            for step in unique_steps:
                # Find indices belonging to this step that are also 'off_idx'
                step_off_indices = [i for i in off_idx if steps_if_stepwise[i] == step]
                
                step_rotation_angles = []
                for idx in step_off_indices:
                    angle = get_image_angle_from_line_fit(xtcavImages[:, :, idx])
                    step_rotation_angles.append(90 + angle)
                    
                # Calculate average angle for this step
                avg_angle_step = np.mean(step_rotation_angles) if step_rotation_angles else 0.0
                correction_angle_step = -avg_angle_step # The angle to rotate by
                step_angles_map[step] = correction_angle_step
                
                print(f"Step {step}: Calculated average angle {avg_angle_step:.2f} deg. Applying correction {correction_angle_step:.2f} deg.")
                
                # Store the correction angle for all shots belonging to this step
                for idx in range(N_shots):
                    if steps_if_stepwise[idx] == step:
                        correction_return_value[idx] = correction_angle_step

        # 1C. APPLY the calculated rotation to ALL images
        for idx in range(N_shots):
            
            # Determine which correction angle to use
            if steps_if_stepwise is None:
                # Use the single calculated angle
                correction_angle = step_angles_map[None]
            else:
                # Use the angle calculated for the image's specific step
                current_step = steps_if_stepwise[idx]
                correction_angle = step_angles_map.get(current_step, 0.0) # Default to 0 if step not found

            processed_image = xtcavImages[:, :, idx]

            # Use scipy.ndimage.rotate for rotation with interpolation (Cubic order=3)
            rotated_image = rotate(
                processed_image,
                angle=correction_angle,
                reshape=False,
                order=3,
                mode='constant',
                cval=0.0
            )

            # Store results in the correct order
            rotated_image_3d = rotated_image[:, :, np.newaxis] # Reshape to (H, W, 1)
            xtcavImages_list_new[idx] = rotated_image_3d
            LPSImage_new[idx] = [rotated_image.ravel()]
            
        print("Completed rigid rotation for all images.")

 # --- 2. CENTROID CORRECTION MODE (Original Logic) ---
    else:
        print("--- Entering Centroid Correction Mode ---")
        
        if steps_if_stepwise is None:
            # Non-stepwise correction
            centroid_corrections = construct_centroid_function(xtcavImages, off_idx)
            
            # Iterate over all shots
            for idx in range(N_shots):
                processed_image = xtcavImages[:, :, idx]
                corrected_image = np.zeros_like(processed_image)
                for row in range(Nrows):
                    shift = int(centroid_corrections[row]) # Ensure shift is integer for np.roll
                    corrected_image[row, :] = np.roll(processed_image[row, :], shift)

                corrected_image = corrected_image[:, :, np.newaxis]
                image_ravel = corrected_image.ravel()

                xtcavImages_list_new.append(corrected_image)
                LPSImage_new.append([image_ravel])
                
            # Prepare return value for non-stepwise mode
            correction_return_value = np.tile(centroid_corrections[:, np.newaxis], (1, N_shots)).T

        else:
            # Stepwise correction
            unique_steps = np.unique(steps_if_stepwise)
            all_corrections = {} # Store corrections by index
            
            for step in unique_steps:
                # Find the indices corresponding to this step that are also 'off_idx'
                step_off_indices = [i for i in off_idx if steps_if_stepwise[i] == step]
                
                # Construct centroid function for this step
                centroid_corrections_step = construct_centroid_function(xtcavImages, step_off_indices)
                
                # Apply correction for all shots belonging to this step
                for idx in range(N_shots):
                    if steps_if_stepwise[idx] == step:
                        processed_image = xtcavImages[:, :, idx]
                        corrected_image = np.zeros_like(processed_image)
                        
                        for row in range(Nrows):
                            shift = int(centroid_corrections_step[row]) # Ensure shift is integer
                            corrected_image[row, :] = np.roll(processed_image[row, :], shift)

                        corrected_image = corrected_image[:, :, np.newaxis]
                        image_ravel = corrected_image.ravel()

                        xtcavImages_list_new.append(corrected_image)
                        LPSImage_new.append([image_ravel])
                        all_corrections[idx] = centroid_corrections_step

            # Sort the results by the original shot index (since stepwise iteration is out of order)
            # This is necessary because the lists were appended out of order.
            
            # 1. Create a list of tuples: (original_index, corrected_image, LPSImage_ravel)
            # This requires a more complex structure, as the original loop order is lost in the provided snippet.
            # A common fix is to ensure the loop over N_shots is the primary one.
            
            # --- RE-IMPLEMENTATION FOR STEPWISE TO PRESERVE ORDER ---
            # Instead of appending in the inner loop, store by index and then reorder.
            
            # Since the original stepwise logic iterates over steps and then all shots,
            # we need to rebuild the lists in the correct (0 to N_shots-1) order.
            
            xtcavImages_ordered = [None] * N_shots
            LPSImage_ordered = [None] * N_shots
            corrections_ordered = [None] * N_shots
            
            for step in unique_steps:
                step_off_indices = [i for i in off_idx if steps_if_stepwise[i] == step]
                centroid_corrections_step = construct_centroid_function(xtcavImages, step_off_indices)
                
                for idx in range(N_shots):
                    if steps_if_stepwise[idx] == step:
                        processed_image = xtcavImages[:, :, idx]
                        corrected_image = np.zeros_like(processed_image)
                        
                        for row in range(Nrows):
                            shift = int(centroid_corrections_step[row])
                            corrected_image[row, :] = np.roll(processed_image[row, :], shift)

                        # Store in the correct index
                        xtcavImages_ordered[idx] = corrected_image[:, :, np.newaxis]
                        LPSImage_ordered[idx] = [corrected_image.ravel()]
                        corrections_ordered[idx] = centroid_corrections_step
                        
            xtcavImages_list_new = xtcavImages_ordered
            LPSImage_new = LPSImage_ordered
            correction_return_value = np.array(corrections_ordered) # Already in (N_shots, H) shape
    # --- 3. FINAL CONCATENATION AND RETURN ---
    
    # Check if the lists were populated (they should be, either by rotation or correction)
    if xtcavImages_list_new[0] is not None:
        xtcavImages_ret = np.concatenate(xtcavImages_list_new, axis=2)
        LPSImage = np.concatenate(LPSImage_new, axis=0)
    else:
        # Handle the edge case where N_shots > 0 but no processing was done (e.g., in a minimal test)
        xtcavImages_ret = xtcavImages
        LPSImage = None # Need actual flattened LPS
        
    return xtcavImages_ret, None, LPSImage, correction_return_value
    """
    Applies centroid correction or rigid rotation to a set of XTCAV images.

    Args:
        xtcavImages (np.ndarray): 3D array of shape (H, W, N) containing XTCAV images.
        off_idx (list or np.ndarray): Indices of images to use for correction/rotation.
        steps_if_stepwise (list or np.ndarray, optional): Step number of each image for stepwise correction. Defaults to None.
                                                          When None, the same correction, inferred from all 'off_idx' images, is applied to all images regardless of step.
        do_rotate (bool): If True, performs **rigid rotation** instead of centroid correction.
                          It calculates the average rotation angle from 'off_idx' images and applies it to all images.

    Returns:
        tuple: Corrected/Rotated xtcavImages, None (horz_proj), flattened LPSImage, and the corrections applied (shift array or rotation angle).
    """

    N_shots = xtcavImages.shape[2]
    Nrows = xtcavImages.shape[0]
    xtcavImages_list_new = []
    LPSImage_new = []
    
    # --- 1. ROTATION MODE ---
    if do_rotate:
        print("--- Entering Rigid Rotation Mode ---")
        rotation_angles = []

        # 1a. Calculate the rotation angle for each OFF image
        for idx in off_idx:
            # Fit a line to the feature in the OFF image and get its angle
            angle = get_image_angle_from_line_fit(xtcavImages[:, :, idx])
            rotation_angles.append(angle)

        # 1b. Calculate the average rotation angle
        if not rotation_angles:
            print("Warning: No valid off_idx images found for rotation angle calculation. Using 0 degrees.")
            avg_angle = 0.0
        else:
            avg_angle = np.mean(rotation_angles)
        
        # The angle to rotate by is the negative of the feature's angle to make it vertical
        correction_angle = -avg_angle
        print(f"Calculated average rotation angle from OFF images: {avg_angle:.2f} deg.")
        print(f"Applying correction rotation of: {correction_angle:.2f} deg.")
        
        # Store the correction angle for the return value
        correction_return_value = np.array([correction_angle]) # A single value for all shots

        # 1c. Apply the rotation to ALL images (on and off)
        for idx in range(N_shots):
            processed_image = xtcavImages[:, :, idx]

            # Use scipy.ndimage.rotate for rotation with interpolation
            # order=3 is cubic interpolation, reshape=False preserves the shape
            rotated_image = rotate(
                processed_image,
                angle=correction_angle,
                reshape=False,
                order=3, # Cubic interpolation
                mode='constant',
                cval=0.0
            )

            # Reshape rotated_image for concatenation
            rotated_image = rotated_image[:, :, np.newaxis] # Reshape to (H, W, 1)

            # --- Prepare LPS Image (flattened) ---
            image_ravel = rotated_image.ravel()

            # --- Combine results into lists ---
            xtcavImages_list_new.append(rotated_image)
            LPSImage_new.append([image_ravel])
            
        print("Completed rigid rotation for all images.")

    # --- 2. CENTROID CORRECTION MODE (Original Logic) ---
    else:
        print("--- Entering Centroid Correction Mode ---")
        
        if steps_if_stepwise is None:
            # Non-stepwise correction
            centroid_corrections = construct_centroid_function(xtcavImages, off_idx)
            
            # Iterate over all shots
            for idx in range(N_shots):
                processed_image = xtcavImages[:, :, idx]
                corrected_image = np.zeros_like(processed_image)
                for row in range(Nrows):
                    shift = int(centroid_corrections[row]) # Ensure shift is integer for np.roll
                    corrected_image[row, :] = np.roll(processed_image[row, :], shift)

                corrected_image = corrected_image[:, :, np.newaxis]
                image_ravel = corrected_image.ravel()

                xtcavImages_list_new.append(corrected_image)
                LPSImage_new.append([image_ravel])
                
            # Prepare return value for non-stepwise mode
            correction_return_value = np.tile(centroid_corrections[:, np.newaxis], (1, N_shots)).T

        else:
            # Stepwise correction
            unique_steps = np.unique(steps_if_stepwise)
            all_corrections = {} # Store corrections by index
            
            for step in unique_steps:
                # Find the indices corresponding to this step that are also 'off_idx'
                step_off_indices = [i for i in off_idx if steps_if_stepwise[i] == step]
                
                # Construct centroid function for this step
                centroid_corrections_step = construct_centroid_function(xtcavImages, step_off_indices)
                
                # Apply correction for all shots belonging to this step
                for idx in range(N_shots):
                    if steps_if_stepwise[idx] == step:
                        processed_image = xtcavImages[:, :, idx]
                        corrected_image = np.zeros_like(processed_image)
                        
                        for row in range(Nrows):
                            shift = int(centroid_corrections_step[row]) # Ensure shift is integer
                            corrected_image[row, :] = np.roll(processed_image[row, :], shift)

                        corrected_image = corrected_image[:, :, np.newaxis]
                        image_ravel = corrected_image.ravel()

                        xtcavImages_list_new.append(corrected_image)
                        LPSImage_new.append([image_ravel])
                        all_corrections[idx] = centroid_corrections_step

            # Sort the results by the original shot index (since stepwise iteration is out of order)
            # This is necessary because the lists were appended out of order.
            
            # 1. Create a list of tuples: (original_index, corrected_image, LPSImage_ravel)
            # This requires a more complex structure, as the original loop order is lost in the provided snippet.
            # A common fix is to ensure the loop over N_shots is the primary one.
            
            # --- RE-IMPLEMENTATION FOR STEPWISE TO PRESERVE ORDER ---
            # Instead of appending in the inner loop, store by index and then reorder.
            
            # Since the original stepwise logic iterates over steps and then all shots,
            # we need to rebuild the lists in the correct (0 to N_shots-1) order.
            
            xtcavImages_ordered = [None] * N_shots
            LPSImage_ordered = [None] * N_shots
            corrections_ordered = [None] * N_shots
            
            for step in unique_steps:
                step_off_indices = [i for i in off_idx if steps_if_stepwise[i] == step]
                centroid_corrections_step = construct_centroid_function(xtcavImages, step_off_indices)
                
                for idx in range(N_shots):
                    if steps_if_stepwise[idx] == step:
                        processed_image = xtcavImages[:, :, idx]
                        corrected_image = np.zeros_like(processed_image)
                        
                        for row in range(Nrows):
                            shift = int(centroid_corrections_step[row])
                            corrected_image[row, :] = np.roll(processed_image[row, :], shift)

                        # Store in the correct index
                        xtcavImages_ordered[idx] = corrected_image[:, :, np.newaxis]
                        LPSImage_ordered[idx] = [corrected_image.ravel()]
                        corrections_ordered[idx] = centroid_corrections_step
                        
            xtcavImages_list_new = xtcavImages_ordered
            LPSImage_new = LPSImage_ordered
            correction_return_value = np.array(corrections_ordered) # Already in (N_shots, H) shape

    # --- 3. FINAL CONCATENATION AND RETURN ---
    
    # Only concatenate if the lists are populated (which they should be)
    if xtcavImages_list_new and LPSImage_new:
        xtcavImages_ret = np.concatenate(xtcavImages_list_new, axis=2)
        LPSImage = np.concatenate(LPSImage_new, axis=0)
    else:
        # Should only happen if N_shots=0
        return xtcavImages, None, None, np.array([])
    
    return xtcavImages_ret, None, LPSImage, correction_return_value