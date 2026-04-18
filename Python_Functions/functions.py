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
import os
import re

def get_min_max_daq(date_path, experiment):
    """
    Helper function to find the minimum and maximum daq_num 
    currently existing in a given date directory.
    """
    print(f"  [DEBUG] get_min_max_daq: Checking directory -> {date_path}")
    if not os.path.exists(date_path):
        print(f"  [DEBUG] get_min_max_daq: Path does not exist. Returning None, None.")
        return None, None
    
    items = os.listdir(date_path)
    daq_nums = []
    # Pattern to match exactly: {experiment}_{daq_num}
    pattern = re.compile(rf"^{re.escape(experiment)}_(\d+)$")
    
    for item in items:
        match = pattern.match(item)
        if match:
            daq_nums.append(int(match.group(1)))
            
    if not daq_nums:
        print(f"  [DEBUG] get_min_max_daq: No matching files found. Returning None, None.")
        return None, None
        
    min_val, max_val = min(daq_nums), max(daq_nums)
    print(f"  [DEBUG] get_min_max_daq: Found {len(daq_nums)} items. Min DAQ = {min_val}, Max DAQ = {max_val}")
    return min_val, max_val

def facet_daq_path(experiment, daq_num, base_path='/nas/nas-li20-pm00/'):
    """
    Finds the path for a specific experiment and daq_num.
    Searches the 4 most recent dates first, then uses binary search.
    Expects structure: {base_path}/{experiment}/YYYY/YYYYMMDD/
    """
    print(f"\n[DEBUG] --- Starting path_fun ---")
    print(f"[DEBUG] Inputs: base_path='{base_path}', experiment='{experiment}', daq_num='{daq_num}'")
    
    exp_path = os.path.join(base_path, experiment)
    if not os.path.exists(exp_path):
        print(f"[DEBUG] Experiment path '{exp_path}' does not exist. Returning None.")
        return None
        
    # 1. Gather all date directories by checking YYYY then YYYYMMDD
    date_dirs = []
    
    print(f"[DEBUG] Scanning for YYYY and YYYYMMDD directories in '{exp_path}'...")
    for year_dir in os.listdir(exp_path):
        year_path = os.path.join(exp_path, year_dir)
        
        # Check if the item is a 4-digit directory (YYYY)
        if re.match(r"^\d{4}$", year_dir) and os.path.isdir(year_path):
            for date_dir in os.listdir(year_path):
                # Check if the item is an 8-digit directory (YYYYMMDD) and starts with the year
                if re.match(r"^\d{8}$", date_dir) and date_dir.startswith(year_dir):
                    if os.path.isdir(os.path.join(year_path, date_dir)):
                        date_dirs.append(date_dir)
            
    # Sort descending so the most recent dates are at the beginning
    date_dirs.sort(reverse=True)
    print(f"[DEBUG] Found {len(date_dirs)} date directories (sorted newest to oldest): {date_dirs}")
    
    target_name = f"{experiment}_{daq_num}"
    target_daq = int(daq_num) 
    
    # 2. Linear search on the 4 most recent dates
    print(f"\n[DEBUG] --- Phase 1: Linear search on up to 4 most recent dates ---")
    for date_dir in date_dirs[:4]:
        year_dir = date_dir[:4] # Extract YYYY from YYYYMMDD
        target_path = os.path.join(exp_path, year_dir, date_dir, target_name)
        print(f"[DEBUG] Checking path: {target_path}")
        if os.path.exists(target_path):
            print(f"[DEBUG] SUCCESS: File found during linear search! Returning path.")
            return target_path+f'/{experiment}_{daq_num}.mat'
    # 3. Binary search on the remaining dates
    remaining_dates = date_dirs[4:]
    if not remaining_dates:
        print(f"[DEBUG] No remaining older dates to search. Returning None.")
        return None
        
    # Sort remaining dates ascending (oldest to newest) for standard binary search
    remaining_dates.sort()
    print(f"\n[DEBUG] --- Phase 2: Binary search on remaining older dates ---")
    print(f"[DEBUG] Dates for binary search (oldest to newest): {remaining_dates}")
    
    low = 0
    high = len(remaining_dates) - 1
    
    step = 1
    while low <= high:
        mid = (low + high) // 2
        mid_date = remaining_dates[mid]
        year_dir = mid_date[:4] # Extract YYYY from YYYYMMDD
        mid_date_path = os.path.join(exp_path, year_dir, mid_date)
        
        print(f"\n[DEBUG] Binary Search Step {step}:")
        print(f"[DEBUG]   low_index={low}, high_index={high}, mid_index={mid}")
        print(f"[DEBUG]   Evaluating mid_date -> {mid_date} (Path: {mid_date_path})")
        
        # Check if the specific target exists in this mid_date directory
        target_path = os.path.join(mid_date_path, target_name)
        print(f"[DEBUG]   Checking direct path: {target_path}")
        
        if os.path.exists(target_path):
            print(f"[DEBUG]   SUCCESS: File found during binary search! Returning path.")
            return target_path+f'/{experiment}_{daq_num}.mat'
            
        # If it doesn't exist, figure out whether to go left (older) or right (newer)
        min_daq, max_daq = get_min_max_daq(mid_date_path, experiment)
        
        if min_daq is None or max_daq is None:
            print(f"[DEBUG]   Directory is empty. Shifting search to older dates (high = mid - 1).")
            high = mid - 1
            step += 1
            continue
            
        print(f"[DEBUG]   Comparing target_daq ({target_daq}) against bounds [{min_daq}, {max_daq}]")
        
        if target_daq < min_daq:
            print(f"[DEBUG]   Target ({target_daq}) < Min ({min_daq}). Shifting search to OLDER dates (high = {mid - 1}).")
            high = mid - 1
        elif target_daq > max_daq:
            print(f"[DEBUG]   Target ({target_daq}) > Max ({max_daq}). Shifting search to NEWER dates (low = {mid + 1}).")
            low = mid + 1
        else:
            print(f"[DEBUG]   Target ({target_daq}) is between Min ({min_daq}) and Max ({max_daq}), but file is missing.")
            print(f"[DEBUG]   This implies the file does not exist in the expected sequence. Returning None.")
            return None
            
        step += 1
            
    # File was not found anywhere
    print(f"\n[DEBUG] --- Search Complete: File not found anywhere. Returning None. ---")
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
						 EOS_cal=17.94e-15, directory_path=None):
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
	EOSdata, _, _, _ = extract_processed_images(data_struct, experiment, xrange=None, yrange=None, hotPixThreshold=1e4, sigma=1, threshold=5, step_list=None, roi_xrange=None, roi_yrange=None, do_load_raw = False, instrument = 'EOS2', directory_path = directory_path, intermediate_datatype = np.uint16)
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
			# Zero out all points below 9.5
			log_proj[log_proj < 9.5] = 0.0
			# 1. Smooth the data to stabilize gradient and peak finding
			smoothed = gaussian_filter(log_proj, sigma=0.8)
			peaks, props = find_peaks(smoothed, height=9.2, prominence=0.03)
			
			if peaks.size == 0:
			    dels[a] = 0.0
			    continue
			
			#%% ######################################################################################
			# Step 1: Find the Separation/Split Point
			split_idx = None
			
			if peaks.size >= 2:
			    # We found at least 2 distinct peaks. Take the top 2 by height.
			    top2_idx = np.argsort(props['peak_heights'])[-2:]
			    p1, p2 = np.sort(peaks[top2_idx])
			    
			    # Split the domain exactly between the two peaks
			    split_idx = int((p1 + p2) / 2)
			
			elif peaks.size == 1:
			    # Only 1 peak found. The peaks have merged. 
			    # We need to find the "shoulder" using the minimal absolute gradient.
			    main_peak = peaks[0]
			    main_peak_height = smoothed[main_peak]
			    
			    # Calculate absolute gradient
			    abs_grad = np.abs(np.gradient(smoothed))
			    
			    # Find local minima (valleys) in the gradient
			    valleys, _ = find_peaks(-abs_grad)
			    
			    # Filter out valleys that are part of the main peak's flat top (below 90% of peak height)
			    valid_valleys = [v for v in valleys if smoothed[v] < (0.90 * main_peak_height)]
			    
			    if valid_valleys:
			        # Pick the gradient minimum closest to the main peak
			        split_idx = min(valid_valleys, key=lambda v: abs(v - main_peak))
			    else:
			        # Fallback if no distinct shoulder is found
			        dels[a] = 0.0 
			        continue
			
			#%% ######################################################################################
			# Step 2: Calculate Weighted Centroids based on 80% Threshold
			x_coords = np.arange(len(log_proj))
			left_mask = x_coords < split_idx
			right_mask = x_coords >= split_idx
			
			# Isolate the two regions (using smoothed data to find the stable 80% boundary)
			left_proj_smooth = smoothed[left_mask]
			right_proj_smooth = smoothed[right_mask]
			
			# Ensure neither region is empty
			if len(left_proj_smooth) == 0 or len(right_proj_smooth) == 0:
			    dels[a] = 0.0
			    continue
			
			# --- Left Peak Weighted Centroid ---
			left_max = np.max(left_proj_smooth)
			left_thresh = 0.8 * left_max
			
			# Find indices where smoothed data is above threshold
			left_above_thresh_mask = left_proj_smooth >= left_thresh
			left_x_coords = x_coords[left_mask][left_above_thresh_mask]
			
			# Use the raw log_proj values as the weights for the centroid
			left_weights = log_proj[left_mask][left_above_thresh_mask]
			centroid_left = np.average(left_x_coords, weights=left_weights)
			
			# --- Right Peak Weighted Centroid ---
			right_max = np.max(right_proj_smooth)
			right_thresh = 0.8 * right_max
			
			# Find indices where smoothed data is above threshold
			right_above_thresh_mask = right_proj_smooth >= right_thresh
			right_x_coords = x_coords[right_mask][right_above_thresh_mask]
			
			# Use the raw log_proj values as the weights for the centroid
			right_weights = log_proj[right_mask][right_above_thresh_mask]
			centroid_right = np.average(right_x_coords, weights=right_weights)
			
			# Step 3: Calculate final separation
			peak_sep_pix = abs(centroid_right - centroid_left)
			
			#%% ######################################################################################
			# Step 4: Debug Plotting
			if debug and a % plot_frequency == 1:
			    plt.figure(figsize=(8, 4))
			    plt.plot(x_coords, log_proj, label='Raw Log Proj', alpha=0.5)
			    plt.plot(x_coords, smoothed, label='Smoothed', linewidth=2)
			    
			    # Mark the split point
			    plt.axvline(split_idx, color='r', linestyle='--', label='Separation Point')
			    
			    # Mark the 80% regions
			    plt.axvspan(left_x_coords.min(), left_x_coords.max(), color='g', alpha=0.2, label='Left 80% Region')
			    plt.axvspan(right_x_coords.min(), right_x_coords.max(), color='m', alpha=0.2, label='Right 80% Region')
			    
			    # Mark the calculated Weighted Centroids
			    plt.axvline(centroid_left, color='g', linestyle='-', label=f'W-Centroid 1: {centroid_left:.1f}')
			    plt.axvline(centroid_right, color='m', linestyle='-', label=f'W-Centroid 2: {centroid_right:.1f}')
			    
			    plt.title("80% Weighted Centroid Separation")
			    plt.legend()
			    plt.grid(True, alpha=0.3)
			    plt.show()
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
						 debug_plot_frequency=8,
						 step_selector=None,
						 min_prom = 2000,
						 find_FWHM = False
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
	# For analyses without two peaks, store first peak positions for diagnostics
	first_peaks = np.zeros(nshots, dtype=float)
	first_peak_fwhms = np.zeros(nshots, dtype=float)
	print("Starting SYAG analysis...")
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
			if debug and a % debug_plot_frequency == 1:
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
			if debug and a % debug_plot_frequency == 1:
				plt.plot(proj)
				plt.plot(peaks, proj[peaks], "x")
				plt.show()
			# The convention of x axis orientation of SYAG screen
			first_peaks[a] = len(proj) - peaks[0]
			if find_FWHM:
				# Find FWHM positions for first peak only for now.
				peak_height = props['prominences'][0]
				half_height = peak_height / 2
				# Smooth the projection for more reliable FWHM finding
				proj_smoothed = gaussian_filter(proj, sigma=2)
				# Find left FWHM
				left_idx = peaks[0]
				while left_idx > 0 and proj_smoothed[left_idx] > half_height:
					left_idx -= 1
				fwhm_left = left_idx
				# Find right FWHM
				right_idx = peaks[0]
				while right_idx < len(proj) - 1 and proj_smoothed[right_idx] > half_height:
					right_idx += 1
				fwhm_right = right_idx
				first_peak_fwhms[a] = fwhm_right - fwhm_left
			if peaks.size != 2:
				dels[a] = 0.0
				continue
			# Centroid analysis is optional.
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
		'first_peak_fwhsm': first_peak_fwhms,
		'first_peaks': first_peaks,
		'dels': dels,
		'bc14BLEN': bc14BLEN,
		'SYAGhorzProj': SYAGhorzProj,
		'fig': fig,
		'sorted_idx': sorted_idx
	}

# 2. Define the objective function for the optimizer
def waterfall_alignment_objective(params, xtcav_axis, syag_axis, xtcav_norm, syag_norm):
	scale, offset = params
		
	# Apply the linear transformation to the XTCAV pixel coordinates
	# mapped_xtcav = scale * original_xtcav + offset
	mapped_xtcav_axis = xtcav_axis * scale + offset
		
	# Interpolate the XTCAV data onto the fixed SYAG pixel grid
	# bounds_error=False and fill_value=0 ensure that data shifted off-screen just becomes 0
	interp_func = interp1d(
		mapped_xtcav_axis, 
		xtcav_norm, 
		axis=0, 
		bounds_error=False, 
		fill_value=0.0
	)
	mapped_xtcav_data = interp_func(syag_axis)
	
	# Calculate the 2D Pearson correlation coefficient across the flattened arrays
	corr = np.corrcoef(syag_norm.flatten(), mapped_xtcav_data.flatten())[0, 1]
	
	# If the arrays are entirely shifted out of bounds, correlation might be NaN
	if np.isnan(corr):
		return 0.0
	
	# We want to MAXIMIZE correlation, so we MINIMIZE the negative correlation
	return -corr

from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
def map_xtcav_to_syag(syag_waterfall, xtcav_waterfall):
	"""
	Finds the global linear offset and scaling to match XTCAV projections to SYAG projections.
	
	Args:
		syag_waterfall (np.ndarray): 2D array of SYAG projections (rows = pixels, cols = shots).
		xtcav_waterfall (np.ndarray): 2D array of XTCAV projections (rows = pixels, cols = shots).
		
	Returns:
		dict: Contains the optimal scale, offset, max correlation achieved, and the mapped XTCAV array.
	"""
	n_syag_pix, n_syag_shots = syag_waterfall.shape
	n_xtcav_pix, n_xtcav_shots = xtcav_waterfall.shape
	
	if n_syag_shots != n_xtcav_shots:
		raise ValueError(f"Shot counts must match! SYAG has {n_syag_shots}, XTCAV has {n_xtcav_shots}.")

	# 1. Normalize both waterfalls shot-by-shot to decouple the spatial correlation 
	# from pure charge/intensity variations between the two cameras.
	syag_norm = syag_waterfall / (np.sum(syag_waterfall, axis=0, keepdims=True) + 1e-9)
	xtcav_norm = xtcav_waterfall / (np.sum(xtcav_waterfall, axis=0, keepdims=True) + 1e-9)

	syag_axis = np.arange(n_syag_pix)
	xtcav_axis = np.arange(n_xtcav_pix)

	# 3. Define search bounds for [scale, offset]
	# Scale: assume XTCAV is anywhere from 25x smaller to 1x larger than SYAG
	# Offset: assume the shift could be up to the size of the SYAG screen
	bounds = [
		(5.0, 12.0), 
		(-n_syag_pix/2, n_syag_pix/2)
	]
	# Define the initial guess based on typical ROI
	initial_guess = [6.0, 100.0]
	
	print("Optimizing XTCAV to SYAG mapping. This may take a few seconds...")
	
	# 4. Run Global Optimization
	# seed=42 for reproducibility. workers=-1 uses all available CPU cores.
	opt_args = (xtcav_axis, syag_axis, xtcav_norm, syag_norm)
	result = differential_evolution(waterfall_alignment_objective, bounds, args=opt_args, seed=42, workers=-1, x0=initial_guess)
	best_scale, best_offset = result.x
	max_corr = -result.fun
	
	print(f"Match found! Scale: {best_scale:.4f}, Offset: {best_offset:.2f} pixels, Correlation: {max_corr:.4f}")
	
	# 5. Generate the final transformed XTCAV array using the best parameters
	final_mapped_axis = xtcav_axis * best_scale + best_offset
	final_interp_func = interp1d(
		final_mapped_axis, 
		xtcav_waterfall, # Interpolating the raw data this time, not the normalized data
		axis=0, 
		bounds_error=False, 
		fill_value=0.0
	)
	aligned_xtcav_waterfall = final_interp_func(syag_axis)
	# alignment_results = map_xtcav_to_syag(syag_waterfall, xtcav_waterfall)
	plot_alignment(syag_waterfall, aligned_xtcav_waterfall)
	return {
		'scale': best_scale,
		'offset': best_offset,
		'correlation': max_corr,
		#'aligned_xtcav': aligned_xtcav_waterfall
	}

import matplotlib.pyplot as plt

def plot_alignment(syag_waterfall, aligned_xtcav):
	"""
	Displays the SYAG waterfall and the mapped XTCAV waterfall side-by-side 
	for visual verification of the alignment.
	"""
	# Create a 1x2 subplot with shared axes for synchronized zooming
	fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
	
	# Plot 1: Original SYAG Waterfall
	im1 = axes[0].imshow(syag_waterfall, aspect='auto', origin='lower', cmap='viridis')
	axes[0].set_title('SYAG Waterfall (Reference)')
	axes[0].set_xlabel('Shot Number')
	axes[0].set_ylabel('Pixel Index')
	fig.colorbar(im1, ax=axes[0], label='Intensity', fraction=0.046, pad=0.04)
	
	# Plot 2: Aligned XTCAV Waterfall
	im2 = axes[1].imshow(aligned_xtcav, aspect='auto', origin='lower', cmap='viridis')
	axes[1].set_title('Aligned XTCAV Waterfall')
	axes[1].set_xlabel('Shot Number')
	# Y-label is omitted here since the axis is shared with the left plot
	fig.colorbar(im2, ax=axes[1], label='Intensity', fraction=0.046, pad=0.04)
	
	plt.tight_layout()
	plt.show()

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
	bsaVarPVNames = []

	for bsaName in isBSA:
		bsaList = dataScalars[bsaName]
		varNames = list(bsaList.keys())
		bsaListData = []

		for varName in varNames:
			varData = np.array(bsaList[varName]).squeeze()
			if varData.size == 0:
				continue
			if varData.ndim != 1:
				continue
			if filter_index:
				varData = varData[idx]  # apply common index. If this fails, this variable is not recorded in some of the steps.
			varData = np.nan_to_num(varData)  # replace NaN with 0
			# Dither by a small random value to avoid overfitting.
			# Suppose by 0.1% of the average value of the variable.
			avg_val = np.mean(varData)
			if avg_val != 0:
				dither_amplitude = 0.001 * abs(avg_val)
				varData = varData + np.random.uniform(-dither_amplitude, dither_amplitude, size=varData.shape)
			bsaListData.append(varData)

		# Add to output arrays
		bsaVarPVNames.extend([vn for vn in varNames if bsaList[vn].size != 0])
		if bsaListData:
			bsaScalarData.extend(bsaListData)

	bsaScalarData = np.array(bsaScalarData)
	return bsaScalarData, bsaVarPVNames
	
def extractDAQNonBSAScalars(data_struct, step_list=None, filter_index=True, debug=False, s20=False):
    """
    Extracts nonBSA scalar data from the provided data structure, filtered with scalar common index.
    Ensures final output is a homogeneous 2D numpy array shaped (N_vars, N_samples).
    """
    data_struct = matstruct_to_dict(data_struct)
    dataScalars = data_struct['scalars']
    
    if step_list is not None:
        idx = [i for i in dataScalars['common_index'].astype(int).flatten() - 1
               if dataScalars['steps'][i] in step_list]
    else:
        idx = dataScalars['common_index'].astype(int).flatten() - 1

    fNames = list(dataScalars.keys())
    if s20:
        isBSA = [name for name in fNames if name.startswith('nonBSA_List_S20')]
    else:
        isBSA = [name for name in fNames if name.startswith('nonBSA_List_')]

    nonBsaScalarData = []
    nonBsaVarPVNames = []

    for nonbsaListName in isBSA:
        bsaList = dataScalars[nonbsaListName]
        varNames = list(bsaList.keys())
        bsaListData = []

        for varName in varNames:
            varData = np.array(bsaList[varName]).squeeze()
            
            # FIXED: Do not flatten 2D arrays. Slice the primary dimension instead.
            if varData.ndim > 1:
                # If shape is (2, N_samples) take the first row. 
                # If (N_samples, 2) take the first column.
                if varData.shape[0] < varData.shape[1]:
                    varData = varData[0, :]
                else:
                    varData = varData[:, 0]
                    
            varData = np.atleast_1d(varData)
            
            if varData.size == 0:
                continue
                
            if filter_index:
                valid_idx = [i for i in idx if i < len(varData)]
                varData = varData[valid_idx]  
                
            varData = np.nan_to_num(varData)
            # Dither by a small random value to avoid overfitting.
            # Suppose by 0.1% of the average value of the variable.
            # But don't do dithering, sbst reading of -10000 implies error, should not dither
            avg_val = np.mean(varData)
            if avg_val != 0:
                dither_amplitude = 0 # 0.001 * abs(avg_val)
                varData = varData + np.random.uniform(-dither_amplitude, dither_amplitude, size=varData.shape)
            bsaListData.append(varData)

        nonBsaVarPVNames.extend([vn for vn in varNames if bsaList[vn].size != 0])
        if bsaListData:
            nonBsaScalarData.extend(bsaListData)

    # ==========================================
    # FIXED: Enforce Homogeneity safely
    # ==========================================
    if nonBsaScalarData:
        # 1. Find the most common length among all variables (the true N_samples)
        lengths = [len(arr) for arr in nonBsaScalarData]
        target_len = max(set(lengths), key=lengths.count) 
        
        cleaned_data = []
        for arr in nonBsaScalarData:
            if len(arr) > target_len:
                # Truncate oversampled arrays (e.g. 120Hz data or unflattened arrays)
                cleaned_data.append(arr[:target_len])
            elif len(arr) < target_len:
                # Pad undersampled arrays (dropped packets)
                cleaned_data.append(np.pad(arr, (0, target_len - len(arr)), mode='constant', constant_values=0))
            else:
                cleaned_data.append(arr)
                
        nonBsaScalarData = cleaned_data
        
    nonBsaScalarData = np.array(nonBsaScalarData)

    if debug:
        print(f"Target N_samples (mode length): {target_len if nonBsaScalarData.size else 0}")
        print(f"Final output matrix shape: {nonBsaScalarData.shape}")

    return nonBsaScalarData, nonBsaVarPVNames

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
        'TCAV_LI20_2400_ADES',  # XTCAV
		'TCAV_LI20_2400_PDES',  # XTCAV
        'TCAV_LI20_2400_C_1_TCTL',  # XTCAV
		'TCAV_LI20_2400_S_AV',  # XTCAV
        'TCAV_LI20_2400_S_PV',  # XTCAV
        'PMT_LI20_3070_QDCRAW', # Unknown position PMT, exclude it for now.
		'SIOC_SYS1_ML00_AO542',
		'SIOC_SYS1_ML00_AO543',
		'SIOC_SYS1_ML00_AO544', #Outputs of this program
		
	]
	# Search for variable names that contain four digit number larger than 3100, this is after the XTCAV.
	for var in bsaVars:
		match = re.search(r'(\d{4})', var)
		if match:
			number = int(match.group(1))
			if number >= 3100:
				exclude_bsa_vars.append(var)
		
		# skip variables starting with 'SIOC', as they are unphysical
		if var.startswith('SIOC'):
			exclude_bsa_vars.append(var)
        # skip variables ending with 'BDES', as they are accompanied with BACT
		if var.endswith('BDES'):
			exclude_bsa_vars.append(var)
		if var.endswith('PDES'):
			exclude_bsa_vars.append(var)
		if var.endswith('ADES'):
			exclude_bsa_vars.append(var)
		if var.endswith('BCON'):
			exclude_bsa_vars.append(var)

	#print("Excluding BSA Variables:", exclude_bsa_vars)
	excluded_var_idx = [i for i, var in enumerate(bsaVars) if var in exclude_bsa_vars]
	#print("Excluded variable indices:", excluded_var_idx)
	return excluded_var_idx

import numpy as np

def sanitize_FACET_input(predictor, var_names):
    """
    Sanitizes the predictor matrix by handling -10000 read failures in LIXX:SBST:1:PHAS variables.
    
    Args:
        predictor (np.ndarray): The 2D array of predictor values (samples x features).
        var_names (list): The list of feature names corresponding to the columns.
        
    Returns:
        np.ndarray: The sanitized predictor matrix.
    """
    # Create a copy to avoid unintended side effects on the original array
    sanitized_predictor = predictor.copy()
    n_samples, n_features = sanitized_predictor.shape
    
    # 1. Identify indices for target variables in L2 and L3 groups
    l2_names = ['LI12:SBST:1:PHAS', 'LI13:SBST:1:PHAS', 'LI14:SBST:1:PHAS']
    l3_names = ['LI15:SBST:1:PHAS', 'LI16:SBST:1:PHAS', 'LI17:SBST:1:PHAS', 
                'LI18:SBST:1:PHAS', 'LI19:SBST:1:PHAS']
    
    # Safely get column indices if the variables exist in the cleaned dataset
    l2_indices = [var_names.index(name) for name in l2_names if name in var_names]
    l3_indices = [var_names.index(name) for name in l3_names if name in var_names]
    
    # Include LI11 if it exists, primarily for the multi-sample forward fill
    all_target_indices = l2_indices + l3_indices
    if 'LI11:SBST:1:PHAS' in var_names:
        all_target_indices.append(var_names.index('LI11:SBST:1:PHAS'))

    # ---------------------------------------------------------
    # Scenario 1: Multiple Samples (Forward Fill / "Scroll Up")
    # ---------------------------------------------------------
    if n_samples > 1:
        for col_idx in all_target_indices:
            for row_idx in range(1, n_samples):
                if sanitized_predictor[row_idx, col_idx] == -10000:
                    # Replace with the value from the previous row
                    sanitized_predictor[row_idx, col_idx] = sanitized_predictor[row_idx - 1, col_idx]

    # ---------------------------------------------------------
    # Scenario 2: Single Sample (Adjacent Fallback within Sections)
    # ---------------------------------------------------------
    elif n_samples == 1:
        
        def fill_adjacent(indices):
            if not indices: 
                return
            
            vals = sanitized_predictor[0, indices]
            # If the whole section is missing (-10000) or perfectly fine, do nothing
            if np.all(vals == -10000) or np.all(vals != -10000):
                return
            
            # Fill missing values with the closest adjacent valid phase
            for i, col_idx in enumerate(indices):
                if sanitized_predictor[0, col_idx] == -10000:
                    min_distance = float('inf')
                    best_replacement_val = -10000
                    
                    # Search for the nearest valid neighbor within this section
                    for j, other_col_idx in enumerate(indices):
                        if sanitized_predictor[0, other_col_idx] != -10000:
                            distance = abs(i - j)
                            if distance < min_distance:
                                min_distance = distance
                                best_replacement_val = sanitized_predictor[0, other_col_idx]
                    
                    sanitized_predictor[0, col_idx] = best_replacement_val

        # Apply the logic to L2 and L3 sections independently
        fill_adjacent(l2_indices)
        fill_adjacent(l3_indices)

    return sanitized_predictor


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
  

def extract_UVVisSpec(data_struct, step_list=None, directory_path = '/nas/nas-li20-pm00/'
							 ):
	"""
	Processes UVVisSpec images from HDF5 files, applying module-defined cropping, 
	filtering, and calculating horizontal projections (current profiles).

	Assumes the following are available in the current module's scope:
	- cropProfmonImg, median_filter, gaussian_filter (functions)
	- xrange, yrange, hotPixThreshold, sigma, threshold (constants)

	Args:
		data_struct (object): Structure containing image locations and backgrounds 
							  (e.g., data_struct.images.UVVisSpec.loc, data_struct.backgrounds.UVVisSpec).
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
		tuple: processed image array.
			   
	"""
	LPSImage = []
	stepsAll = data_struct.params.stepsAll
	step_size = None
	if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
		stepsAll = [1]
	steps_to_process = stepsAll if step_list is None else [s for s in stepsAll if s in step_list]
	images_each_step = []
	images_raw_each_step = []
	pid_list_each_step = []
	for a in range(1, np.max(steps_to_process)+1):
		xtcavImages_list = []
		xtcavImages_list_raw = []
		# --- Determine File Path ---
		if a in steps_to_process:
			width = 2
			UVdatalocation = str(directory_path) + '/' + f'images/UVVisSpec/UVVisSpec_data_step{a:0{width}d}.h5'

			# --- Read and Prepare Data ---
			with h5py.File(UVdatalocation, 'r') as f:
				data_raw = f['entry']['data']['data'][:].astype(np.float64) # shape: (N, H, W)
				pid_list = f["entry/instrument/NDAttributes/NDArrayUniqueId"][:].astype(np.int64)  # shape: (N,)
			
			# Transpose to shape: (H, W, N) - Height, Width, Shots
			UVdata_step = np.transpose(data_raw, (2, 1, 0))
			# Subtract background (H, W) from all shots (H, W, N)
			try:
				# If there is background data
				xtcavImages_step = UVdata_step - data_struct.backgrounds.UVVisSpec[:,:,np.newaxis].astype(np.float64)
			except:
				xtcavImages_step = UVdata_step
			step_size = UVdata_step.shape[2]
			# --- Process Individual Shots ---
			for idx in tqdm(range(step_size), desc="Processing Shots Step {}, {} samples".format(a, step_size)):
				if idx is None:
					continue
					
				image = xtcavImages_step[:,:,idx] # Single shot (H, W)
				raw_width = image.shape[1]
				raw_height = image.shape[0]
				
				# Calculate current profiles (Horizontal Projection)
				
				# Prepare for collection
				processed_image = image[:,:,np.newaxis]
				image_ravel = processed_image.ravel()
				xtcavImages_list.append(processed_image)
				LPSImage.append([image_ravel]) 
			images_each_step.append(xtcavImages_list)
			pid_list_each_step.append(pid_list)
		else:
			print(f"Skipping step {a} as it's not in steps_to_process.")
			images_each_step.append([])
			pid_list_each_step.append([])
	# --- Apply Common Indexing ---
	xtcavImages = []
	xtcavImages_raw = []
	for step in steps_to_process:
		#Find data_struct.scalars.steps row numbers that correspond to this step
		row_indices = [i for i, ci in enumerate(data_struct.scalars.common_index) if data_struct.scalars.steps[ci-1] == step]
		# Now find the image common indices that correspond to these row indices
		img_common_indices = data_struct.images.UVVisSpec.common_index[row_indices] - 1  # Convert to zero-based
		img_hdf5_pids = data_struct.images.UVVisSpec.pid[img_common_indices]
		# reconstruct hdf5 indices by matching pids
		img_hdf5_indices = [np.where(pid_list_each_step[step-1] == pid)[0][0] for pid in img_hdf5_pids]
		for i in img_hdf5_indices:
			xtcavImages.append(images_each_step[step-1][i])
	return np.array(xtcavImages)[:,:,:,0]

def extract_processed_images(data_struct, experiment, xrange=100, yrange=100, hotPixThreshold=1e3, sigma=1, threshold=5, step_list=None,
							 roi_xrange=None, roi_yrange=None, do_load_raw = False, directory_path = '/nas/nas-li20-pm00/', instrument = 'DTOTR2', intermediate_datatype = np.uint16
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
	stepsAll = data_struct.params.stepsAll
	step_size = None
	raw_width = None
	raw_height = None
	if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
		stepsAll = [1]
	steps_to_process = stepsAll if step_list is None else [s for s in stepsAll if s in step_list]
	images_each_step = []
	images_raw_each_step = []
	pid_list_each_step = []
	for a in range(1, np.max(steps_to_process)+1):
		xtcavImages_list = []
		xtcavImages_list_raw = []
		# --- Determine File Path ---
		if a in steps_to_process:
			width = 2
			DTOTR2datalocation = str(directory_path) + '/' + f'images/{instrument}/{instrument}_data_step{a:0{width}d}.h5'

			# --- Read and Prepare Data ---
			with h5py.File(DTOTR2datalocation, 'r') as f:
				data_raw = f['entry']['data']['data'][:].astype(intermediate_datatype) # shape: (N, H, W)
				pid_list = f["entry/instrument/NDAttributes/NDArrayUniqueId"][:].astype(np.int64)  # shape: (N,)
			
			# Transpose to shape: (H, W, N) - Height, Width, Shots
			DTOTR2data_step = np.transpose(data_raw, (2, 1, 0))
			# Subtract background (H, W) from all shots (H, W, N)
			try:
				# If there is background data
				xtcavImages_step = DTOTR2data_step - data_struct.backgrounds[instrument][:,:,np.newaxis].astype(intermediate_datatype)
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
				if xrange is not None and yrange is not None:
					image_cropped, _ = cropProfmonImg(image, xrange, yrange, plot_flag=False)
				else:
					image_cropped = image
				
				# Filter and mask hot pixels
				img_filtered = median_filter(image_cropped, size=3)
				hotPixels = img_filtered > hotPixThreshold
				img_filtered = np.ma.masked_array(img_filtered, hotPixels).astype(np.float64)
				
				# Gaussian smoothing and thresholding
				processed_image = gaussian_filter(img_filtered, sigma=sigma)
				processed_image[processed_image < threshold] = 0.0
				
				# Calculate current profiles (Horizontal Projection)
				
				# Prepare for collection
				processed_image = processed_image[:,:,np.newaxis]
				xtcavImages_list.append(processed_image)
			images_each_step.append(xtcavImages_list)
			pid_list_each_step.append(pid_list)
			if do_load_raw:
				images_raw_each_step.append(xtcavImages_list_raw)
		else:
			print(f"Skipping step {a} as it's not in steps_to_process.")
			images_each_step.append([])
			pid_list_each_step.append([])
			if do_load_raw:
				images_raw_each_step.append([])
	# --- Apply Common Indexing ---
	print("StepsToProcess:"+str(steps_to_process))
	xtcavImages = []
	xtcavImages_raw = []
	for step in steps_to_process:
		#Find data_struct.scalars.steps row numbers that correspond to this step
		row_indices = [i for i, ci in enumerate(data_struct.scalars.common_index) if data_struct.scalars.steps[ci-1] == step]
		# Now find the image common indices that correspond to these row indices
		img_common_indices = getattr(data_struct.images, instrument).common_index[row_indices] - 1  # Convert to zero-based
		img_hdf5_pids = getattr(data_struct.images, instrument).pid[img_common_indices]
		# reconstruct hdf5 indices by matching pids
		img_hdf5_indices = [np.where(pid_list_each_step[step-1] == pid)[0][0] for pid in img_hdf5_pids]
		for i in img_hdf5_indices:
			xtcavImages.append(images_each_step[step-1][i])
			if do_load_raw:
				xtcavImages_raw.append(images_raw_each_step[step-1][i])
	if do_load_raw:
		return np.array(xtcavImages)[:,:,:,0].transpose(1,2,0), np.array(xtcavImages_raw)[:,:,:,0].transpose(1,2,0), None, None
	else:
		return np.array(xtcavImages)[:,:,:,0].transpose(1,2,0), None, None, None

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
def apply_centroid_correction(xtcavImages, off_idx, steps_if_stepwise=None, do_rotate=False, do_substepwise=True):
	
	"""
	Applies centroid correction or rigid rotation to a set of XTCAV images.

	Args:
		xtcavImages (np.ndarray): 3D array of shape (H, W, N) containing XTCAV images.
		off_idx (list or np.ndarray): Indices of images to use for correction/rotation.
		steps_if_stepwise (list or np.ndarray, optional): Step number of each image for stepwise correction. Defaults to None.
														  When None, the same correction, inferred from all 'off_idx' images, is applied to all images regardless of step.
		do_rotate (bool): If True, performs **rigid rotation** instead of centroid correction.
						  It calculates the average rotation angle from 'off_idx' images and applies it to all images.
		do_substepwise (bool): If True, performs sub-stepwise correction/rotation within each step.
	Returns:
		tuple: Corrected/Rotated xtcavImages, None (horz_proj), flattened LPSImage, and the corrections applied (shift array or rotation angle).
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
		# Stepwise / Substep correction
		unique_steps = np.unique(steps_if_stepwise)
		
		# Pre-allocate lists to maintain the exact shot order
		xtcavImages_ordered = [None] * N_shots
		LPSImage_ordered = [None] * N_shots
		corrections_ordered = [None] * N_shots
		
		for step in unique_steps:
			# 1. Identify all shots and off_shots belonging to this specific step
			step_shots = [idx for idx in range(N_shots) if steps_if_stepwise[idx] == step]
			step_off_indices = sorted([i for i in off_idx if steps_if_stepwise[i] == step])
			
			print(f"Step {step}: Total shots = {len(step_shots)}, Off shots = {len(step_off_indices)}")
			
			# 2. Decide between Normal Stepwise vs. Substep mode
			# Trigger substeps if > 200 shots AND we actually have off shots to cluster
			if len(step_shots) > 200 and len(step_off_indices) > 0 and do_substepwise:
				
				# --- A. Form Clusters ---
				clusters = []
				current_cluster = [step_off_indices[0]]
				
				for off_i in step_off_indices[1:]:
					# If the gap between off shots is 5 or less, add to current cluster
					if off_i - current_cluster[-1] <= 5:
						current_cluster.append(off_i)
					else:
						# Gap is larger than 5; finalize the current cluster and start a new one
						clusters.append(current_cluster)
						current_cluster = [off_i]
				clusters.append(current_cluster) # Add the final cluster
				
				# --- B. Calculate Midpoints ---
				midpoints = []
				for i in range(len(clusters) - 1):
					# Midpoint between the end of one cluster and the start of the next
					mid = (clusters[i][-1] + clusters[i+1][0]) // 2
					midpoints.append(mid)
					
				print(f"  -> Divided into {len(clusters)} substeps.")
				
				# --- C. Process Each Substep ---
				for k, cluster in enumerate(clusters):
					# Define absolute shot index boundaries for this substep
					lower_bound = midpoints[k-1] if k > 0 else -1
					upper_bound = midpoints[k] if k < len(midpoints) else N_shots + 1
					
					# Find shots within this step that fall into this substep boundary
					substep_shots = [idx for idx in step_shots if lower_bound < idx <= upper_bound]
					
					# Construct centroid function exclusively from this cluster's off shots
					centroid_corrections_substep = construct_centroid_function(xtcavImages, cluster)
					
					# Apply to substep shots
					for idx in substep_shots:
						processed_image = xtcavImages[:, :, idx]
						corrected_image = np.zeros_like(processed_image)
						
						for row in range(Nrows):
							shift = int(centroid_corrections_substep[row])
							corrected_image[row, :] = np.roll(processed_image[row, :], shift)
							
							# Applying the zero-fill logic you had in the non-stepwise block
							if shift > 0:
								corrected_image[row, :shift] = 0
							elif shift < 0:
								corrected_image[row, shift:] = 0
						
						corrected_image = corrected_image[:, :, np.newaxis]
						
						# Store in the correct index mapping
						xtcavImages_ordered[idx] = corrected_image
						LPSImage_ordered[idx] = [corrected_image.ravel()]
						corrections_ordered[idx] = centroid_corrections_substep

			else:
				# --- Standard Stepwise Mode (<= 200 shots or no off_shots) ---
				print(f"Step {step}: Using standard stepwise correction (no substeps).")
				# Construct centroid function for the entire step at once
				if len(step_off_indices) > 0:
					centroid_corrections_step = construct_centroid_function(xtcavImages, step_off_indices)
				else:
					# Fallback if a step has zero off shots (prevents crashing)
					# We use an array of zeros (no shift)
					centroid_corrections_step = np.zeros(Nrows) 
				
				for idx in step_shots:
					processed_image = xtcavImages[:, :, idx]
					corrected_image = np.zeros_like(processed_image)
					
					for row in range(Nrows):
						shift = int(centroid_corrections_step[row])
						corrected_image[row, :] = np.roll(processed_image[row, :], shift)
						
						if shift > 0:
							corrected_image[row, :shift] = 0
						elif shift < 0:
							corrected_image[row, shift:] = 0

					corrected_image = corrected_image[:, :, np.newaxis]
					
					# Store in the correct index mapping
					xtcavImages_ordered[idx] = corrected_image
					LPSImage_ordered[idx] = [corrected_image.ravel()]
					corrections_ordered[idx] = centroid_corrections_step

		# Final assignment
		xtcavImages_list_new = xtcavImages_ordered
		LPSImage_new = LPSImage_ordered
		correction_return_value = np.array(corrections_ordered) # Array shape (N_shots, H)
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

def get_fwhm(projection, peak_idx):
    """Calculates the Full Width at Half Maximum (FWHM) in pixels using linear interpolation."""
    peak_val = projection[peak_idx]
    half_max = peak_val / 2.0
    N = len(projection)
    
    # Trace left
    left_idx = peak_idx
    while left_idx > 0 and projection[left_idx] > half_max:
        left_idx -= 1
        
    # Interpolate exact left crossing
    if left_idx < peak_idx:
        slope = projection[left_idx + 1] - projection[left_idx]
        left_cross = left_idx + (half_max - projection[left_idx]) / slope if slope != 0 else left_idx
    else:
        left_cross = left_idx

    # Trace right
    right_idx = peak_idx
    while right_idx < N - 1 and projection[right_idx] > half_max:
        right_idx += 1
        
    # Interpolate exact right crossing
    if right_idx > peak_idx:
        slope = projection[right_idx] - projection[right_idx - 1]
        right_cross = right_idx - 1 + (half_max - projection[right_idx - 1]) / slope if slope != 0 else right_idx
    else:
        right_cross = right_idx

    return right_cross - left_cross

def find_2d_mask_intervals(image_data, separation, ratio_bounds, max_fwhm_smaller, sep_tolerance=0):
    """
    Finds vertical slice intervals [a, b] and [c, end] such that the horizontal 
    projection of the masked 2D image satisfies specific peak constraints.
    """
    H, W = image_data.shape
    min_ratio, max_ratio = ratio_bounds
    
 
    # 1. Precompute projections
    w_proj = np.sum(image_data, axis=0) # Vertical projection (Width)
    total_charge = np.sum(w_proj)
    cumsum_w = np.cumsum(w_proj)
    
    # Precompute for O(1) horizontal projection (Time) sums
    cumsum_img = np.cumsum(image_data, axis=1)
    def get_h_proj(start, end):
        if start <= 0: return cumsum_img[:, end]
        return cumsum_img[:, end] - cumsum_img[:, start - 1]

    # 2. Find 'c' such that [c, W-1] contains 70% of the total charge
    # Search from the right to find the leftmost index that captures 70%
    target_charge = 0.70 * total_charge
    c = W - 1
    while c > 0 and (total_charge - (cumsum_w[c-1] if c > 0 else 0)) < target_charge:
        c -= 1
    print("c found:", c)
        
    h_proj_right = get_h_proj(c, W - 1)
    peak_right_idx = np.argmax(h_proj_right)
    print("peak right:", peak_right_idx)
    charge_right = np.sum(h_proj_right)
    while (charge_right > 0.4):
        
        h_proj_right = get_h_proj(c, W - 1)
        peak_right_idx = np.argmax(h_proj_right)
        print("peak right:", peak_right_idx)
        charge_right = np.sum(h_proj_right)
        # 3. Scan 'b' to satisfy the peak separation requirement
        # We look for a peak in [0, b] that is 'separation' away from peak_right_idx
        for b in range(1, c):
                h_proj_left_temp = get_h_proj(0, b)
                peak_left_idx = np.argmax(h_proj_left_temp)
                print("peak left:", peak_left_idx)
                dist = abs(peak_left_idx - peak_right_idx)
                print("dist:", dist)
                print("target sep:", separation)
                if abs(dist - separation) > sep_tolerance:
                        continue
                print("b found:", b)
                # 4. Scan 'a' to satisfy ratio and FWHM (Bunch Length)
                # We narrow the left window [a, b] to refine the "bunch"
                for a in range(0, b):
                        h_proj_left = get_h_proj(a, b)
                        charge_left = np.sum(h_proj_left)
                        
                        if charge_left <= 0: continue
                        
                        # Check Charge Ratio
                        ratio = charge_left / charge_right
                        if not (min_ratio <= ratio <= max_ratio or min_ratio <= (1/ratio) <= max_ratio):
                                continue
                        
                        # Check FWHM (Maximal bunch length)
                        # Use the smaller peak for the FWHM constraint
                        h_proj_combined = h_proj_left + h_proj_right
                        smaller_peak_idx = peak_left_idx if charge_left < charge_right else peak_right_idx
                        
                        fwhm = get_fwhm(h_proj_combined, smaller_peak_idx)
                        if fwhm <= max_fwhm_smaller:
                                return a, b, c
        c -= 3

    return None
