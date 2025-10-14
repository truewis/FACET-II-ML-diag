import numpy as np
import os
import re
from pathlib import Path
from scipy.io import loadmat, savemat # To load the MATLAB .mat file
from PIL import Image # For reading/rotating/resizing images

# ======================================================================
# NOTE: You MUST define these functions and constants for the code to run.
# ======================================================================

# 1. External functions
import extractDAQBSAScalars 
import processNoisyTCAVImage
# 2. Path Sanitiation
# Use a configuration dictionary for paths instead of hardcoding.
# You'll need to update these paths for your actual environment.
CONFIG_2 = {
    # Base directory for DAQ data (e.g., '/nas/nas-li20-pm00/TEST')
    "DAQ_BASE_PATH": "data/raw/",
    # The specific DAQ folder suffix
    "DAQ_PATH_SUFFIX": "TEST/2023/20230901/TEST_03748",
    "INSTRUMENT": "PR10711",# The camera/instrument name in the DAQ data structure
    
    "X_CROP": slice(49,300),# The image crop slice in X (0-based indexing)
    "Y_CROP": slice(0,491,10),# The image crop slice in Y (0-based indexing)
}
CONFIG = {
    # Base directory for DAQ data (e.g., '/nas/nas-li20-pm00/TEST')
    "DAQ_BASE_PATH": "data/raw/",
    # The specific DAQ folder suffix
    "DAQ_PATH_SUFFIX": "E300/E300_12427",
    "INSTRUMENT": "DTOTR2",# The camera/instrument name in the DAQ data structure
    
    "X_CROP": slice(20,272),# The image crop slice in X (0-based indexing)
    "Y_CROP": slice(0,821,20), # The image crop slice in Y (0-based indexing)
}

# 3. Colormap Placeholder (if you use matplotlib for plotting)
# In Python, you'd typically use a library like Matplotlib for plotting.
# The MATLAB 'colormap jetvar' is not a standard Matplotlib colormap.
# You'd use 'jet' or 'viridis' or define a custom one.
#%% ======================================================================
def truncate_path_from_substring(full_path: str, start_substring: str) -> str:
    """
    Truncates a file path so that it begins exactly at the first occurrence 
    of the specified substring.

    Args:
        full_path (str): The original, complete file path.
        start_substring (str): The substring (e.g., 'images/') to start 
                                the new path from.
                                This must not be absolute (i.e., no leading slash).

    Returns:
        str: The truncated path, or the original path if the substring isn't found.
    """
    try:
        # Find the starting index of the desired substring
        start_index = full_path.index(start_substring)
        
        # Slice the string from that index to the end
        truncated_path = full_path[start_index:]
        
        return truncated_path
        
    except ValueError:
        print(f"Warning: Substring '{start_substring}' not found in the path.")
        return full_path
    
#%% ======================================================================

def load_image_transparently(data_struct, image_base_path, common_index):
    """
    Loads a single image from either a .h5 stack or a sequence of .tif files.

    Args:
        data_struct (dict): Dictionary containing 'loc' and optionally 'step' keys.
        image_base_path (str): The base directory for the image data.
        common_index (int): The 0-based index of the image to load from the overall sequence.

    Returns:
        np.ndarray: The loaded and rotated image as a NumPy array, or None on failure.
    """
    try:
        # Determine file format from the first entry in 'loc'
        first_loc_path = Path(data_struct.loc[0])
        file_extension = first_loc_path.suffix.lower()

        # Handle HDF5 format
        if file_extension == '.h5':
            import h5py
            # Determine the correct H5 file using the 'step' array
            target_file_num = data_struct.step[common_index] # e.g.,1-5
            
            h5_relative_path = truncate_path_from_substring(data_struct.loc[target_file_num-1], 'images/')
            h5_full_path = Path(image_base_path) / h5_relative_path

            # Compute the relative index of the image within its H5 file
            steps_array = np.array(data_struct.step)
            # Count how many shots with this file number occurred up to the common_index
            relative_index = np.count_nonzero(steps_array[:common_index + 1] == target_file_num) - 1

            print(f"Loading H5 image: file {target_file_num}, relative index {relative_index}")
            with h5py.File(h5_full_path, 'r') as f:
                
                img_data = f['entry']['data']['data'][relative_index]
                return np.rot90(img_data, 2)

        # Handle standard image formats (e.g., TIFF)
        else:
            img_relative_path = truncate_path_from_substring(data_struct.loc[common_index], 'images/')
            img_full_path = Path(image_base_path) / img_relative_path
            
            print(f"Loading TIFF image from: {img_full_path}")
            return np.array(Image.open(img_full_path).rotate(180))

    except FileNotFoundError:
        print(f"Error: Image file not found.")
        return None
    except (ImportError, ValueError, NotImplementedError, IOError, IndexError) as e:
        print(f"An error occurred while reading the image: {e}")
        return None

#%% ======================================================================
def analyze_daq_images(daq_path_suffix: str, daq_base_path: str, instrument: str, x_crop: slice, y_crop: slice):
    """
    Loads DAQ data, extracts BSA scalars, processes images, and flattens them.

    Args:
        daq_path_suffix (str): The specific DAQ folder suffix 
                               (e.g., '/TEST/2023/20230901/TEST_03748').
        daq_base_path (str): The root path where DAQ data is stored 
                             (e.g., '/Users/cemma/nas/nas-li20-pm00').

    Returns:
        tuple: (lpsFlattened, data_struct, save_filename)
               - lpsFlattened (np.ndarray): Flattened processed image data (N_shots x N_pixels).
               - data_struct (dict): The loaded DAQ data.
               - save_filename (str): The suggested filename for saving.
    """
    
    # --- Step 1: Construct Paths and Load DAQ .mat file ---
    
    # The MATLAB logic: load([DAQpath,'/',DAQpath(51:end),'.mat'])
    # DAQpath is daq_base_path + daq_path_suffix, and (51:end) gets the run name.
    
    full_daq_dir = Path(daq_base_path) / daq_path_suffix.strip(os.sep)
    daq_run_name = full_daq_dir.name # Gets the last part, e.g., 'TEST_03748'
    mat_file_path = full_daq_dir / f'{daq_run_name}.mat'
    image_base_path = Path(daq_base_path) / daq_path_suffix.strip(os.sep)
    
    print(f"Loading data from: {mat_file_path}")
    
    # Load DAQ .mat file
    try:
        # squeeze_me=True helps convert 1-element arrays to scalars
        data_struct = loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)['data_struct']
        # loadmat returns a dict; we assume the main structure is named 'data_struct' inside.
        # struct_as_record=False returns object arrays for MATLAB structs, which is easier to access.
    except FileNotFoundError:
        print(f"Error: .mat file not found at {mat_file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None, None, None

    # --- Step 2: Initialize and Check Common Indices ---

    # data_struct.images.PR10711 access is complex due to loadmat's object array handling.
    # We navigate to the PR10711 structure.
    try:
        # Accessing nested fields from the loaded MATLAB structure object (using ._content for consistency)
        INSTRUMENT = data_struct.images.__getattribute__(instrument)
         # Adjusted to match the actual field name in the MATLAB struct 
        
        # Find matching timestamps between the cameras
        C = INSTRUMENT.common_index # C is the 1-based index from MATLAB
        C_0based = np.array(C) - 1 # Convert to 0-based Python indices
        
    except AttributeError:
        print("Error: Could not find PR10711 structure in DAQ data.")
        # Print the available fields for debugging
        print(f"Available fields in data_struct.images: {data_struct.images._fieldnames}")
        return None, None, None
    except Exception as e:
        print(f"Error accessing common index: {e}")
        return None, None, None

    # Load DAQ BSA scalar Data Nvars x Nshots
    bsaScalarData, bsaVars = extractDAQBSAScalars.extractDAQBSAScalars(data_struct)

    # Error/Return checks
    if bsaScalarData.shape[1] != len(C_0based):
        # NOTE: Using len(C) for the comparison, as bsaScalarData is likely N_vars x N_shots (1-based count)
        print('Error: BSA scalar shot count does not match common index length.')
        return None, None, None
    
    print(f'Length C = {len(C)}')
    if len(C) == 0:
        return None, None, None

    # --- Step 3: Loop Prep and Get Initial Image Size ---
    
    # MATLAB: imgLoc = ['/Users/cemma/',data_struct.images.PR10711.loc{1}];
    # The 'loc' field holds a list of relative paths. loc{1} is the first shot.

    # MATLAB: img = imgfull(1:10:491,50:300); downsample and crop
    # Parameters for processing
    HOT_PIX_THRESHOLD = 20
    SIGMA = 1
    THRESHOLD = 5
    
    # Load the first image (index 0) using the new transparent loader function
    imgfull_first = load_image_transparently(INSTRUMENT, image_base_path, 0)

    if imgfull_first is None:
        print("Halting initialization because the first image could not be loaded.")
        return None, None

    # Downsample and crop (MATLAB 1:10:491, 50:300)
    # Python slices are [start:stop:step] and [start:stop)
    # 1:10:491  (1-based, inclusive end) -> [0:491:10] (0-based, exclusive end)
    # 50:300 (1-based, inclusive end) -> [49:300] (0-based, exclusive end)
    
    img_first_cropped = imgfull_first[x_crop, y_crop]
    
    # Process the first image to get the size of the *processed* data
    imgProc_first = processNoisyTCAVImage.processNoisyTCAVImage(img_first_cropped, HOT_PIX_THRESHOLD, SIGMA, THRESHOLD)
    
    # Calculate the size for pre-allocation
    rows, cols = imgProc_first.shape
    N_shots = len(C_0based)
    N_pixels = rows * cols
    
    # Pre-allocate the final array
    lpsFlattened = np.zeros((N_shots, N_pixels), dtype=np.float64)
    print(f'Pre-allocating lpsFlattened: {N_shots} shots x {N_pixels} pixels')

    # --- Step 4: Loop over all shots ---
    
    for n in range(N_shots):
        # 1. Get image location for the nth shot (using the common index C)
        current_C_idx = C_0based[n]

        # 2. Read and Rotate the image
        # MATLAB: imrotate(imread(imgLoc),180)
        try:
            imgfull = load_image_transparently(INSTRUMENT, image_base_path, current_C_idx)
        except Exception as e:
            print(f"Warning: {e} for shot {n}. Skipping.")
            continue # Skip this shot and leave the row as zeros

        # 3. Downsample and Crop
        # MATLAB: img = imgfull(1:10:491,50:300); 
        img = imgfull[x_crop, y_crop]

        # 4. Process the image
        # MATLAB: imgProc = processNoisyTCAVImage(img,20,1,5);
        imgProc = processNoisyTCAVImage.processNoisyTCAVImage(img, HOT_PIX_THRESHOLD, SIGMA, THRESHOLD)

        # 5. Flatten and store
        # MATLAB: lpsFlattened(n,:) = reshape(imgProc,1,[]);
        # Python uses n for the 0-based index
        lpsFlattened[n, :] = imgProc.flatten()
        
        # MATLAB: n (print index)
        if (n + 1) % 100 == 0 or n == N_shots - 1:
            print(f'Processed shot {n + 1}/{N_shots}')


    # --- Step 5: Final Output ---
    
    # Suggested save filename
    save_filename = f'lpsFlattened_{daq_run_name}.mat'
    
    # NOTE: The MATLAB code includes: save('lpsFlattened_TEST_03748','lpsFlattened')
    # If you wish to save the data in the Python function, use:
    # from scipy.io import savemat
    # savemat(save_filename, {'lpsFlattened': lpsFlattened})
    
    return lpsFlattened, data_struct, save_filename

#If two arguments are provided, run the main function with those arguments
import sys
if len(sys.argv) == 4:
    daq_base_path = sys.argv[1]
    daq_path_suffix = sys.argv[2]
    instrument = sys.argv[3]
    print(f"Running with provided arguments:\n Base Path: {daq_base_path}\n Path Suffix: {daq_path_suffix}\n Instrument: {instrument}")
    #Warn that cropping parameters are set in the CONFIG dictionary
    print("Note: Image cropping parameters (X_CROP, Y_CROP) are set in the CONFIG dictionary.")
    lpsFlattened, data_struct, save_filename = analyze_daq_images(daq_path_suffix, daq_base_path, instrument, CONFIG["X_CROP"], CONFIG["Y_CROP"])
    if lpsFlattened is not None:
        print(f"Data processed successfully. Suggested save filename: {save_filename}")
        savemat(save_filename, {'lpsFlattened': lpsFlattened})
# If no arguments are provided, use the CONFIG dictionary
else:   
    #Main Function using CONFIG paths
    lpsFlattened, data_struct, save_filename = analyze_daq_images(
            CONFIG["DAQ_PATH_SUFFIX"],
            CONFIG["DAQ_BASE_PATH"],
            CONFIG["INSTRUMENT"],
            CONFIG["X_CROP"],
            CONFIG["Y_CROP"]
    )
    if lpsFlattened is not None:
        print(f"Data processed successfully. Suggested save filename: {save_filename}")
        savemat(save_filename, {'lpsFlattened': lpsFlattened})