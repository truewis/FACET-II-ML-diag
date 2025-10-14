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
CONFIG = {
    # Base directory for DAQ data (e.g., '/nas/nas-li20-pm00/TEST')
    "DAQ_BASE_PATH": "data/raw/",
    # The specific DAQ folder suffix
    "DAQ_PATH_SUFFIX": "TEST/2023/20230901/TEST_03748"
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
def initialize_and_allocate(data_struct, image_base_path, C_0based):
    """
    Initializes image processing by loading the first image (from .h5 or .tif),
    determines the final processed dimensions, and pre-allocates an array for all images.
    """
    # MATLAB: imgLoc = ['/Users/cemma/',data_struct.images.PR10711.loc{1}];
    # The 'loc' field holds a list of relative paths. loc{1} is the first shot.
    # NOTE: The logic has been adapted to handle either a single H5 file path
    # or a path to the first TIFF in a sequence.
    first_img_relative_path = truncate_path_from_substring(data_struct['loc'][0], 'images/')
    first_imgLoc_path = Path(image_base_path) / first_img_relative_path

    # MATLAB: img = imgfull(1:10:491,50:300); downsample and crop
    # Parameters for processing
    HOT_PIX_THRESHOLD = 20
    SIGMA = 1
    THRESHOLD = 5
    
    imgfull_first = None

    # Read, Rotate, Downsample, and Crop the FIRST image to get the final flattened size
    try:
        print(f"Attempting to load initial image from: {first_imgLoc_path}")
        file_extension = first_imgLoc_path.suffix.lower()

        # Check for HDF5 format
        if file_extension == '.h5':
            print("Detected HDF5 format.")
            import h5py
            with h5py.File(first_imgLoc_path, 'r') as f:
                # Find the primary dataset (assuming the largest array is the image stack)
                dataset_sizes = {name: d.size for name, d in f.items() if hasattr(d, 'shape')}
                if not dataset_sizes:
                    raise ValueError(f"No datasets found in H5 file: {first_imgLoc_path}")
                dataset_key = max(dataset_sizes, key=dataset_sizes.get)
                print(f"Reading from dataset: '{dataset_key}'")
                
                # Load the first image from the stack (index 0)
                img_data = f[dataset_key][0]
                # Rotate 180 degrees using numpy
                imgfull_first = np.rot90(img_data, 2)

        # Handle standard image formats (default case)
        else:
            print("Assuming standard image format (e.g., TIFF, PNG).")
            from PIL import Image
            # Load and rotate using Pillow
            imgfull_first = np.array(Image.open(first_imgLoc_path).rotate(180))

    except FileNotFoundError:
        print(f"Error: Initial image file not found at {first_imgLoc_path}")
        return None, None
    except (ImportError, ValueError, NotImplementedError, IOError) as e:
        print(f"An error occurred while reading the image: {e}")
        return None, None

    # --- The rest of the processing pipeline remains unchanged ---

    # Downsample and crop (MATLAB 1:10:491, 50:300)
    # Python slices are [start:stop:step] and [start:stop)
    # 1:10:491  (1-based, inclusive end) -> [0:491:10] (0-based, exclusive end)
    # 50:300 (1-based, inclusive end) -> [49:300] (0-based, exclusive end)
    
    print("Cropping and downsampling the first image.")
    img_first_cropped = imgfull_first[0:491:10, 49:300]
    
    # Process the first image to get the size of the *processed* data
    imgProc_first = processNoisyTCAVImage.processNoisyTCAVImage(img_first_cropped, HOT_PIX_THRESHOLD, SIGMA, THRESHOLD)
    
    # Calculate the size for pre-allocation
    rows, cols = imgProc_first.shape
    N_shots = len(C_0based)
    N_pixels = rows * cols
    
    # Pre-allocate the final array
    lpsFlattened = np.zeros((N_shots, N_pixels), dtype=np.float64)
    print(f'Pre-allocating lpsFlattened: {N_shots} shots x {N_pixels} pixels ({lpsFlattened.nbytes / 1e6:.2f} MB)')

    return lpsFlattened, N_pixels
#%%
def analyze_daq_images(daq_path_suffix: str, daq_base_path: str, opts=None):
    """
    Loads DAQ data, extracts BSA scalars, processes images, and flattens them.

    Args:
        daq_path_suffix (str): The specific DAQ folder suffix 
                               (e.g., '/TEST/2023/20230901/TEST_03748').
        daq_base_path (str): The root path where DAQ data is stored 
                             (e.g., '/Users/cemma/nas/nas-li20-pm00').
        opts (dict, optional): Options struct, currently unused in Python equivalent.

    Returns:
        tuple: (lpsFlattened, data_struct, save_filename)
               - lpsFlattened (np.ndarray): Flattened processed image data (N_shots x N_pixels).
               - data_struct (dict): The loaded DAQ data.
               - save_filename (str): The suggested filename for saving.
    """
    if opts is None:
        # Default options mirror the MATLAB script
        opts = {'usemethod': 2}
    
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
        PR10711 = data_struct.images.DTOTR2 # Adjusted to match the actual field name in the MATLAB struct 
        
        # Find matching timestamps between the cameras
        C = PR10711.common_index # C is the 1-based index from MATLAB
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
    first_img_relative_path = truncate_path_from_substring(PR10711.loc[C_0based[0]], 'images/')
    first_imgLoc_path = Path(image_base_path) / first_img_relative_path

    # MATLAB: img = imgfull(1:10:491,50:300); downsample and crop
    # Parameters for processing
    HOT_PIX_THRESHOLD = 20
    SIGMA = 1
    THRESHOLD = 5
    
    # Read, Rotate, Downsample, and Crop the FIRST image to get the final flattened size
    try:
        print(f"Attempting to load initial image from: {first_imgLoc_path}")
        file_extension = first_imgLoc_path.suffix.lower()

        # Check for HDF5 format
        if file_extension == '.h5':
            print("Detected HDF5 format.")
            import h5py
            with h5py.File(first_imgLoc_path, 'r') as f:
                # Find the primary dataset (assuming the largest array is the image stack)
                dataset_sizes = {name: d.size for name, d in f.items() if hasattr(d, 'shape')}
                if not dataset_sizes:
                    raise ValueError(f"No datasets found in H5 file: {first_imgLoc_path}")
                dataset_key = max(dataset_sizes, key=dataset_sizes.get)
                print(f"Reading from dataset: '{dataset_key}'")
                
                # Load the first image from the stack (index 0)
                img_data = f[dataset_key][0]
                # Rotate 180 degrees using numpy
                imgfull_first = np.rot90(img_data, 2)

        # Handle standard image formats (default case)
        else:
            print("Assuming standard image format (e.g., TIFF, PNG).")
            from PIL import Image
            # Load and rotate using Pillow
            imgfull_first = np.array(Image.open(first_imgLoc_path).rotate(180))

    except FileNotFoundError:
        print(f"Error: Initial image file not found at {first_imgLoc_path}")
        return None, None
    except (ImportError, ValueError, NotImplementedError, IOError) as e:
        print(f"An error occurred while reading the image: {e}")
        return None, None

    # Downsample and crop (MATLAB 1:10:491, 50:300)
    # Python slices are [start:stop:step] and [start:stop)
    # 1:10:491  (1-based, inclusive end) -> [0:491:10] (0-based, exclusive end)
    # 50:300 (1-based, inclusive end) -> [49:300] (0-based, exclusive end)
    
    img_first_cropped = imgfull_first[0:491:10, 49:300]
    
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
        img_relative_path = truncate_path_from_substring(PR10711.loc[current_C_idx], 'images/')
        imgLoc_path = Path(image_base_path) / img_relative_path

        # 2. Read and Rotate the image
        # MATLAB: imrotate(imread(imgLoc),180)
        try:
            imgfull = np.array(Image.open(imgLoc_path).rotate(180))
        except FileNotFoundError:
            print(f"Warning: Image file not found for shot {n} at {imgLoc_path}. Skipping.")
            continue # Skip this shot and leave the row as zeros

        # 3. Downsample and Crop
        # MATLAB: img = imgfull(1:10:491,50:300); 
        img = imgfull[0:491:10, 49:300]

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
if len(sys.argv) == 3:
    daq_base_path = sys.argv[1]
    daq_path_suffix = sys.argv[2]
    lpsFlattened, data_struct, save_filename = analyze_daq_images(daq_path_suffix, daq_base_path)
    if lpsFlattened is not None:
        print(f"Data processed successfully. Suggested save filename: {save_filename}")
        savemat(save_filename, {'lpsFlattened': lpsFlattened})
# If no arguments are provided, use the CONFIG dictionary
else:   
    #Main Function using CONFIG paths
    lpsFlattened, data_struct, save_filename = analyze_daq_images(
            CONFIG["DAQ_PATH_SUFFIX"],
            CONFIG["DAQ_BASE_PATH"]
    )
    if lpsFlattened is not None:
        print(f"Data processed successfully. Suggested save filename: {save_filename}")
        savemat(save_filename, {'lpsFlattened': lpsFlattened})