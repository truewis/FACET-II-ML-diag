import numpy as np
from scipy.ndimage import median_filter, gaussian_filter

def processNoisyTCAVImage(img, hot_pix_threshold, sigma, threshold):
    """
    Processes a noisy TCAV image by removing hot pixels and applying a threshold.
    
    Equivalent to the provided MATLAB function.
    
    :param img: 2D numpy array (the input image).
    :param hot_pix_threshold: Value above which a pixel is considered 'hot'.
    :param sigma: Standard deviation for the Gaussian filter (unused due to MATLAB's 'Don't do the blur' comment).
    :param threshold: Value below which processed pixels are set to 0.0.
    :return: 2D numpy array (the processed image).
    """

    # --- 1. Remove hot pixels (equivalent to MATLAB's medfilt2 and logical indexing) ---
    
    # Identify hot pixels
    hot_pixels = img > hot_pix_threshold
    
    # Apply a median filter to the whole image. 
    # The default kernel size for medfilt2 is 3x3.
    # We use a 3x3 median filter kernel here.
    filtered_img = median_filter(img, size=3)
    
    # Start with the median-filtered image as the corrected image
    corrected_image = filtered_img.copy() 
    
    # NOTE: The MATLAB logic 'correctedImage(hotPixels) = filteredImg(hotPixels)' is redundant 
    # because correctedImage is ALREADY a copy of filteredImg.
    # A common non-redundant MATLAB pattern would be:
    # correctedImage = img;
    # correctedImage(hotPixels) = filteredImg(hotPixels);
    # To be faithful to the *intent* of removing bad pixels, we will use the median-filtered
    # image for the rest of the processing and skip the redundant assignment.
    # We keep the name 'corrected_image' for clarity.
    
    corrected_image = filtered_img.copy()
    
    # --- 2. Apply Gaussian Blur (Equivalent to fspecial('gaussian') and imfilter) ---
    
    # MATLAB: processedImage = imfilter(correctedImage,gaussian_filter,'conv');
    # MATLAB: processedImage = correctedImage; % Don't do the blur
    
    # We follow the explicit MATLAB comment and skip the Gaussian blur.
    # The processedImage variable simply holds the corrected_image.
    processed_image = corrected_image 

    # If you later decide to enable the Gaussian blur, replace the line above with:
    # filter_size = 2 * np.ceil(3 * sigma).astype(int) + 1
    # gaussian_kernel = cv2.getGaussianKernel(filter_size, sigma)
    # gaussian_2D = gaussian_kernel @ gaussian_kernel.T # Create 2D kernel
    # processed_image = cv2.filter2D(corrected_image, -1, gaussian_2D)


    # --- 3. Apply thresholding (Equivalent to logical indexing) ---
    
    # MATLAB: processedImage(processedImage<threshold) = 0.0;
    
    # Create a boolean mask where the condition is met
    below_threshold_mask = processed_image < threshold
    
    # Apply the mask to set those pixels to 0.0
    processed_image[below_threshold_mask] = 0.0
    
    return processed_image