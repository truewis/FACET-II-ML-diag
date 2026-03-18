import numpy as np
import torch
from typing import Dict
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

LOG_CONVERSION_FACTOR=1e3
"""
Wrapper class for GMM conversion functions. This provides a consistent interface for converting between images and z parameters across different models.
"""
class SliceGMM():
    def __init__(self, xrange, yrange):
        self.xrange = xrange
        self.yrange = yrange
    def params_to_image(self, params, total_charge):
        """
        (mu_1, sigma_1, charge_1, mu_2, sigma_2, charge_2, ...) -> 2D image of shape (2* xrange, 2* yrange)
        """
        return slice_gaussian_image_from_flattened_params(params, total_charge, self.xrange, self.yrange)
class GeometricSliceScaler():
    """
    Scales geometric parameters (means and covariances) by a given range.
    Used for normalizing parameters for ML model training.
    """
    SPREAD_FACTOR = 5.0  # Factor to spread out the covariance values during normalization
    DISTANCE_FACTOR = 1.0 # Factor to scale the means during normalization

    def __init__(self, xrange, yrange):
        self.xrange = xrange
        self.yrange = yrange
        
    # Wrapper function because GeometricScaler does not fit anything.    
    def transform(self, flat_params):
        return self.fit_transform(flat_params)
        
    def fit_transform(self, flat_params):
        """
        Normalize flattened parameters (mu, sigma, charge) for ML model training.
        Expected input shape: (n_samples, 3*n_eslice) where each triplet is (mu, sigma, charge)
        """
        n_eslice = flat_params.shape[1] // 3
        flat_params = flat_params.reshape(-1, n_eslice, 3)
        
        mu = flat_params[:, :, 0]
        sigma = flat_params[:, :, 1]
        charge = flat_params[:, :, 2]
        
        mu_normalized = mu / self.xrange * self.DISTANCE_FACTOR
        sigma_normalized = sigma / self.xrange * self.SPREAD_FACTOR
        charge_normalized = charge  # Keep charge as-is
        
        # Stack back to (n_samples, 3*n_eslice)
        result = np.stack([mu_normalized, sigma_normalized, charge_normalized], axis=2)
        return result.reshape(result.shape[0], -1)
    
    def inverse_transform(self, flat_params):
        """
        Denormalize flattened parameters back to original scale.
        Expected input shape: (n_samples, 3*n_eslice)
        """
        n_eslice = flat_params.shape[1] // 3
        flat_params = flat_params.reshape(-1, n_eslice, 3)
        
        mu = flat_params[:, :, 0] * self.xrange / self.DISTANCE_FACTOR
        sigma = flat_params[:, :, 1] * self.xrange / self.SPREAD_FACTOR
        charge = flat_params[:, :, 2]
        
        result = np.stack([mu, sigma, charge], axis=2)
        return result.reshape(result.shape[0], -1)
    
def slice_gaussian_image_from_flattened_params(params, total_charge = None, xrange = 100, yrange = 100):
    """
    Generate a 2D image from flattened bi-Gaussian parameters. Also applies inverse operation of whatever pre-processing applied in image_to_bigaussian_params.
    Args:
        params (np.ndarray): Flattened parameters array of shape (3*n_eslice).
    Returns:
        np.ndarray: Generated 2D image of shape (2* xrange, 2* yrange).
    """
    
    # params are 3 * n_eslice, where each slice has (mu, sigma, charge) parameters. We need to reshape it to (n_eslice, 3) first.
    n_eslice = params.shape[0] // 3
    # Make sure the order is correct: (mu_1, sigma_1, charge_1, mu_2, sigma_2, charge_2, ...)
    params = params.reshape(n_eslice, 3)
    charge_per_eslice = params[:, 2] # Extract the charge for each slice
    image_size = (yrange, xrange)
    generated_image = np.zeros(image_size)
    # Take each vertical slice 
    for i in range(n_eslice):
        # Extract parameters for the current slice
        mu = params[i, 0]
        sigma = params[i, 1]
        # Generate the Gaussian profile for the current slice
        x = np.arange(xrange)
        gaussian_profile = gaussian_1d(x, mu, sigma, 1.0)
        # Add the profile to the generated image
        generated_image[:, i] = gaussian_profile * charge_per_eslice[i] * total_charge if charge_per_eslice is not None else gaussian_profile
        # Zoom the generated profile to fit the image size (if necessary)
        import scipy.ndimage
        generated_image[:, i] = scipy.ndimage.zoom(generated_image[:, i], yrange / generated_image[:, i].shape[0], order=1)
    return generated_image.T

import numpy as np

class SliceRegressorWrapper():
    def __init__(self, model, n_eslice=20):
        self.model = model
        self.n_eslice = n_eslice

    def predict(self, X):
        """
        Predicts (mu, sigma, charge) for each slice given input features X.
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features + n_eslice) where the last n_eslice features are the charge ratio [0, 1] per slice.
            Charge ratio must add up to 1.
        Returns:
            np.ndarray: Predicted parameters of shape (n_samples, 3 * n_eslice) where each triplet is (mu, sigma, charge) for each slice.
        """
        if X.shape[1] <= self.n_eslice:
            raise ValueError("Input X does not have enough features for charge per slice.")
            
        charge_per_eslice = X[:, -self.n_eslice:]
        # Make sure charge ratios add up to 1 for each sample, raise an error if not
        if not np.allclose(charge_per_eslice.sum(axis=1), 1.0):
            raise ValueError("Charge ratios for each sample must add up to 1.")
        X_modified = X[:, :-self.n_eslice]
        n_samples = X_modified.shape[0]
        
        # Expand X to be n_samples * n_eslice long
        X_modified = np.repeat(X_modified, self.n_eslice, axis=0)
        linear_feature = np.tile(np.arange(self.n_eslice) / self.n_eslice, n_samples)
        X_modified = np.hstack((X_modified, linear_feature.reshape(-1, 1)))
        
        # Predict mu and sigma: shape -> (n_samples * n_eslice, 2)
        predictions = self.model.predict(X_modified)
        
        # Reshape to combine with charge
        predictions = predictions.reshape(n_samples, self.n_eslice, 2)
        charge_reshaped = charge_per_eslice.reshape(n_samples, self.n_eslice, 1)
        
        # Combine into (mu, sigma, charge) triplets
        final_preds = np.concatenate((predictions, charge_reshaped), axis=2)
        
        # Return flattened array of shape (n_samples, 3 * n_eslice)
        return final_preds.reshape(n_samples, -1)
    
    def fit(self, X, y):
        if X.shape[1] <= self.n_eslice:
            raise ValueError("Input X does not have enough features for charge per slice.")
            
        charge_per_eslice = X[:, -self.n_eslice:]
        X_modified = X[:, :-self.n_eslice]
        n_samples = X_modified.shape[0]
        
        # Expand X to be n_samples * n_eslice long
        X_modified = np.repeat(X_modified, self.n_eslice, axis=0)
        linear_feature = np.tile(np.arange(self.n_eslice) / self.n_eslice, n_samples)
        X_modified = np.hstack((X_modified, linear_feature.reshape(-1, 1)))
        
        # y must be reshaped to (N * n_eslice, 2) so row counts perfectly match X_modified
        self.model.fit(X_modified, y.reshape(-1, 2))
        
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
# ----------------- Main Conversion Function -----------------

def image_to_bigaussian_params(target_image: np.ndarray, do_current_profile = False, debug = True) -> Dict:
    # Constants for a HxW image normalized to a range of [-1, 1]
    IMG_SIZE_X = target_image.shape[1]  # Assuming square image
    IMG_SIZE_Y = target_image.shape[0]  # Assuming square image
    K = 2 # Fixed number of components

    
    # If distances are greater than a threshold, we could assign to no component, which is -1.
    # For a noisy image, this helps avoid skewing Sigma estimates by only considering points close enough to any mean.
    DIST_THRESHOLD = 4 # This threshold can be tuned based on expected spread
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
    # Applying MSE to log is equivalent to performing negative log loss function, which is useful for predicting images with peaks whose amplitudes differ by orders of magnitudes.
    target_density = np.log1p(target_density * LOG_CONVERSION_FACTOR)  # log(1 + density * scale)

    y_coords = np.arange(IMG_SIZE_Y)
    x_coords = np.arange(IMG_SIZE_X)

    # --- Step 1: Vertical Projection and Y-Mean/Weight Estimation ---

    vertical_projection = target_density.max(axis=1) # Max across X-axis (200,)

    # Plot the whole image for debugging
    if debug:
        plt.figure()
        plt.imshow(target_density, cmap='viridis', aspect='auto')
        plt.title('Target Density Map')
        plt.colorbar(label='Density')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    try:
        peaks, properties = find_peaks(vertical_projection, distance=15, height=np.max(vertical_projection)*0.1, width = 3)

        if debug and len(peaks) < 2:
            print("Warning: Less than 2 peaks found in vertical projection.")
            plt.figure()
            plt.plot(y_coords, vertical_projection, label='Vertical Projection', color='blue')
            plt.scatter(y_coords[peaks], vertical_projection[peaks], color='red', label='Detected Peaks')
            plt.legend()
            plt.title('Vertical Projection with Detected Peaks')
            plt.xlabel('Y Coordinate')
            plt.ylabel('Density')
            plt.show()

        if len(peaks) < 2:
            raise RuntimeError("Less than 2 peaks found in vertical projection.")
        
        # If there are more than 2 peaks, the second peak must be at least twice the height of the third peak
        if len(peaks) > 2:
            sorted_peak_heights = np.sort(properties['peak_heights'])[::-1]
            if sorted_peak_heights[1] < 2 * sorted_peak_heights[2]:
                raise RuntimeError("Second peak is not sufficiently prominent compared to third peak.")
        # Extract fitted parameters
        mu_y = np.array([y_coords[peaks[0]], y_coords[peaks[1]]])
        pi_y_raw = np.array([properties['peak_heights'][0], properties['peak_heights'][1]])
        sigma_y = np.array([properties['widths'][0]/2, properties['widths'][1]/2]) # Ensure positive stddev

        if debug:
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
        if debug:
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
            raise RuntimeError(f"1D horizontal Gaussian fitting failed for component {k+1}.")

        means.append(np.array([mu_x_k, y_k]))
        sigma_x.append(sigma_x_k)

    if debug:
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
    # For a noisy image, this helps avoid skewing Sigma estimates by only considering points close enough to any mean.
    assigned_component[distances.min(axis=1) > DIST_THRESHOLD] = -1

    if debug:
        # Plot the assignment for debugging
        plt.figure()
        # Overlay original density as background
        plt.imshow(target_density, cmap='gray', alpha=0.3, extent=(0, IMG_SIZE_X, 0, IMG_SIZE_Y), origin='lower')
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=assigned_component, s=1, cmap='jet', alpha=0.2)
        plt.title('Pixel Assignment to GMM Components')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar(label='Assigned Component')
        plt.show()

    #In current profile mode, we are more interested in the fit of the zeta projection, rather than the 2D image itself.
    #Nudge the x means based on the current profile (horizontal projection of the entire image)
    if do_current_profile:
        if K is not 2:
            raise ValueError("In Current Profile Mode, K has to be 2.")
        else:
            try:
                mu_k_list = []
                sigma_k_list = []
                amp_k_list = []
                for k in range(K):
                    # Refit the current profile (horizontal projection of the component)
                    k_component_mask = (assigned_component == k)
                    current_profile = np.zeros(IMG_SIZE_X)
                    for idx in range(coordinates.shape[0]):
                        if k_component_mask[idx]:
                            x_idx = int(coordinates[idx, 0])
                            current_profile[x_idx] += flat_density[idx]
                    # Initial guess based on previous estimates
                    p0_x = [means[0][0], sigma_x[0], current_profile[int(means[0][0])]]
                    if debug:
                        print (f"Fitting Current Profile with initial guess {p0_x}")
                    popt_x, _ = curve_fit(gaussian_1d, x_coords, current_profile, p0=p0_x, maxfev=5000)
                    mu_x_k = popt_x[0]
                    sigma_x_k = np.abs(popt_x[1])  # Not used further but could be stored if needed
                    mu_k_list.append(mu_x_k)
                    sigma_k_list.append(sigma_x_k)
                    amp_k_list.append(popt_x[2])

                # Check if two peaks are sufficiently separated
                if np.abs(mu_k_list[0] - mu_k_list[1]) < 10:
                    raise RuntimeError("Fitted peaks in current profile are too close to each other.")
                # Overwrite the previously estimated means and sigma_x with the re-fitted values from the current profile
                means[0][0] = mu_k_list[0]
                means[1][0] = mu_k_list[1]
                sigma_x[0] = sigma_k_list[0]
                sigma_x[1] = sigma_k_list[1]
                new_popt_x = [mu_k_list[0], sigma_k_list[0], amp_k_list[0], mu_k_list[1], sigma_k_list[1], amp_k_list[1]]
                if debug:
                    print("Fitted Current Profile Means (Mu):", [mu_k_list[0], mu_k_list[1]])
                    plt.figure()
                    plt.plot(x_coords, current_profile, label='Current Profile', color='blue')
                    plt.plot(x_coords, bigaussian_1d(x_coords, *new_popt_x), label='Re-Fitted Bi-Gaussian', color='red')
                    plt.legend()
                    plt.title('Current Profile and Re-Fitted Bi-Gaussian')
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Density')
                    plt.show()
            except RuntimeError:
                # If fitting fails, retain the original estimates from the horizontal slices
                print(f"Current profile fitting failed for component {k+1}. Retaining original estimates.")
    
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

    if debug:
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
        
        if debug:
            plt.plot(ellipse_x, ellipse_y, label=f'Component {k+1} 1-sigma', linewidth=2)
    if debug:
        plt.title('Fitted GMM 1-Sigma Contours')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar(label='Density')
        plt.show()

    return {
        'pi': torch.tensor(pi_y_raw / np.sum(pi_y_raw), dtype=torch.float32), 
        'mu': torch.tensor(np.stack(means), dtype=torch.float32), 
        'Sigma': torch.tensor(np.stack(sigmas), dtype=torch.float32)
    }
