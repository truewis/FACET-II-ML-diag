import numpy as np
import torch
from typing import Dict
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

LOG_CONVERSION_FACTOR=1e3
class GeometricScaler():
    """
    Scales geometric parameters (means and covariances) by a given range.
    Used for normalizing parameters for ML model training.
    """
    SPREAD_FACTOR = 20.0  # Factor to spread out the covariance values during normalization
    DISTANCE_FACTOR = 1.0 # Factor to scale the means during normalization

    def __init__(self, xrange, yrange):
        self.xrange = xrange
        self.yrange = yrange
    def fit_transform(self, flat_params):
        pi = flat_params[:,:2]
        mu = flat_params[:,2:4]
        Sigma = flat_params[:,4:]
        pi_normalized = pi - 0.5
        # Below slices are to keep the dimensions correct for concatenation
        mu_normalized = np.concatenate([mu[:, 0:1] / self.xrange, mu[:, 1:2] / self.yrange], axis=1) * self.DISTANCE_FACTOR
        Sigma_normalized = np.concatenate([Sigma[:, 0:1] / (self.xrange **2), 
                                          Sigma[:, 1:2] / (self.xrange * self.yrange),
                                          Sigma[:, 2:3] / (self.yrange **2),
                                          Sigma[:, 3:4] / (self.xrange **2),
                                          Sigma[:, 4:5] / (self.xrange * self.yrange),
                                          Sigma[:, 5:6] / (self.yrange **2)], axis=1) * self.SPREAD_FACTOR
        return np.concatenate([pi_normalized, mu_normalized, Sigma_normalized], axis=1)
    def inverse_transform(self, flat_params):
        pi = flat_params[:,:2]
        # Below slices are to keep the dimensions correct for concatenation
        mu = np.concatenate([flat_params[:,2:3] * self.xrange, flat_params[:,3:4] * self.yrange], axis=1) / self.DISTANCE_FACTOR
        Sigma = np.concatenate([flat_params[:,4:5] / self.SPREAD_FACTOR * (self.xrange **2), 
                                flat_params[:,5:6] / self.SPREAD_FACTOR * (self.xrange * self.yrange),
                                flat_params[:,6:7] / self.SPREAD_FACTOR * (self.yrange **2),
                                flat_params[:,7:8] / self.SPREAD_FACTOR * (self.xrange **2),
                                flat_params[:,8:9] / self.SPREAD_FACTOR * (self.xrange * self.yrange),
                                flat_params[:,9:10] / self.SPREAD_FACTOR * (self.yrange **2)], axis=1)
        return np.concatenate([pi + 0.5, mu, Sigma], axis=1)
    
def biGaussian_image_from_flattened_params(params, total_charge = None, xrange = 100, yrange = 100):
    """
    Generate a 2D image from flattened bi-Gaussian parameters. Also applies inverse operation of whatever pre-processing applied in image_to_bigaussian_params.
    Args:
        params (np.ndarray): Flattened parameters array of shape (12,).
    Returns:
        np.ndarray: Generated 2D image of shape (200, 200).
    """
    # Unflatten parameters, ensuring Sigma off-diagonal elements are clipped because plotting hyperbolic contours is not useful
    uf = unflatten_biGaussian_params(params, clip_sigma=True)
    pi = uf['pi'].numpy()
    mu = uf['mu'].numpy().reshape(2,2)
    Sigma = uf['Sigma'].numpy()
    # Center of first Gaussian is at (xrange, yrange)
    generated_image = gaussian_2d_pdf(np.array([xrange,yrange]), Sigma[0], xrange, yrange) * pi[0] + \
                      gaussian_2d_pdf(np.array([xrange,yrange])+mu[1]-mu[0], Sigma[1], xrange, yrange) * pi[1]
    # WIP
    if total_charge is not None:
        generated_image = (np.exp(generated_image)-1)/LOG_CONVERSION_FACTOR
        generated_image = generated_image / np.sum(generated_image) * total_charge
    return generated_image.T

def flatten_biGaussian_params(biGaussian_params):
    """Flatten bi-Gaussian parameters into a 1D array.
    Sigma is assumed to be 2x2 for each component, but is a symmetric matrix.
    Hence, we omit the second off-diagonal element to reduce redundancy.
    Args:
        biGaussian_params (Dict): Dictionary with keys 'pi', 'mu', 'Sigma'. Output of image_to_bigaussian_params.
    Returns:
        np.ndarray: Flattened parameters array of shape (12,).
    """
    pi = biGaussian_params['pi'].numpy()
    mu = biGaussian_params['mu'].numpy().reshape(2,2)
    mu_diff = mu[1] - mu[0]  # Store difference of means
    Sigma = biGaussian_params['Sigma'].numpy()
    # Extract unique elements of Sigma (since it's symmetric)
    Sigma_flat = []
    for k in range(Sigma.shape[0]):
        Sigma_flat.extend([Sigma[k,0,0], Sigma[k,0,1], Sigma[k,1,1]])  # Omit Sigma[k,1,0] as it's equal to Sigma[k,0,1]
    Sigma_flat = np.array(Sigma_flat)
    return np.concatenate([pi, mu_diff, Sigma_flat])

def is_valid_biGaussian_params(flat_params, do_current_profile = False, xrange = 100, yrange = 100)-> bool:
    """
    Check if the flattened bi-Gaussian parameters are valid. Check includes:
    - No NaN values.
    - Mixture weights (pi) are non-negative.
    - Means (mu) are within [-2*xrange, 2*xrange].
    - Covariance matrices (Sigma) are positive definite and have positive diagonal elements.
    Args:
        flat_params (np.ndarray): Flattened parameters array of shape (12,).
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        uf = unflatten_biGaussian_params(flat_params)
        pi = uf['pi'].numpy()
        mu = uf['mu'].numpy()
        Sigma = uf['Sigma'].numpy()
        
        # Check for NaN values
        if np.isnan(flat_params).any():
            return False
        
        # Check mixture weights
        if np.any(pi < 0):
            return False
        
        # Check first weight is larger than second (to avoid label switching)
        if pi[0] < pi[1]:
            return False
        
        # Check means
        if np.any(mu[:, 0] < -2*xrange) or np.any(mu[:, 0] > 2*xrange):
            return False
        if np.any(mu[:, 1] < -2*yrange) or np.any(mu[:, 1] > 2*yrange):
            return False

        if do_current_profile:
            # In current profile mode, ensure that the x means are sufficiently separated
            if abs(mu[1][0]) < 10:
                return False
        
        # Check covariance matrices
        for k in range(Sigma.shape[0]):
            if Sigma[k,0,0] <= 0 or Sigma[k,1,1] <= 0:
                return False
            # Check positive definiteness
            if np.linalg.det(Sigma[k]) <= 0:
                return False
            # Trace must be at least ( 5 px ) ** 2. This prevents extremely small covariances which mean the Gaussian captured a noise spike.
            if np.trace(Sigma[k]) < 25:
                return False
        
        return True
    except:
        return False


def unflatten_biGaussian_params(flat_params, clip_sigma = False):
    """Convert a flattened 1D array back into bi-Gaussian parameters.
    Args:
        flat_params (np.ndarray): Flattened parameters array of shape (10,).
        clip_sigma (bool): If True, Clip the off-diagonal Sigmas to be less than or equal to the geometric mean of the diagonal Sigmas
    """
    pi = flat_params[:2]

    # When flattened, we only store K-1 means (the first mean is fixed at [0,0]).
    mu = np.array([[0,0], flat_params[2:4]])
    Sigma = np.zeros((2,2,2))
    for k in range(2):
        Sigma[k,0,0] = flat_params[4 + k*3]
        Sigma[k,1,1] = flat_params[6 + k*3]
        if not clip_sigma:
            Sigma[k,0,1] = flat_params[5 + k*3]
            Sigma[k,1,0] = flat_params[5 + k*3]  # Symmetric
        else:
            Sigma[k,0,0] = np.clip(Sigma[k,0,0], 10, 1e6)  # Prevent extreme values
            Sigma[k,1,1] = np.clip(Sigma[k,1,1], 10, 1e6)  # Prevent extreme values
            geometric_mean = np.sqrt(Sigma[k,0,0] * Sigma[k,1,1]) # Stddev of x times stddev of y. This is the max allowed off-diagonal value for positive-definite Sigma
            Sigma[k,0,1] = np.clip(flat_params[5 + k*3], -geometric_mean*0.9, geometric_mean*0.9)  # Clip to be less than geometric mean
            Sigma[k,1,0] = Sigma[k,0,1]  # Symmetric
    return {
        'pi': torch.tensor(pi, dtype=torch.float32),
        'mu': torch.tensor(mu, dtype=torch.float32),
        'Sigma': torch.tensor(Sigma, dtype=torch.float32)
    }

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

def gaussian_2d_pdf(mu: np.ndarray, Sigma: np.ndarray, 
    xrange = 100, yrange = 100) -> np.ndarray:
    """Calculates the 2D Gaussian PDF evaluated at coordinates.
    Args:
        mu (np.ndarray): Mean vector of shape (2,).
        Sigma (np.ndarray): Covariance matrix of shape (2, 2).
        xrange (int): Range for x-coordinates (image width / 2).
        yrange (int): Range for y-coordinates (image height / 2).
    """
    Sigma_inv = np.linalg.inv(Sigma)

    # Coordinates centered relative to mu
    xy_coords = np.array(np.meshgrid(np.arange(0, 2* xrange), np.arange(0, 2* yrange))).T.reshape(-1, 2)  # (N, 2)
    centered_coords = (xy_coords - mu)
    
   
    # Step A: V = D @ Sigma_inv  (Shape: (N, 2) @ (2, 2) -> (N, 2))
    V = centered_coords @ Sigma_inv 
    
    # Step B: Sum(V * D, axis=1) (Element-wise product, then sum along D=2 axis -> (N,))
    # This correctly computes the quadratic form for all N points in parallel.
    quad_form = np.sum(V * centered_coords, axis=1) # (N,)
    
    # PDF formula: 1 / ( (2*pi)^D * |Sigma| )^0.5 * exp(-0.5 * quad_form)
    # D=2 for 2D
    numerator = np.exp(-0.5 * quad_form)

    return numerator.reshape((2*xrange, 2*yrange)).T

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
