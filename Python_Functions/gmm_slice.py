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
    DISTANCE_FACTOR = 0.5 # Factor to scale the means during normalization

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
    image_size = (n_eslice, 2*xrange)
    generated_image = np.zeros(image_size)
    # Take each vertical slice 
    for i in range(n_eslice):
        # Extract parameters for the current slice
        mu = params[i, 0]
        sigma = params[i, 1]
        # Generate the Gaussian profile for the current slice
        x = np.arange(2*xrange)
        # print(f"Mu: {mu}, Sigma: {sigma}")
        gaussian_profile = gaussian_1d(x, mu, sigma, 1.0)
        # Add the profile to the generated image
        generated_image[i, :] = gaussian_profile * charge_per_eslice[i] * total_charge if charge_per_eslice is not None else gaussian_profile
        # Zoom the generated profile to fit the image size (if necessary)
    import scipy.ndimage
    generated_image = scipy.ndimage.zoom(generated_image, (2*yrange / generated_image.shape[0], 1), order=1)
    return generated_image

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
            print(charge_per_eslice)
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
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2) / sigma / np.sqrt(2*np.pi)

def bigaussian_1d(x, mu1, sigma1, pi1, mu2, sigma2, pi2):
    """Sum of two 1D Gaussians for vertical projection fitting."""
    return (
        gaussian_1d(x, mu1, sigma1, pi1) + 
        gaussian_1d(x, mu2, sigma2, pi2)
    )
