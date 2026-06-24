"""
Pure-Python training routines for the CVAE + Random Forest pipeline.

Extracted from VTCAVDisplay.train_model_cvae_wrf so the same logic can be
called from notebooks (e.g. predicting 1-D UVVis spectra) without pulling
in PyDM/Qt/EPICS. Saving of the trained artifacts stays at the call site,
because different consumers want different metadata in the saved dict.
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from Python_Functions.cvae import CVAE, CVAE1D, proj_vae_loss, vae_loss


def encode_with_cvae(
    target_data,
    latent_dim,
    *,
    n_epochs=8,
    batch_size=16,
    lr=1e-3,
    proj_loss_strength=0.9,
    proj_log_strength=0.0,
    device=None,
    model_class=None,
    log=print,
):
    """
    Fit a CVAE on `target_data` and return (model, latent_means).

    target_data:
        np.ndarray of shape (N, H, W) for image data, or (N, L) for 1-D
        waveforms. Values do not need to be pre-normalized; the routine
        divides by the global max to land in [0, 1] (matches the original
        VTCAVDisplay training behavior).
    latent_dim:
        Dimensionality of the latent space.
    model_class:
        Optional explicit subclass to instantiate. If None, picked by
        target_data.ndim: 3-D -> CVAE, 2-D -> CVAE1D(input_length=L).

    Returns:
        (model, latent_z_array) where latent_z_array has shape (N, latent_dim).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    arr = np.asarray(target_data)
    if model_class is None:
        if arr.ndim == 3:
            model = CVAE(latent_dim=latent_dim).to(device)
            use_proj_loss = True
        elif arr.ndim == 2:
            model = CVAE1D(latent_dim=latent_dim, input_length=arr.shape[1]).to(device)
            use_proj_loss = False
        else:
            raise ValueError(
                f"target_data must be (N,H,W) or (N,L); got shape {arr.shape}"
            )
    else:
        model = model_class(latent_dim=latent_dim).to(device)
        use_proj_loss = arr.ndim == 3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tensor = torch.from_numpy(arr).unsqueeze(1).float()
    tensor /= tensor.max()
    dataset = TensorDataset(tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log(f"\nStarting training loop ({n_epochs} epochs)...")
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for (data,) in data_loader:
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            data = data.clamp(min=0)
            if use_proj_loss:
                loss = proj_vae_loss(
                    reconstruction, data, mu, logvar,
                    strength=proj_loss_strength,
                    log_strength=proj_log_strength,
                )
            else:
                loss = vae_loss(reconstruction, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        log(f"Epoch {epoch}/{n_epochs}, Average VAE Loss: {avg_loss:.4f}")

    latent_z_array = np.zeros((arr.shape[0], latent_dim))
    for i in range(arr.shape[0]):
        sample = arr[i] / arr[i].max() if arr[i].max() > 0 else arr[i]
        sample_tensor = torch.from_numpy(sample).unsqueeze(0).unsqueeze(0).float().to(device)
        mu_tensor = model.generate_latent_mu(sample_tensor)
        latent_z_array[i] = mu_tensor.cpu().detach().numpy()

    return model, latent_z_array


def train_random_forest_predictor(
    predictors,
    targets,
    *,
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=2,
    test_size=0.2,
    weight_strategy="importance_weighted",
    kde_bandwidth=0.1,
    random_state=42,
    log=print,
):
    """
    Fit a weighted RandomForestRegressor mapping `predictors` -> `targets`.

    predictors:
        (N, F) array of raw scalar inputs.
    targets:
        (N, K) array of regression targets, typically the latent codes
        returned by encode_with_cvae.
    weight_strategy:
        "importance_weighted" (default) — first pass an unweighted RF to
        get feature importances, weight the feature space by importance,
        then density-weight via KDE so undersampled regions contribute
        more.
        "density" — same KDE weighting but in raw scaled space.
        None — uniform weights.

    Returns dict with keys:
        model, x_scaler, iz_scaler, feature_importances,
        train_mse, test_mse, train_r2, test_r2,
        train_indices, test_indices.
    """
    predictors = np.asarray(predictors)
    targets = np.asarray(targets)
    log(f"Predictors shape: {predictors.shape}, Targets shape: {targets.shape}")

    x_scaler = MinMaxScaler()
    iz_scaler = MinMaxScaler()
    x_scaled = x_scaler.fit_transform(predictors)
    Iz_scaled = iz_scaler.fit_transform(targets)

    n = Iz_scaled.shape[0]
    indices = np.arange(n)
    x_train_scaled, x_test_scaled, Iz_train_scaled, Iz_test_scaled, train_indices, test_indices = train_test_split(
        x_scaled, Iz_scaled, indices, test_size=test_size, random_state=random_state,
    )

    log(f"X_train shape: {x_train_scaled.shape}")
    log(f"Y_train shape: {Iz_train_scaled.shape}")

    if weight_strategy == "density":
        log("Calculating weights using naive local density estimation...")
        x_train_weighted = x_train_scaled
    elif weight_strategy == "importance_weighted":
        log("Calculating weights using a first-pass feature importance...")
        first_pass_model = RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1,
        )
        first_pass_model.fit(x_train_scaled, Iz_train_scaled)
        importances = first_pass_model.feature_importances_
        x_train_weighted = x_train_scaled * importances
    elif weight_strategy is None:
        sample_weights = None
    else:
        raise ValueError(f"Unknown weight_strategy: {weight_strategy!r}")

    if weight_strategy in ("density", "importance_weighted"):
        kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(x_train_weighted)
        log_densities = kde.score_samples(x_train_weighted)
        log_densities_shifted = log_densities - np.max(log_densities)
        density = np.exp(log_densities_shifted)
        sample_weights = 1.0 / (density + 1e-6)
        log(f"Max sample weight: {np.max(sample_weights)}")
        log(f"Min sample weight: {np.min(sample_weights)}")
        sample_weights = sample_weights / np.mean(sample_weights)

    log("\n--- Initializing Weighted Random Forest Model ---")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    t0 = time.time()
    log("\n--- Starting Model Fitting (Weighted) ---")
    model.fit(x_train_scaled, Iz_train_scaled, sample_weight=sample_weights)
    t1 = time.time()

    Y_train_pred = model.predict(x_train_scaled)
    Y_test_pred = model.predict(x_test_scaled)
    train_mse = mean_squared_error(Iz_train_scaled, Y_train_pred)
    test_mse = mean_squared_error(Iz_test_scaled, Y_test_pred)

    log("\n--- Feature Importance ---")
    for i, importance in enumerate(model.feature_importances_):
        log(f"feature[{i}] importance: {importance:.4f}")

    log("\n--- Training Results ---")
    log(f"Total Fitting Time: {t1 - t0:.2f} seconds")
    log(f"Final Train MSE: {train_mse:.6f}")
    log(f"Final Test MSE: {test_mse:.6f}")

    def r2(true, pred):
        RSS = np.sum((true - pred) ** 2)
        TSS = np.sum((true - np.mean(true)) ** 2)
        return 1 - RSS / TSS if TSS != 0 else 0

    train_r2 = r2(Iz_train_scaled.ravel(), Y_train_pred.ravel())
    test_r2 = r2(Iz_test_scaled.ravel(), Y_test_pred.ravel())
    log("Train R²: {:.2f} %".format(train_r2 * 100))
    log("Test R²: {:.2f} %".format(test_r2 * 100))

    return {
        "model": model,
        "x_scaler": x_scaler,
        "iz_scaler": iz_scaler,
        "feature_importances": model.feature_importances_,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
