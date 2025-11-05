import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.fft as fft  # Added for FFT calculation
import matplotlib.pyplot as plt # Added for plotting

# --- Configuration ---
IMG_SIZE = 200
LATENT_DIM = 12
INPUT_CHANNELS = 1 # Assuming LPSimg is grayscale/single-channel data

class CVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 200x200 single-channel images.
    Encodes the input image into a 12-dimensional latent space.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # The target flattened size after the last convolutional layer.
        # Calculated from the architecture: 512 channels * 6 * 6 spatial size.
        self.FLATTEN_SIZE = 512 * 6 * 6 # 18432

        # --- Encoder (Downsampling from 200x200 to 6x6) ---
        # Output sizes using (k=4, s=2, p=1)
        # 200 -> 100 -> 50 -> 25 -> 12 -> 6
        self.encoder = nn.Sequential(
            # 1. 200x200 -> 100x100
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 2. 100x100 -> 50x50
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 3. 50x50 -> 25x25
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4. 25x25 -> 12x12
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 5. 12x12 -> 6x6
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            # Final output shape: (batch_size, 512, 6, 6)
        )
        
        # Linear layers for Mu and Log Variance
        self.fc_mu = nn.Linear(self.FLATTEN_SIZE, latent_dim)
        self.fc_logvar = nn.Linear(self.FLATTEN_SIZE, latent_dim)

        # --- Decoder (Upsampling from 6x6 to 200x200) ---
        self.fc_decode = nn.Linear(latent_dim, self.FLATTEN_SIZE)

        # Transposed convolutional layers
        # The output_padding=1 is crucial for ConvTranspose2 to match the odd size 25x25
        self.decoder = nn.Sequential(
            # 1. 6x6 -> 12x12
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 2. 12x12 -> 25x25 (Requires output_padding=1)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 3. 25x25 -> 50x50
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 4. 50x50 -> 100x100
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 5. 100x100 -> 200x200 (Output layer)
            nn.ConvTranspose2d(32, INPUT_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Use Sigmoid to output pixel values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick to sample from the latent space.
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        # Generate random noise epsilon from a standard normal distribution
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # --- Encoding ---
        # x shape: (N, 1, 200, 200)
        encoded = self.encoder(x)
        # Flatten: (N, 18432)
        encoded_flat = encoded.view(-1, self.FLATTEN_SIZE)

        # Latent variables
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # --- Decoding ---
        # Linear layer: (N, 18432)
        decoded_flat = self.fc_decode(z)
        # Reshape to (N, 512, 6, 6)
        decoded_reshaped = decoded_flat.view(-1, 512, 6, 6)

        # Transposed convolutions
        reconstruction = self.decoder(decoded_reshaped)

        return reconstruction, mu, logvar
    
    def generate_latent_mu(self, x):
        """
        Given input x, encode it and return the latent vector mu.
        Mu is the mean of the latent distribution, hence is deterministic.
        """
        encoded = self.encoder(x)
        encoded_flat = encoded.view(-1, self.FLATTEN_SIZE)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        return mu

    def decode_latent_mu(self, mu):
        """
        Given latent vector mu, decode it to reconstruct the image.
        Mu is used here  instead of z for deterministic reconstruction.
        """
        decoded_flat = self.fc_decode(mu)
        decoded_reshaped = decoded_flat.view(-1, 512, 6, 6)
        reconstruction = self.decoder(decoded_reshaped)
        return reconstruction

def vae_loss(reconstruction, x, mu, logvar):
    """
    The VAE loss function: Reconstruction Loss (Binary Cross-Entropy) + KL Divergence.
    """
    # 1. Reconstruction Loss (Binary Cross-Entropy)
    # The BCE is typically used when the output is a Sigmoid and target is 0/1 (or 0-1 continuous)
    BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    
    # 2. KL Divergence Loss
    # D_KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Here, sigma^2 = exp(logvar)
    KL_Divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss is the sum of these two components
    return BCE + KL_Divergence

def apply_fft_and_plot(LPSimg_np):
    """
    Applies 2D FFT to a random sample from the LPSimg numpy array,
    and plots the original image and its centered magnitude spectrum.

    Args:
        LPSimg_np (np.ndarray): The input numpy array of shape [n_samples, 200, 200].
    """
    n_samples = LPSimg_np.shape[0]
    # Select a random sample index
    sample_idx = np.random.randint(0, n_samples)
    img = LPSimg_np[sample_idx]
    
    print(f"\n--- FFT Analysis for Sample {sample_idx} ---")

    # 1. Apply 2D FFT
    # This computes the 2D Fourier Transform
    f_transform = fft.fft2(img)

    # 2. Shift the zero-frequency component (DC component) to the center for visualization
    f_shift = fft.fftshift(f_transform)

    # 3. Calculate the magnitude spectrum
    # We take the absolute value (magnitude) and use a log scale 
    # (multiplied by 20 to convert to dB, common practice for visualization)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    
    print(f"Original image shape: {img.shape}")
    print(f"Magnitude spectrum max value (log scale): {magnitude_spectrum.max():.2f}")
    
    # 4. Plotting
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Sample {sample_idx} (Original Image and FFT Magnitude Spectrum)", fontsize=14)
        
        # Plot Original Image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot Magnitude Spectrum
        axes[1].imshow(magnitude_spectrum, cmap='gray')
        axes[1].set_title('FFT Magnitude Spectrum (Log Scale)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        # Catch exception if running in an environment without display server
        print(f"Could not display plot. Error: {e}")
        print("Note: Plotting with matplotlib requires a suitable backend. The FFT calculation completed successfully.")


# --- Example Usage and Dummy Training Loop ---

if __name__ == '__main__':
    print(f"Initializing VAE with Image Size: {IMG_SIZE}x{IMG_SIZE}, Latent Dim: {LATENT_DIM}")

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Model and Optimizer
    model = CVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Simulate Data (LPSimg)
    # LPSimg is a numpy array of shape [n_samples, 200, 200].
    N_SAMPLES = 100
    BATCH_SIZE = 16
    
    # Create dummy data: N_SAMPLES images (200x200) with random values between 0 and 1
    # We simulate data that is already normalized.
    LPSimg_np = np.random.rand(N_SAMPLES, IMG_SIZE, IMG_SIZE).astype(np.float32)
    
    # --- NEW: Perform FFT Analysis on a Random Sample ---
    # Call the new function before converting to tensor for training
    apply_fft_and_plot(LPSimg_np)
    # ---------------------------------------------------

    # Convert numpy data to PyTorch Tensor, add the Channel dimension (C=1)
    LPSimg_tensor = torch.from_numpy(LPSimg_np).unsqueeze(1) 
    
    # Create a simple DataLoader for the dummy data
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(LPSimg_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Dummy Training Loop
    N_EPOCHS = 5
    print("\nStarting dummy training loop (5 epochs)...")

    for epoch in range(1, N_EPOCHS + 1):
        total_loss = 0
        for batch_idx, (data,) in enumerate(data_loader):
            data = data.to(device)
            
            # Forward pass
            reconstruction, mu, logvar = model(data)
            
            # Calculate loss
            loss = vae_loss(reconstruction, data, mu, logvar)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{N_EPOCHS}, Average VAE Loss: {avg_loss:.4f}")

    print("\nTraining complete.")
    
    # 5. Test Latent Vector Generation
    # Take one sample image
    sample_data = LPSimg_tensor[0:1].to(device)
    
    # Run through the model (only need mu and logvar)
    with torch.no_grad():
        _, mu_test, logvar_test = model(sample_data)
        z_sample = model.reparameterize(mu_test, logvar_test)

    print(f"\nExample Input Data Shape: {sample_data.shape}")
    print(f"Generated Latent Vector (mu) Shape: {mu_test.shape} (Confirmed {LATENT_DIM} components)")
    print(f"Generated Latent Vector (z_sample):\n{z_sample.cpu().numpy().round(3)}")

    # 6. Test Reconstruction
    with torch.no_grad():
        reconstructed_image, _, _ = model(sample_data)
    
    print(f"Reconstructed Image Shape: {reconstructed_image.shape}")
    print(f"Reconstruction Pixel Max Value: {reconstructed_image.max().item():.3f}")
    print(f"Reconstruction Pixel Min Value: {reconstructed_image.min().item():.3f}")

    # Note: The model is now trained and ready to be used for dimensionality reduction
    # or generation tasks.
    
