"""
Plain fully-connected 1-D Variational Autoencoder.

Drop-in sibling of `Python_Functions.cvae.CVAE1D` that exposes the same
interface (`forward`, `generate_latent_mu`, `decode_latent_mu`) so that
the existing training helpers in `Python_Functions.training` can target
it without changes. The difference is purely architectural: this VAE
uses an MLP encoder/decoder rather than stacked 1-D convolutions, which
often works better than CVAE1D for smooth low-dimensional waveforms
(e.g. UVVis spectra) where convolutional inductive biases give no
benefit and the bottleneck is the dense projection layer's capacity.

Input/output convention matches CVAE1D: tensors of shape (N, 1, L). The
singleton channel is flattened away inside the model so the caller does
not have to special-case shape handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 1


class VAE1D(nn.Module):
    """
    Dense (MLP) 1-D Variational Autoencoder.

    Args:
        latent_dim:    Dimensionality of the latent vector.
        input_length:  Length L of the input waveform (no multiple-of-32
                       constraint; any positive int works).
        hidden_dims:   Sequence of hidden layer widths for the encoder.
                       The decoder mirrors this in reverse. Defaults to
                       (512, 256, 128).
        dropout:       Optional dropout probability applied after each
                       hidden activation. 0 disables it.
    """

    def __init__(self, latent_dim, input_length=256, hidden_dims=(512, 256, 128), dropout=0.0):
        super().__init__()
        if input_length <= 0:
            raise ValueError(f"input_length must be positive, got {input_length}")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer width")

        self.latent_dim = latent_dim
        self.input_length = input_length
        self.hidden_dims = tuple(hidden_dims)

        encoder_layers = []
        prev = input_length
        for h in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        decoder_layers = []
        prev = latent_dim
        for h in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev, h))
            decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev = h
        decoder_layers.append(nn.Linear(prev, input_length))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def _flatten(self, x):
        # Accept (N, L), (N, 1, L), or (N, C, L) with C==1.
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if x.size(1) != INPUT_CHANNELS:
                raise ValueError(
                    f"VAE1D expects a single channel, got {x.size(1)} on dim 1"
                )
            return x.view(x.size(0), -1)
        raise ValueError(f"VAE1D input must be 2-D or 3-D, got shape {tuple(x.shape)}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        flat = self._flatten(x)
        h = self.encoder(flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decoder(z)
        reconstruction = recon_flat.view(-1, INPUT_CHANNELS, self.input_length)
        return reconstruction, mu, logvar

    def generate_latent_mu(self, x):
        flat = self._flatten(x)
        h = self.encoder(flat)
        return self.fc_mu(h)

    def decode_latent_mu(self, mu):
        recon_flat = self.decoder(mu)
        return recon_flat.view(-1, INPUT_CHANNELS, self.input_length)


def vae1d_loss(reconstruction, x, mu, logvar):
    """
    Standard VAE loss (BCE + KL). Wraps the same formulation used by
    `Python_Functions.cvae.vae_loss`, kept here so callers that import
    only `vae` get a matching loss without pulling in the 2-D module.
    """
    if reconstruction.shape != x.shape:
        x = x.view_as(reconstruction)
    BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    KL_Divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KL_Divergence
