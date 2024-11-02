import torch
import torch.fft as fft
import torch.nn.functional as F


class PNPMSELoss(torch.nn.Module):
    def __init__(
            self,
            fourier_weight=1.0,
            chebyshev_weight=1.0,
            mse_weight=1.0,
            weight_frequencies=True,
            fft_dim=-1,
            fourier_norm_method='max',
            frequency_weight_fn=None,
            chebyshev_dim=-1
    ):
        """
        Hybrid loss function combining MSE, Chebyshev, and Fourier losses with customizable options.

        :param fourier_weight: Weight for the Fourier loss component.
        :param chebyshev_weight: Weight for the Chebyshev loss component.
        :param mse_weight: Weight for the MSE loss component.
        :param weight_frequencies: Whether to apply frequency weighting in Fourier loss.
        :param fft_dim: Dimension(s) along which to perform the FFT. Can be int or tuple of ints.
        :param fourier_norm_method: Method to normalize Fourier magnitudes ('max', 'l1', 'l2', 'none').
        :param frequency_weight_fn: Function to compute frequency weights. If None, default weighting is used.
        :param chebyshev_dim: Dimension(s) along which to compute the maximum absolute difference.
        """
        super(PNPMSELoss, self).__init__()
        self.fourier_weight = fourier_weight
        self.chebyshev_weight = chebyshev_weight
        self.mse_weight = mse_weight
        self.weight_frequencies = weight_frequencies
        self.fft_dim = fft_dim
        self.fourier_norm_method = fourier_norm_method
        self.frequency_weight_fn = frequency_weight_fn
        self.chebyshev_dim = chebyshev_dim

    def forward(self, predictions, targets):
        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        # Mean Squared Error (MSE) Loss
        mse_loss = F.mse_loss(predictions, targets)

        # Fourier Loss: Compare Fourier components of predictions and targets
        predictions_fft = fft.fftn(predictions, dim=self.fft_dim, norm='ortho')
        targets_fft = fft.fftn(targets, dim=self.fft_dim, norm='ortho')

        # Compute magnitudes of the Fourier components
        predictions_fft_mag = predictions_fft.abs()
        targets_fft_mag = targets_fft.abs()

        # Normalize the Fourier magnitudes
        epsilon = 1e-8  # Small constant to prevent division by zero
        if self.fourier_norm_method == 'max':
            pred_denominator = predictions_fft_mag.amax(dim=self.fft_dim, keepdim=True) + epsilon
            targ_denominator = targets_fft_mag.amax(dim=self.fft_dim, keepdim=True) + epsilon
        elif self.fourier_norm_method == 'l1':
            pred_denominator = predictions_fft_mag.sum(dim=self.fft_dim, keepdim=True) + epsilon
            targ_denominator = targets_fft_mag.sum(dim=self.fft_dim, keepdim=True) + epsilon
        elif self.fourier_norm_method == 'l2':
            pred_denominator = predictions_fft_mag.norm(p=2, dim=self.fft_dim, keepdim=True) + epsilon
            targ_denominator = targets_fft_mag.norm(p=2, dim=self.fft_dim, keepdim=True) + epsilon
        elif self.fourier_norm_method == 'none':
            pred_denominator = targ_denominator = 1.0
        else:
            raise ValueError(
                f"Invalid fourier_norm_method: '{self.fourier_norm_method}'. Choose from 'max', 'l1', 'l2', or 'none'.")

        predictions_fft_norm = predictions_fft_mag / pred_denominator
        targets_fft_norm = targets_fft_mag / targ_denominator

        # Compute Fourier loss
        if self.weight_frequencies:
            # Generate frequency grids
            freq_dims = self.fft_dim if isinstance(self.fft_dim, (tuple, list)) else [self.fft_dim]
            freq_shape = [predictions.size(d) for d in freq_dims]
            frequencies = [torch.fft.fftfreq(n, d=1.0, device=predictions.device) for n in freq_shape]
            meshgrids = torch.meshgrid(*frequencies, indexing='ij')

            # Compute frequency magnitude (Euclidean distance in frequency space)
            freq_magnitude = torch.sqrt(sum(f ** 2 for f in meshgrids))

            # Expand dimensions to match predictions_fft_norm
            for dim in sorted(set(range(predictions_fft_norm.ndim)) - set(freq_dims)):
                freq_magnitude = freq_magnitude.unsqueeze(dim)

            # Apply frequency weighting function
            if self.frequency_weight_fn is not None:
                frequency_weight = self.frequency_weight_fn(freq_magnitude)
            else:
                # Default weighting: inverse frequency (emphasize lower frequencies)
                frequency_weight = 1 / (freq_magnitude + epsilon)

            # Handle NaNs or Infs in frequency_weight
            frequency_weight = torch.where(torch.isfinite(frequency_weight), frequency_weight,
                                           torch.zeros_like(frequency_weight))

            # Compute weighted Fourier loss
            fourier_diff = predictions_fft_norm - targets_fft_norm
            fourier_loss = torch.mean(frequency_weight * (fourier_diff ** 2))
        else:
            # Unweighted Fourier loss
            fourier_loss = F.mse_loss(predictions_fft_norm, targets_fft_norm)

        # Chebyshev Loss: Maximum absolute difference along specified dimensions
        chebyshev_loss = torch.mean(torch.amax(torch.abs(predictions - targets), dim=self.chebyshev_dim))

        # Total loss: Weighted sum of the three components
        total_loss = (
                self.mse_weight * mse_loss +
                self.fourier_weight * fourier_loss +
                self.chebyshev_weight * chebyshev_loss
        )

        return total_loss
