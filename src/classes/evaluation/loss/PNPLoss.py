import torch
import torch.fft as fft

class PNPMSELoss(torch.nn.Module):
    def __init__(self, fourier_weight=1.0, chebyshev_weight=1.0, mse_weight=1.0):
        """
        Hybrid loss function that combines Fourier, Chebyshev, and MSE losses.

        :param fourier_weight: Weight for the Fourier loss component.
        :param chebyshev_weight: Weight for the Chebyshev loss component.
        :param mse_weight: Weight for the MSE loss component.
        """
        super(PNPMSELoss, self).__init__()
        self.fourier_weight = fourier_weight
        self.chebyshev_weight = chebyshev_weight
        self.mse_weight = mse_weight

    def forward(self, predictions, targets):
        # Mean Squared Error (MSE) Loss
        mse_loss = torch.mean((predictions - targets) ** 2)

        # Fourier Loss: Compare Fourier components of predictions and targets
        predictions_fft = fft.fft(predictions, dim=-1)
        targets_fft = fft.fft(targets, dim=-1)
        fourier_loss = torch.mean(torch.abs(predictions_fft - targets_fft))

        # Chebyshev Loss: Maximum absolute difference along each sample
        chebyshev_loss = torch.max(torch.abs(predictions - targets), dim=-1).values.mean()

        # Weighted sum of the three components
        total_loss = (
            self.mse_weight * mse_loss +
            self.fourier_weight * fourier_loss +
            self.chebyshev_weight * chebyshev_loss
        )

        return total_loss
