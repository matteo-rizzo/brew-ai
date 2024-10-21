from torch import nn
import torch


class FeatureSelector(nn.Module):

    @staticmethod
    def apply_feature_selection(x: torch.Tensor, periodicity_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature selection by weighting input features for Fourier and Chebyshev transformations.

        :param x: Input feature tensor of shape [batch_size, input_size].
        :param periodicity_scores: Periodicity scores tensor of shape [batch_size, input_size].
        :return: Tuple of (x_fourier_weighted, x_chebyshev_weighted) tensors.
        """
        # Fourier weighting using periodicity scores
        x_fourier_weighted = x * periodicity_scores  # Shape: [batch_size, input_size]
        # Chebyshev weighting using (1 - periodicity_scores)
        x_chebyshev_weighted = x * (1 - periodicity_scores)  # Shape: [batch_size, input_size]

        return x_fourier_weighted, x_chebyshev_weighted

    @staticmethod
    def apply_gating(x_fourier: torch.Tensor, x_chebyshev: torch.Tensor, periodicity_scores: torch.Tensor,
                     num_fourier_features_per_input: int, num_chebyshev_features_per_input: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gating mechanism on Fourier and Chebyshev transformations based on periodicity scores.

        :param x_fourier: Tensor from Fourier transformation of shape [batch_size, total_fourier_features].
        :param x_chebyshev: Tensor from Chebyshev transformation of shape [batch_size, total_chebyshev_features].
        :param periodicity_scores: Periodicity scores of shape [batch_size, input_size].
        :param num_fourier_features_per_input: Number of Fourier features per input.
        :param num_chebyshev_features_per_input: Number of Chebyshev features per input.
        :return: Tuple of gated (x_fourier_weighted, x_chebyshev_weighted) tensors.
        """
        batch_size = x_fourier.size(0)

        # Reshape periodicity scores for gating Fourier features
        periodicity_scores_fourier = periodicity_scores.unsqueeze(2).expand(-1, -1, num_fourier_features_per_input)
        periodicity_scores_fourier = periodicity_scores_fourier.reshape(batch_size, -1)

        # Reshape periodicity scores for gating Chebyshev features
        periodicity_scores_chebyshev = (1 - periodicity_scores).unsqueeze(2).expand(-1, -1, num_chebyshev_features_per_input)
        periodicity_scores_chebyshev = periodicity_scores_chebyshev.reshape(batch_size, -1)

        # Apply gating
        x_fourier_weighted = x_fourier * periodicity_scores_fourier
        x_chebyshev_weighted = x_chebyshev * periodicity_scores_chebyshev

        return x_fourier_weighted, x_chebyshev_weighted
