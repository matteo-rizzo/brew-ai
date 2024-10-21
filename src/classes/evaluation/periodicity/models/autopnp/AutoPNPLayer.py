import torch
from torch import nn

from src.classes.evaluation.periodicity.factories.FeatureSelectorFactory import FeatureSelectorFactory
from src.classes.evaluation.periodicity.models.chebyshev.AdaptiveChebyshevLayer import AdaptiveChebyshevLayer
from src.classes.evaluation.periodicity.models.fourier.FourierLayer import FourierLayer


class AutoPNPLayer(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_fourier_features: int,
            num_chebyshev_terms: int,
            feature_selector: str = "default",
            feature_selection_before_transform: bool = True
    ):
        """
        AutoPNPNet that integrates Fourier and Chebyshev layers with feature selection, embeddings, and an MLP regressor.

        :param input_size: Size of the input features.
        :param num_fourier_features: Number of Fourier features to generate.
        :param num_chebyshev_terms: Number of Chebyshev polynomial terms.
        :param feature_selector: Type of feature selector to use.
        """
        super(AutoPNPLayer, self).__init__()

        self.apply_feature_selection_before_transform = feature_selection_before_transform

        # Fourier and Chebyshev layers
        self.fourier_layer = FourierLayer(input_size, num_fourier_features)
        self.chebyshev_layer = AdaptiveChebyshevLayer(input_size, num_chebyshev_terms)

        # Feature selector initialization
        self.feature_selector = FeatureSelectorFactory.get_feature_selector(feature_selector, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Estimate periodicity scores using the feature selector
        periodicity_scores = self.feature_selector(x)

        # Apply feature selection or gating
        if self.apply_feature_selection_before_transform:
            x_fourier, x_chebyshev = self.feature_selector.apply_feature_selection(x, periodicity_scores)
        else:
            x_fourier, x_chebyshev = x, x

        # Apply Fourier and Chebyshev transformations
        x_fourier = self.fourier_layer(x_fourier)
        x_chebyshev = self.chebyshev_layer(x_chebyshev)

        if not self.apply_feature_selection_before_transform:
            # Apply gating to transformed features
            x_fourier, x_chebyshev = self.feature_selector.apply_gating(
                x_fourier, x_chebyshev, periodicity_scores,
                self.fourier_layer.num_features_per_input * 2,  # Fourier features per input
                self.chebyshev_layer.max_terms  # Chebyshev terms
            )

        # Combine Fourier and Chebyshev transformed features
        return torch.cat([x_fourier, x_chebyshev], dim=1)
