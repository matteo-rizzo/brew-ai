from torch import nn

from src.classes.evaluation.periodicity.feature_selector.FeatureSelector import FeatureSelector


class Conv1DFeatureSelector(FeatureSelector):
    def __init__(self, input_size, num_filters=64, kernel_size=1, stride=1, pooling='max',
                 selection_activation='sigmoid'):
        """
        1D Convolutional Feature Selector.

        :param input_size: Number of input features.
        :param num_filters: Number of filters for the 1D convolution.
        :param kernel_size: Size of the convolutional kernel.
        :param stride: Stride of the convolution.
        :param pooling: Type of pooling to apply ('max' or 'average').
        :param selection_activation: Activation function for selection ('sigmoid' or 'softmax').
        """
        super(Conv1DFeatureSelector, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, stride=stride,
                                padding=kernel_size // 2)

        # Pooling layer
        if pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'average':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Unsupported pooling type. Use 'max' or 'average'.")

        # Output size is reduced to number of filters
        self.fc = nn.Linear(num_filters, input_size)

        # Feature selection activation
        if selection_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif selection_activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            raise ValueError("Unsupported selection activation. Use 'sigmoid' or 'softmax'.")

    def forward(self, x):
        # Reshape input to [batch_size, channels=1, input_size]
        x = x.unsqueeze(1)  # Adding channel dimension for 1D convolution

        # Apply 1D convolution
        x_conv = self.conv1d(x)  # Shape: [batch_size, num_filters, input_size]

        # Apply pooling
        x_pooled = self.pool(x_conv).squeeze(-1)  # Shape: [batch_size, num_filters]

        # Apply fully connected layer to project back to input size
        x_fc = self.fc(x_pooled)  # Shape: [batch_size, input_size]

        # Apply activation to get selection scores
        x_selected = self.activation(x_fc)  # Shape: [batch_size, input_size]

        return x_selected
