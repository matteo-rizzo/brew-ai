import torch.nn as nn


class ActivationFactory:
    @staticmethod
    def get_activation_function(name: str):
        """Factory method to get the activation function based on the name."""
        activations = {
            'ReLU': nn.ReLU(),
            'SiLU': nn.SiLU(),
            'LeakyReLU': nn.LeakyReLU()
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation function: {name}")
        return activations[name]
