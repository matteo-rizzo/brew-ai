from torch import nn

from src.classes.evaluation.periodicity.embedding.DeepFeatureEmbedding import DeepFeatureEmbedding
from src.classes.evaluation.periodicity.embedding.FeatureEmbedding import FeatureEmbedding


class FeatureEmbeddingFactory:
    @staticmethod
    def get_feature_embedding(embedding_type: str, input_size: int) -> nn.Module:
        """
        Factory method to create different types of feature embeddings.

        :param embedding_type: The type of embedding ('linear', 'deep').
        :param input_size: Size of the input features.
        :return: A PyTorch Module that performs the embedding.
        """

        if embedding_type == 'linear':
            return FeatureEmbedding(input_size)

        elif embedding_type == 'deep':
            return DeepFeatureEmbedding(input_size)

        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
