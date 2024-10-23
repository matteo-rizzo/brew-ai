from src.classes.evaluation.periodicity.feature_selector.AttentionFeatureSelector import \
    AttentionFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.Conv1DFeatureSelector import Conv1DFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.PNPFeatureSelector import PNPFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.PeriodicityFeatureSelector import PeriodicityFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.SEPeriodicityFeatureSelector import \
    SEPeriodicityFeatureSelector


class FeatureSelectorFactory:
    @staticmethod
    def get_feature_selector(name: str, input_size: int):
        """Factory method to get the appropriate feature selector."""
        selectors = {
            "attention": AttentionFeatureSelector(input_size),
            "se": SEPeriodicityFeatureSelector(input_size),
            "conv": Conv1DFeatureSelector(input_size),
            "pnp": PNPFeatureSelector(input_size),
            "default": PeriodicityFeatureSelector(input_size)
        }
        return selectors.get(name, selectors["default"])
