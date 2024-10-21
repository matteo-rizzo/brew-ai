from src.classes.evaluation.periodicity.feature_selector.AttentionPeriodicityFeatureSelector import \
    AttentionPeriodicityFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.GatedPeriodicityFeatureSelector import \
    GatedPeriodicityFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.PeriodicityFeatureSelector import PeriodicityFeatureSelector
from src.classes.evaluation.periodicity.feature_selector.SEPeriodicityFeatureSelector import \
    SEPeriodicityFeatureSelector


class FeatureSelectorFactory:
    @staticmethod
    def get_feature_selector(name: str, input_size: int):
        """Factory method to get the appropriate feature selector."""
        selectors = {
            "attention": AttentionPeriodicityFeatureSelector(input_size),
            "se": SEPeriodicityFeatureSelector(input_size),
            "gated": GatedPeriodicityFeatureSelector(input_size),
            "default": PeriodicityFeatureSelector(input_size)
        }
        return selectors.get(name, selectors["default"])
