import pandas as pd
from ..utils.memoizer import Memoizer
import re
import numpy as np

advanced_feature_cache = Memoizer(name='advanced_feature_cache')

class AdvancedFeatureBuilder:
    def __init__(self):
        self.cache = advanced_feature_cache

    def clear_cache(self):
        self.cache.clear()

    @advanced_feature_cache
    def action_per_tspan(self, df: pd.DataFrame) -> pd.DataFrame:
        print(type(df))
        df['action_per_session'] = df[['nb_actions', 'time_span']].apply(
            lambda x: x[0] / x[1] if x[1] != 0 else 0,
            axis=1)
        return df
