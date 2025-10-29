import pandas as pd
from ..utils.memoizer import Memoizer
import re
import numpy as np




base_feature_cache = Memoizer(name="base_feature_cache")

class BaseFeatureBuilder:
    def __init__(self):
        self.cache = base_feature_cache

        self.pattern_time = re.compile(r"t\d*[05](?=,|$)")
        self.pattern_ecran = re.compile(r"\((.*?)\)")
        self.pattern_conf_ecran = re.compile(r"<(.*?)>")
        self.pattern_chaine = re.compile(r"\$(.*?)\$")
        self.pattern_actions = re.compile(r"(?!t\d+)[^,]+")

        self.nav_dict = {
            'Google Chrome': 0,
            'Firefox': 1,
            'Microsoft Edge': 2,
            "Opera": 3
            }

    def clear_cache(self):
        self.cache.clear()

    def encode_nav(self, x):
        if x in list(self.nav_dict.keys()):
            return self.nav_dict[x]
        else: 
            return 4

    @base_feature_cache
    def nav(self, df: pd.DataFrame) -> pd.DataFrame:
        df['nav'] = df['nav'].apply(self.encode_nav)
        return df


    def compute_speed(self, fi):
        blocks = []
        count = 0
        last_ts = 0
        fis = fi.split(',')
        for elem in fis:
            count += 1
            if self.pattern_time.fullmatch(elem):
                delta_ts = int(elem.split('t')[1]) - last_ts
                blocks.append(count / delta_ts)

        if blocks == []:
            return -1, -1
        arr = np.array(blocks)
        return arr.mean(), arr.std()
    

    @base_feature_cache
    def speed(self, df: pd.DataFrame) -> pd.DataFrame:
        df[['speed_mean', 'speed_std']] = df['full_items'].apply(self.compute_speed).apply(pd.Series)
        return df

    def compute_session_time_span(self, fi):
        all_times = self.pattern_time.findall(fi)
        if all_times[-1][1:-1:1] != '':
            return int(all_times[-1][1:-1:1])
        else:
            return 0


    @base_feature_cache
    def session_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df['time_span'] = df['full_items'].apply(self.compute_session_time_span)
        return df

    def compute_nb_action(self, fi):
        tokens = fi.split(",")
        actions = [t.split('(')[0].split('1')[0] for t in tokens if not re.fullmatch(r"t\d+", t)]
        return len(actions)
    

    @base_feature_cache
    def nb_action(self, df: pd.DataFrame) -> pd.DataFrame:
        df["nb_actions"] = df['full_items'].apply(self.compute_nb_action)
        return df

