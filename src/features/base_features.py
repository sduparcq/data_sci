import pandas as pd
from ..utils.memoizer import Memoizer
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import tee


base_feature_cache = Memoizer(name="base_feature_cache")

class BaseFeatureBuilder:
    def __init__(self):
        self.cache = base_feature_cache

        self.le_action = LabelEncoder()
        self.le_module = LabelEncoder()

        self.top_n_screen_conf = 10
        self.top_n_chaine = 20
        self.top_n_action = 20


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

    def nav(self, df: pd.DataFrame) -> pd.DataFrame:
        df['nav'] = df['nav'].apply(self.encode_nav)
        return df

    def temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:

        def extract_times(fi):
            return [int(t[1:]) for t in self.pattern_time.findall(fi)]

        def compute_speed(fi):
            times = extract_times(fi)
            if len(times) < 2:
                return -1, -1
            deltas = np.diff(times)
            speeds = np.arange(1, len(deltas) + 1) / deltas
            return np.mean(speeds), np.std(speeds)

        def compute_time_gaps(fi):
            times = extract_times(fi)
            if len(times) < 2:
                return -1, -1
            gaps = np.diff(times)
            return np.mean(gaps), np.std(gaps)

        def compute_time_entropy(fi):
            times = extract_times(fi)
            if len(times) < 3:
                return 0
            diffs = np.diff(times)
            probs = np.array(diffs) / np.sum(diffs)
            return entropy(probs)

        def compute_speed_trend(fi):
            times = extract_times(fi)
            if len(times) < 3:
                return 0
            gaps = np.diff(times)
            trend = np.polyfit(range(len(gaps)), gaps, 1)[0]
            return trend

        def compute_time_span(fi):
            times = extract_times(fi)
            return times[-1] - times[0] if len(times) > 1 else 0

        df[["speed_mean", "speed_std"]] = df["full_items"].apply(compute_speed).apply(pd.Series)
        df[["avg_gap", "std_gap"]] = df["full_items"].apply(compute_time_gaps).apply(pd.Series)
        df["time_entropy"] = df["full_items"].apply(compute_time_entropy)
        df["speed_trend"] = df["full_items"].apply(compute_speed_trend)
        df["time_span"] = df["full_items"].apply(compute_time_span)

        df["actions_per_second"] = df["n_actions"] / (df["time_span"] + 1)
        df["modules_per_minute"] = df["n_modules"] / ((df["time_span"] / 60) + 1)

        df["speed_mean"] = df["speed_mean"].clip(lower=0)
        df["speed_std"] = df["speed_std"].clip(lower=0)
        df["avg_gap"] = df["avg_gap"].clip(lower=0)
        df["std_gap"] = df["std_gap"].clip(lower=0)
        df["time_span"] = df["time_span"].clip(lower=0)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df


    def extract_chaines(self, fi: str):
        return re.findall(self.pattern_chaine, fi)

    def extract_actions(self, fi):
        tokens = fi.split(',')
        actions = [t.split('(')[0].split('1')[0] for t in tokens if not re.fullmatch(r"t\d+", t)]
        return actions

    def extract_full_path(self, fi):
        screens = re.findall(self.pattern_ecran, fi)
        return screens
    
    def extract_screen_configs(self, fi: str):
        configs = re.findall(self.pattern_conf_ecran, fi)
        return configs


    def action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        top_n = self.top_n_action
        df["derived_action"] = df["full_items"].apply(self.extract_actions)

        df["n_actions"] = df["derived_action"].apply(len)
        df["n_unique_actions"] = df["derived_action"].apply(lambda x: len(set(x)) if x else 0)
        df["action_repeat_ratio"] = df.apply(
            lambda row: 0 if row["n_actions"] == 0 else (1 - row["n_unique_actions"] / row["n_actions"]),
            axis=1,
        )
        df["action_entropy"] = df["derived_action"].apply(
            lambda x: 0 if not x else -sum((c / len(x)) * np.log2(c / len(x)) for c in Counter(x).values())
        )

        # --- Features catégorielles classiques ---
        df["first_action"] = df["derived_action"].apply(lambda x: x[0] if x else "NONE")
        df["second_action"] = df["derived_action"].apply(lambda x: x[1] if len(x) > 1 else "NONE")
        df["last_action"] = df["derived_action"].apply(lambda x: x[-1] if x else "NONE")
        df["penultimate_action"] = df["derived_action"].apply(lambda x: x[-2] if len(x) > 1 else "NONE")
        df["most_common_action"] = df["derived_action"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else "NONE"
        )
        df["second_common_action"] = df["derived_action"].apply(
            lambda x: Counter(x).most_common(2)[1][0] if len(Counter(x)) > 1 else "NONE"
        )

        categorical_cols = [
            "first_action", "second_action", "last_action", "penultimate_action",
            "most_common_action", "second_common_action"
        ]
        for col in categorical_cols:
            le = getattr(self, f"le_{col}", LabelEncoder())
            encoded = le.fit_transform(df[col].astype(str))
            setattr(self, f"le_{col}", le)
            df[col + "_enc"] = encoded

        # --- Top N actions ---
        all_actions = [a for actions in df["derived_action"] for a in actions]
        top_actions = [k for k, _ in Counter(all_actions).most_common(top_n)]

        for action in top_actions:
            df[f"count_{action}"] = df["derived_action"].apply(lambda x: x.count(action))

        return df


    def screen_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["derived_full_path"] = df["full_items"].apply(self.extract_full_path)

        df["n_modules"] = df["derived_full_path"].apply(lambda x: len(x))
        df["n_unique_modules"] = df["derived_full_path"].apply(lambda x: len(set(x)) if x else 0)
        df["module_repeat_ratio"] = df.apply(
            lambda row: 0 if row["n_modules"] == 0 else (1 - row["n_unique_modules"] / row["n_modules"]),
            axis=1,
        )

        df["first_module"] = df["derived_full_path"].apply(lambda x: x[0] if x else "NONE")
        df["last_module"] = df["derived_full_path"].apply(lambda x: x[-1] if x else "NONE")
        df["most_common_module"] = df["derived_full_path"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else "NONE"
        )

        df["n_module_transitions"] = df["derived_full_path"].apply(
            lambda x: sum(1 for i in range(1, len(x)) if x[i] != x[i - 1]) if len(x) > 1 else 0
        )

        categorical_cols = ["first_module", "last_module", "most_common_module"]

        for col in categorical_cols:
            le = getattr(self, f"le_{col}", LabelEncoder())
            encoded = le.fit_transform(df[col].astype(str))
            setattr(self, f"le_{col}", le)
            df[col + "_enc"] = encoded

        return df
    
    def get_transitions(self, path):
        if len(path) < 2:
            return []
        a, b = tee(path)
        next(b, None)
        return [f"{x}->{y}" for x, y in zip(a, b)]

    def screen_transition_tfidf_features(self, df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        df["derived_full_path"] = df["full_items"].apply(self.extract_full_path)

        df["screen_transition_text"] = df["derived_full_path"].apply(
            lambda x: " ".join(self.get_transitions(x))
        )

        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df["screen_transition_text"])

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{t}" for t in vectorizer.get_feature_names_out()],
            index=df.index
        )

        df = pd.concat([df, tfidf_df], axis=1)

        df.drop(columns=["screen_transition_text"], inplace=True)

        return df

    def full_path_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["derived_full_path"] = df["full_items"].apply(self.extract_full_path)

        df['screen_path'] = df['derived_full_path'].apply(lambda x: "->".join(x) if x else "NONE")

        le_path = LabelEncoder()
        df['screen_path_enc'] = le_path.fit_transform(df['screen_path'])
        self.le_screen_path = le_path

        df['path_length'] = df['derived_full_path'].apply(len)
        df['n_unique_screens'] = df['derived_full_path'].apply(lambda x: len(set(x)) if x else 0)
        df['unique_screens_ratio'] = df.apply(
            lambda row: 0 if row['path_length'] == 0 else row['n_unique_screens'] / row['path_length'],
            axis=1
        )
        df['n_screen_transitions'] = df['derived_full_path'].apply(
            lambda x: sum(1 for i in range(1, len(x)) if x[i] != x[i-1]) if len(x) > 1 else 0
        )

        return df


    def screen_config_features(self, df: pd.DataFrame) -> pd.DataFrame:
        top_n = self.top_n_screen_conf
        df["derived_screen_configs"] = df["full_items"].apply(self.extract_screen_configs)

        df["n_screen_configs"] = df["derived_screen_configs"].apply(len)
        df["n_unique_screen_configs"] = df["derived_screen_configs"].apply(lambda x: len(set(x)) if x else 0)
        df["screen_config_repeat_ratio"] = df.apply(
            lambda row: 0 if row["n_screen_configs"] == 0 else (1 - row["n_unique_screen_configs"] / row["n_screen_configs"]),
            axis=1,
        )

        df["first_screen_config"] = df["derived_screen_configs"].apply(lambda x: x[0] if x else "NONE")
        df["last_screen_config"] = df["derived_screen_configs"].apply(lambda x: x[-1] if x else "NONE")
        df["most_common_screen_config"] = df["derived_screen_configs"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else "NONE"
        )
        df["second_common_screen_config"] = df["derived_screen_configs"].apply(
            lambda x: Counter(x).most_common(2)[1][0] if len(Counter(x)) > 1 else "NONE"
        )

        categorical_cols = [
            "first_screen_config", "last_screen_config",
            "most_common_screen_config", "second_common_screen_config"
        ]
        for col in categorical_cols:
            le = getattr(self, f"le_{col}", LabelEncoder())
            encoded = le.fit_transform(df[col].astype(str))
            setattr(self, f"le_{col}", le)
            df[col + "_enc"] = encoded

        # --- Top N configurations ---
        all_configs = [c for configs in df["derived_screen_configs"] for c in configs]
        top_configs = [k for k, _ in Counter(all_configs).most_common(top_n)]

        for config in top_configs:
            df[f"count_{config}"] = df["derived_screen_configs"].apply(lambda x: x.count(config))

        return df

    def chaine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        top_n = self.top_n_chaine
        df["derived_chaines"] = df["full_items"].apply(self.extract_chaines)

        df["n_chaines"] = df["derived_chaines"].apply(len)
        df["n_unique_chaines"] = df["derived_chaines"].apply(lambda x: len(set(x)) if x else 0)
        df["chaine_repeat_ratio"] = df.apply(
            lambda row: 0 if row["n_chaines"] == 0 else (1 - row["n_unique_chaines"] / row["n_chaines"]),
            axis=1,
        )
        df["chaine_entropy"] = df["derived_chaines"].apply(
            lambda x: 0 if not x else -sum((c / len(x)) * np.log2(c / len(x)) for c in Counter(x).values())
        )

        df["first_chaine"] = df["derived_chaines"].apply(lambda x: x[0] if x else "NONE")
        df["last_chaine"] = df["derived_chaines"].apply(lambda x: x[-1] if x else "NONE")
        df["most_common_chaine"] = df["derived_chaines"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else "NONE"
        )

        categorical_cols = ["first_chaine", "last_chaine", "most_common_chaine"]
        for col in categorical_cols:
            le = getattr(self, f"le_{col}", LabelEncoder())
            encoded = le.fit_transform(df[col].astype(str))
            setattr(self, f"le_{col}", le)
            df[col + "_enc"] = encoded

        all_chaines = [c for chaines in df["derived_chaines"] for c in chaines]
        top_chaines = [k for k, _ in Counter(all_chaines).most_common(top_n)]

        for chaine in top_chaines:
            df[f"count_{chaine}"] = df["derived_chaines"].apply(lambda x: x.count(chaine))

        return df







