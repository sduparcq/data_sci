import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.stats import entropy

class BaseFeatureBuilder:
    def __init__(self):
        self.pattern_time = re.compile(r"t\d*[05](?=,|$)")
        self.pattern_ecran = re.compile(r"\((.*?)\)")
        self.pattern_conf_ecran = re.compile(r"<(.*?)>")
        self.pattern_chaine = re.compile(r"\$(.*?)\$")
        self.pattern_actions = re.compile(r"(?!t\d+)[^,]+")

    def extract_actions(self, fi): 
        return [t.split('(')[0].split('1')[0] for t in fi.split(',') if not re.fullmatch(r"t\d+", t)]
    def extract_full_path(self, fi): 
        return re.findall(self.pattern_ecran, fi)
    def extract_screen_configs(self, fi): 
        return re.findall(self.pattern_conf_ecran, fi)
    def extract_chaines(self, fi): 
        return re.findall(self.pattern_chaine, fi)

    def nav(self, df):
        BROWSER_MAPPING = {
            "Google Chrome": 0,
            "Microsoft Edge": 1,
            "Opera": 2,
            "Firefox": 3,
        }
        df = df.copy()
        df['browser_enc'] = df['nav'].map(BROWSER_MAPPING)
        return df


    #region ACTION
    def action_features(self, df):
        df = df.copy()
        df["derived_action"] = df["full_items"].apply(self.extract_actions)
        
        # Statistiques de base
        df["n_actions"] = df["derived_action"].apply(len)
        df["n_unique_actions"] = df["derived_action"].apply(lambda x: len(set(x)))
        df["action_repeat_ratio"] = df.apply(
            lambda r: 0 if r["n_actions"] == 0 else 1 - r["n_unique_actions"]/r["n_actions"], axis=1
        )
        
        # Entropie
        df["action_entropy"] = df["derived_action"].apply(
            lambda x: 0 if not x else -sum((c/len(x))*np.log2(c/len(x)) for c in Counter(x).values())
        )
        
        # Encode catégories principales
        df["first_action"] = df["derived_action"].apply(lambda x: x[0] if x else "NONE")
        df["last_action"] = df["derived_action"].apply(lambda x: x[-1] if x else "NONE")
        df["most_common_action"] = df["derived_action"].apply(lambda x: Counter(x).most_common(1)[0][0] if x else "NONE")
        df["second_most_common_action"] = df["derived_action"].apply(
            lambda x: Counter(x).most_common(2)[1][0] if len(Counter(x))>1 else "NONE"
        )
        
        for col in ["first_action", "last_action", "most_common_action", "second_most_common_action"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        
        # Top-N actions
        all_actions = [a for actions in df["derived_action"] for a in actions]
        top_actions = [k for k,_ in Counter(all_actions).most_common(10)]
        for action in top_actions:
            df[f"count_{action}"] = df["derived_action"].apply(lambda x: x.count(action))
            df[f"freq_{action}"] = df.apply(lambda r: r["count_" + action]/r["n_actions"] if r["n_actions"]>0 else 0, axis=1)
        
        # Transitions
        def get_transitions(path):
            if len(path)<2: return []
            return [f"{a}->{b}" for a,b in zip(path[:-1], path[1:])]
        
        df["derived_transitions"] = df["derived_action"].apply(get_transitions)
        df["action_transition_entropy"] = df["derived_transitions"].apply(
            lambda x: 0 if not x else -sum((c/len(x))*np.log2(c/len(x)) for c in Counter(x).values())
        )
        
        df = df.drop(columns=["derived_action", "derived_transitions"])
        return df
    #endregion

    #region SCREEN
    def screen_features(self, df):
        df = df.copy()
        df["derived_full_path"] = df["full_items"].apply(self.extract_full_path)
        
        # Statistiques de base
        df["n_modules"] = df["derived_full_path"].apply(len)
        df["n_unique_modules"] = df["derived_full_path"].apply(lambda x: len(set(x)))
        df["module_repeat_ratio"] = df.apply(
            lambda r: 0 if r["n_modules"]==0 else 1 - r["n_unique_modules"]/r["n_modules"], axis=1
        )
        
        # Entropie
        df["module_entropy"] = df["derived_full_path"].apply(
            lambda x: 0 if not x else -sum((c/len(x))*np.log2(c/len(x)) for c in Counter(x).values())
        )
        
        # Encode catégories principales
        df["first_module"] = df["derived_full_path"].apply(lambda x: x[0] if x else "NONE")
        df["last_module"] = df["derived_full_path"].apply(lambda x: x[-1] if x else "NONE")
        df["most_common_module"] = df["derived_full_path"].apply(lambda x: Counter(x).most_common(1)[0][0] if x else "NONE")
        df["second_most_common_module"] = df["derived_full_path"].apply(lambda x: Counter(x).most_common(2)[1][0] if len(Counter(x))>1 else "NONE")
        
        for col in ["first_module", "last_module", "most_common_module", "second_most_common_module"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        
        # Top-N modules
        all_modules = [m for path in df["derived_full_path"] for m in path]
        top_modules = [k for k,_ in Counter(all_modules).most_common(10)]
        for module in top_modules:
            df[f"count_{module}"] = df["derived_full_path"].apply(lambda x: x.count(module))
            df[f"freq_{module}"] = df.apply(lambda r: r["count_" + module]/r["n_modules"] if r["n_modules"]>0 else 0, axis=1)
        
        # Transitions
        def get_transitions(path):
            if len(path)<2: return []
            return [f"{a}->{b}" for a,b in zip(path[:-1], path[1:])]
        
        df["derived_transitions"] = df["derived_full_path"].apply(get_transitions)
        df["module_transition_entropy"] = df["derived_transitions"].apply(
            lambda x: 0 if not x else -sum((c/len(x))*np.log2(c/len(x)) for c in Counter(x).values())
        )
        
        df = df.drop(columns=["derived_full_path","derived_transitions"])
        return df
    #endregion

    #region TEMPORAL
    def temporal_features(self, df):
        df = df.copy()
        
        def extract_times(fi):
            return [int(t[1:]) for t in self.pattern_time.findall(fi)]
        
        def compute_time_span(times):
            return times[-1] - times[0] if len(times) > 1 else 0
        
        def compute_gaps(times):
            if len(times) < 2: return np.array([0])
            return np.diff(times)
        
        def compute_entropy_gaps(gaps):
            if len(gaps) < 2: return 0
            probs = gaps / np.sum(gaps)
            return entropy(probs)
        
        def compute_speed(gaps):
            speeds = np.arange(1, len(gaps)+1) / gaps if len(gaps)>0 else [0]
            return np.mean(speeds), np.std(speeds)
        
        def compute_trend(gaps):
            if len(gaps)<2: return 0
            return np.polyfit(range(len(gaps)), gaps, 1)[0]
        
        time_stats = []
        for fi in df["full_items"]:
            times = extract_times(fi)
            gaps = compute_gaps(times)
            time_span = compute_time_span(times)
            avg_gap = np.mean(gaps) if len(gaps)>0 else 0
            gap_std = np.std(gaps) if len(gaps)>0 else 0
            gap_min = np.min(gaps) if len(gaps)>0 else 0
            gap_max = np.max(gaps) if len(gaps)>0 else 0
            gap_median = np.median(gaps) if len(gaps)>0 else 0
            gap_entropy = compute_entropy_gaps(gaps)
            speed_mean, speed_std = compute_speed(gaps)
            speed_trend = compute_trend(gaps)
            first_gap = gaps[0] if len(gaps)>0 else 0
            last_gap = gaps[-1] if len(gaps)>0 else 0
            n_short_gaps = np.sum(gaps<5) if len(gaps)>0 else 0
            n_long_gaps = np.sum(gaps>20) if len(gaps)>0 else 0
            time_stats.append([
                time_span, avg_gap, gap_std, gap_min, gap_max, gap_median,
                gap_entropy, speed_mean, speed_std, speed_trend, first_gap, last_gap,
                n_short_gaps, n_long_gaps
            ])
        
        columns = [
            "time_span", "avg_gap", "gap_std", "gap_min", "gap_max", "gap_median",
            "gap_entropy", "speed_mean", "speed_std", "speed_trend",
            "first_gap", "last_gap", "n_short_gaps", "n_long_gaps"
        ]
        
        df[columns] = pd.DataFrame(time_stats, index=df.index)
        
        df["actions_per_second"] = df["full_items"].apply(lambda fi: len(extract_times(fi))) / (df["time_span"] + 1)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    #endregion

    #region SCREEN_CONFIG
    def screen_config_features(self, df):
        df = df.copy()
        df["derived_screen_configs"] = df["full_items"].apply(self.extract_screen_configs)
        
        df["n_screen_configs"] = df["derived_screen_configs"].apply(len)
        df["n_unique_screen_configs"] = df["derived_screen_configs"].apply(lambda x: len(set(x)))
        df["screen_config_repeat_ratio"] = df.apply(
            lambda r: 0 if r["n_screen_configs"] == 0 else 1 - r["n_unique_screen_configs"]/r["n_screen_configs"], axis=1
        )

        # First / last / most common
        df["first_screen_config"] = df["derived_screen_configs"].apply(lambda x: x[0] if x else "NONE")
        df["last_screen_config"] = df["derived_screen_configs"].apply(lambda x: x[-1] if x else "NONE")
        df["first_last_match"] = (df["first_screen_config"] == df["last_screen_config"]).astype(int)

        df["most_common_screen_config"] = df["derived_screen_configs"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else "NONE"
        )
        df["most_common_screen_config_count"] = df["derived_screen_configs"].apply(
            lambda x: Counter(x).most_common(1)[0][1] if x else 0
        )
        df["most_common_screen_config_ratio"] = df.apply(
            lambda r: r["most_common_screen_config_count"] / r["n_screen_configs"] if r["n_screen_configs"] > 0 else 0,
            axis=1
        )

        # Entropie et rareté
        df["screen_config_entropy"] = df["derived_screen_configs"].apply(
            lambda x: 0 if not x else -sum((c/len(x))*np.log2(c/len(x)) for c in Counter(x).values())
        )
        df["n_configs_appearing_once"] = df["derived_screen_configs"].apply(
            lambda x: sum(1 for v in Counter(x).values() if v == 1)
        )

        # Transitions
        def n_transitions(seq):
            return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        
        def avg_streak(seq):
            if not seq: return 0
            streaks = []
            count = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    count += 1
                else:
                    streaks.append(count)
                    count = 1
            streaks.append(count)
            return np.mean(streaks)
        
        df["n_config_transitions"] = df["derived_screen_configs"].apply(n_transitions)
        df["avg_config_streak"] = df["derived_screen_configs"].apply(avg_streak)

        # Encodage catégoriel
        for col in ["first_screen_config", "last_screen_config", "most_common_screen_config"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

        return df.drop(columns=["derived_screen_configs"])

    #endregion

    #region CHAINE
    def chaine_features(self, df):
        df = df.copy()
        df["derived_chaines"] = df["full_items"].apply(self.extract_chaines)

        # Nombre et diversité
        df["n_chaines"] = df["derived_chaines"].apply(len)
        df["n_unique_chaines"] = df["derived_chaines"].apply(lambda x: len(set(x)))
        df["chaine_repeat_ratio"] = df.apply(
            lambda r: 0 if r["n_chaines"] == 0 else 1 - r["n_unique_chaines"]/r["n_chaines"], axis=1
        )

        # First / last / most common
        df["first_chaine"] = df["derived_chaines"].apply(lambda x: x[0] if x else "NONE")
        df["last_chaine"] = df["derived_chaines"].apply(lambda x: x[-1] if x else "NONE")
        df["first_last_match"] = (df["first_chaine"] == df["last_chaine"]).astype(int)

        df["most_common_chaine"] = df["derived_chaines"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else "NONE"
        )
        df["most_common_chaine_count"] = df["derived_chaines"].apply(
            lambda x: Counter(x).most_common(1)[0][1] if x else 0
        )
        df["most_common_chaine_ratio"] = df.apply(
            lambda r: r["most_common_chaine_count"] / r["n_chaines"] if r["n_chaines"] > 0 else 0,
            axis=1
        )

        # Entropie et rareté
        df["chaine_entropy"] = df["derived_chaines"].apply(
            lambda x: 0 if not x else -sum((c/len(x))*np.log2(c/len(x)) for c in Counter(x).values())
        )
        df["n_chaines_appearing_once"] = df["derived_chaines"].apply(
            lambda x: sum(1 for v in Counter(x).values() if v == 1)
        )

        # Transitions et streaks
        def n_transitions(seq):
            return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        
        def avg_streak(seq):
            if not seq: return 0
            streaks = []
            count = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    count += 1
                else:
                    streaks.append(count)
                    count = 1
            streaks.append(count)
            return np.mean(streaks)
        
        df["n_chaine_transitions"] = df["derived_chaines"].apply(n_transitions)
        df["avg_chaine_streak"] = df["derived_chaines"].apply(avg_streak)

        # Encodage catégoriel
        for col in ["first_chaine", "last_chaine", "most_common_chaine"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

        return df.drop(columns=["derived_chaines"])

    #endregion


    def build_features(self, df, include_groups=None, exclude_features=None):
        if include_groups is None:
            include_groups = ['nav', 'actions', 'screens', 'temporal', 'screen_config', 'chaine']
        if exclude_features is None:
            exclude_features = []

        df_out = df.copy()
        group_map = {
            'nav': self.nav,
            'actions': self.action_features,
            'screens': self.screen_features,
            'temporal': self.temporal_features,
            'screen_config': self.screen_config_features,
            'chaine': self.chaine_features
        }

        for g in include_groups:
            if g in group_map:
                df_out = group_map[g](df_out)

        to_drop = [col for col in exclude_features if col in df_out.columns]
        df_out.drop(columns=to_drop, inplace=True, errors="ignore")

        return df_out
