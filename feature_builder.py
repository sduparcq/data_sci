import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from test3 import (
    extract_chaine,
    extract_conf_ecran,
    extract_ecran,
    extract_temps,
    filter_action,
    most_common_chaine,
    most_common_conf,
    most_common_ecran,
)

DATA_PATH = Path(__file__).parent / "data" / "train.csv"
BROWSER_MAPPING = {
    "Google Chrome": 0,
    "Microsoft Edge": 1,
    "Opera": 2,
    "Firefox": 3,
}
TOP_ACTIONS = 20
TOP_SCREENS = 15
TOP_CONFS = 10
TOP_CHAINES = 10


def slugify(value: str) -> str:
    base = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    base = base.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = base.strip("_")
    return base or "misc"


def build_label_map(labels: Sequence[str], prefix: str) -> List[Tuple[str, str]]:
    seen = set()
    mapping = []
    for label in labels:
        column = f"{prefix}__{slugify(label)}"
        if column in seen:
            suffix = 2
            while f"{column}_{suffix}" in seen:
                suffix += 1
            column = f"{column}_{suffix}"
        seen.add(column)
        mapping.append((label, column))
    return mapping


def load_sessions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) < 2:
                continue
            rows.append({
                "user": parts[0],
                "browser": parts[1],
                "events": [event for event in parts[2:] if event],
            })
    return rows


def is_time_token(token: str) -> bool:
    return extract_temps(token) is not None


def compute_top_labels(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    action_counter: Counter[str] = Counter()
    screen_counter: Counter[str] = Counter()
    conf_counter: Counter[str] = Counter()
    chaine_counter: Counter[str] = Counter()

    for row in rows:
        for event in row["events"]:
            if is_time_token(event):
                continue
            action = filter_action(event)
            if action:
                action_counter[action] += 1
            screen = extract_ecran(event)
            if screen:
                screen_counter[screen] += 1
            conf = extract_conf_ecran(event)
            if conf:
                conf_counter[conf] += 1
            chaine = extract_chaine(event)
            if chaine:
                chaine_counter[chaine] += 1

    top_actions = [label for label, _ in action_counter.most_common(TOP_ACTIONS)]
    top_screens = [label for label, _ in screen_counter.most_common(TOP_SCREENS)]
    top_confs = [label for label, _ in conf_counter.most_common(TOP_CONFS)]
    top_chaines = [label for label, _ in chaine_counter.most_common(TOP_CHAINES)]

    return top_actions, top_screens, top_confs, top_chaines


def keyword_counts(actions: Sequence[str], keyword: str) -> int:
    keyword_lower = keyword.lower()
    return sum(1 for action in actions if keyword_lower in action.lower())


def compute_time_metrics(times: Sequence[int]) -> Dict[str, float]:
    if not times:
        return {
            "time_total": 0.0,
            "time_mean": 0.0,
            "time_std": 0.0,
            "time_min": 0.0,
            "time_max": 0.0,
            "time_span": 0.0,
            "time_count": 0,
            "avg_time_gap": 0.0,
        }

    times_sorted = sorted(times)
    total = float(sum(times_sorted))
    count = len(times_sorted)
    min_value = float(times_sorted[0])
    max_value = float(times_sorted[-1])
    mean_value = total / count
    std_value = float(pstdev(times_sorted)) if count > 1 else 0.0
    span = max_value - min_value
    avg_gap = span / (count - 1) if count > 1 else 0.0

    return {
        "time_total": total,
        "time_mean": mean_value,
        "time_std": std_value,
        "time_min": min_value,
        "time_max": max_value,
        "time_span": span,
        "time_count": count,
        "avg_time_gap": avg_gap,
    }




def compute_mean_speed(events: Sequence[str]) -> float:
    speeds: List[float] = []
    actions_since_last = 0
    last_time = 0

    for event in events:
        if is_time_token(event):
            time_value = extract_temps(event)
            if time_value is None:
                continue
            delta = time_value - last_time
            if delta > 0:
                speeds.append(actions_since_last / delta)
            last_time = time_value
            actions_since_last = 0
        else:
            actions_since_last += 1

    if speeds:
        return float(sum(speeds) / len(speeds))
    return 0.0


def compute_row_features(
    row: Dict[str, Any],
    action_labels: Sequence[Tuple[str, str]],
    screen_labels: Sequence[Tuple[str, str]],
    conf_labels: Sequence[Tuple[str, str]],
    chaine_labels: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    events = row["events"]
    time_values: List[int] = []
    action_events: List[str] = []

    for event in events:
        if is_time_token(event):
            time_value = extract_temps(event)
            if time_value is not None:
                time_values.append(time_value)
            continue
        action_events.append(event)

    cleaned_actions: List[str] = []
    screens: List[str] = []
    confs: List[str] = []
    chaines: List[str] = []

    for event in action_events:
        action_value = filter_action(event)
        if action_value:
            cleaned_actions.append(action_value)
        screen_value = extract_ecran(event)
        if screen_value:
            screens.append(screen_value)
        conf_value = extract_conf_ecran(event)
        if conf_value:
            confs.append(conf_value)
        chaine_value = extract_chaine(event)
        if chaine_value:
            chaines.append(chaine_value)

    action_counter = Counter(cleaned_actions)
    screen_counter = Counter(screens)
    conf_counter = Counter(confs)
    chaine_counter = Counter(chaines)

    time_metrics = compute_time_metrics(time_values)
    mean_speed = compute_mean_speed(events)
    total_tokens = len(events)
    nb_actions = len(action_events)
    nb_unique_actions = len(action_counter)
    ratio_unique_actions = nb_unique_actions / nb_actions if nb_actions else 0.0

    actions_series = pd.Series(action_events, dtype="object")
    main_action_data = action_counter.most_common(1)
    main_action = main_action_data[0][0] if main_action_data else None
    main_action_share = (
        main_action_data[0][1] / nb_actions if nb_actions and main_action_data else 0.0
    )

    features: Dict[str, Any] = {
        "user": row["user"],
        "browser": row["browser"],
        "browser_code": BROWSER_MAPPING.get(row["browser"], -1),
        "total_tokens": total_tokens,
        "nb_actions": nb_actions,
        "nb_time_marks": time_metrics["time_count"],
        "nb_unique_actions": nb_unique_actions,
        "ratio_unique_actions": ratio_unique_actions,
        "nb_screens": len(screens),
        "nb_unique_screens": len(screen_counter),
        "nb_confs": len(confs),
        "nb_unique_confs": len(conf_counter),
        "nb_chaines": len(chaines),
        "nb_unique_chaines": len(chaine_counter),
        "vitesse_moyenne": mean_speed,
        "count_dialogue_actions": keyword_counts(cleaned_actions, "dialogue"),
        "count_bouton_actions": keyword_counts(cleaned_actions, "bouton"),
        "count_saisie_actions": keyword_counts(cleaned_actions, "saisie"),
        "count_toast_actions": keyword_counts(cleaned_actions, "toast"),
        "count_filtrage_actions": keyword_counts(cleaned_actions, "filtrage"),
        "count_double_clic_actions": keyword_counts(cleaned_actions, "double-clic"),
        "main_action": main_action,
        "main_action_share": main_action_share,
        "main_screen": most_common_ecran(actions_series),
        "main_conf": most_common_conf(actions_series),
        "main_chaine": most_common_chaine(actions_series),
        "actions_per_time_span": (
            nb_actions / time_metrics["time_span"] if time_metrics["time_span"] else float(nb_actions)
        ),
    }

    features.update(time_metrics)

    for label, column in action_labels:
        features[column] = action_counter.get(label, 0)
    for label, column in screen_labels:
        features[column] = screen_counter.get(label, 0)
    for label, column in conf_labels:
        features[column] = conf_counter.get(label, 0)
    for label, column in chaine_labels:
        features[column] = chaine_counter.get(label, 0)

    return features


def build_feature_dataframe(data_path: Path = DATA_PATH) -> pd.DataFrame:
    sessions = load_sessions(data_path)
    top_actions, top_screens, top_confs, top_chaines = compute_top_labels(sessions)
    action_labels = build_label_map(top_actions, "action_count")
    screen_labels = build_label_map(top_screens, "screen_count")
    conf_labels = build_label_map(top_confs, "conf_count")
    chaine_labels = build_label_map(top_chaines, "chaine_count")

    records = [
        compute_row_features(session, action_labels, screen_labels, conf_labels, chaine_labels)
        for session in sessions
    ]

    return pd.DataFrame(records)


if __name__ == "__main__":
    df_features = build_feature_dataframe()
    print(df_features.head())
    print(f"Shape: {df_features.shape}")



