from src.pipeline import Pipeline


train_path="/data/train.csv"
test_path="/data/test.csv"

base_features = [
    'nav',
    'speed',
    'session_time',
    'nb_action'
]

advanced_features = [
    'action_per_tspan'
]

pipe = Pipeline(
    test_path=test_path,
    train_path=train_path,
    base_features_list=base_features,
    advanced_features_list=advanced_features
)


pipe.load_all_data()

# pipe.clear_cache()

pipe.compute_base_features()
pipe.compute_advanced_features()


pipe.train_random_forest()
