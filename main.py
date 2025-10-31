from src.pipeline import Pipeline


train_path="/data/train.csv"
test_path="/data/test.csv"

base_features = [
    'nav',
    'speed',
    'session_time',
    'global_features',
    "action_features",
    "screen_features",
    'screen_config_features',
    "chaine_features",
    "temporal_features",
    "full_path_features",
    "screen_transition_tfidf_features"
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
# print(pipe.train_df_computed.describe())

pipe.compute_advanced_features()



pipe.train_random_forest()


pipe.feature_correlation()
pipe.feature_summary()
pipe.feature_importance_analysis()

pipe.train_and_predict_full()
