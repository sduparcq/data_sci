from src import pipeline

train_path="/data/train.csv"
test_path="/data/test.csv"


pipe_action = pipeline.Pipeline(
    test_path=test_path,
    train_path=train_path
)

pipe_temporal = pipeline.Pipeline(
    test_path=test_path,
    train_path=train_path
)

pipe_screen = pipeline.Pipeline(
    test_path=test_path,
    train_path=train_path
)

pipe_screen_config = pipeline.Pipeline(
    test_path=test_path,
    train_path=train_path
)

pipe_chaine = pipeline.Pipeline(
    test_path=test_path,
    train_path=train_path
)

pipes = [pipe_action, pipe_temporal, pipe_screen, pipe_screen_config, pipe_chaine]

for pipe in pipes:
    pipe.load_all_data()


features_at_test = ["action_features","temporal_features",
                    "screen_features", "screen_config_features",
                    "chaine_features"]


for i, features in enumerate(features_at_test):
    pipes[i].compute_features(feature_list=[features])
