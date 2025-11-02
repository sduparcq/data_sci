# data_sci

## Overview
Data science workspace for classifying users from web interaction logs. The project builds structured features from raw session strings, evaluates several supervised models, and produces a competition-ready submission file.

## Repository Layout
- `data/` - raw CSV files (`train.csv`, `test.csv`, `sample_submission.csv`).
- `feature_builder.py` - feature engineering routines (event counters, timings, main labels, etc.).
- `extraction.py`, `dataset/` - legacy helpers for loading core datasets.
- `user_behavior_pipeline.ipynb` - main notebook with EDA, correlation/VIF filtering, model selection, and final submission steps.
- `embedding_combined.ipynb` - experiment mixing handcrafted features with TF-IDF embeddings.
- `artifacts/` - saved pipelines, encoders, and generated submissions.
- `requirements.txt` - Python dependencies for the environment.

## Getting Started
1. Create and activate a virtual environment (`python -m venv .venv` then `./.venv/Scripts/Activate.ps1` on Windows).
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure the competition CSV files are present in `data/`.
4. Launch Jupyter (`jupyter lab` or `jupyter notebook`) from the project root.

## Feature Engineering Pipeline
- `feature_builder.build_feature_dataframe` reads `train.csv` and constructs numerical and categorical features per session.
- Frequent actions, screens, configurations, and channels are selected on the train set to define stable one-hot columns shared with the test set.
- Correlation (>= 0.85) and VIF screening identify `drop_candidates` that are removed before modelling to reduce multicollinearity.

## Modelling Workflow
- The main notebook performs:
  - Exploratory data analysis and sanity checks.
  - Stratified `train_test_split` to compare Logistic Regression, SVM (linear and RBF), Random Forest, and XGBoost.
  - `GridSearchCV` with macro F1 scoring for each estimator.
  - Consolidation of metrics, optional confusion matrices, and SHAP interpretability when available.

## Full-Train Retraining and Submission
- The best pipeline from the grid search is cloned and refit on the full filtered dataset (`X`, `y`).
- Artifacts saved:
  - `artifacts/pipeline_full_train.joblib`
  - `artifacts/target_encoder_full.joblib`
- Test sessions are rebuilt with the same feature mappings, aligned to the filtered columns, and scored to produce `artifacts/behavior_submission.csv` (matching `RowId` / `prediction` schema).

## Notebook Execution Checklist
1. Run all cells in `user_behavior_pipeline.ipynb` up to the model comparison section.
2. Execute the "Final training and submission" cells to retrain on 100% of the data and write the submission file.
3. Inspect `artifacts/behavior_submission.csv` and the saved joblib files as needed.

## Additional Notes
- `embedding_combined.ipynb` demonstrates combining TF-IDF vectors with structured features; reuse the same correlation thresholds if you promote that workflow.
- Hyperparameters live in `model_specs` within the main notebook; adapt them to balance runtime and performance.
- Large CSVs remain in `data/` and are ignored by Git via `.gitignore`.
