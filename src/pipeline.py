from typing import List, Optional, Dict
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Pipeline:
    
    def __init__(self, test_path: str, train_path: str,
                 target_column: str = 'user'):
        self.target_column = target_column
        self.test_path = test_path
        self.train_path = train_path
        
        self.test_df = None
        self.train_df = None
        self.test_df_computed = None
        self.train_df_computed = None

        # Builders externes
        from .features import base_features, advanced_features
        from .utils import loader
        self.base_feature_builder = base_features.BaseFeatureBuilder()
        self.advanced_feature_builder = advanced_features.AdvancedFeatureBuilder()
        self.loader = loader

        self.label_encoder = None

    # ==========================================================
    #   DATA LOADING
    # ==========================================================
    def load_all_data(self):
        self.test_df = self.loader.load_data(path=self.test_path)
        self.train_df = self.loader.load_data(path=self.train_path)
        print("✅ Data loaded successfully")

    # ==========================================================
    #   FEATURE COMPUTATION
    # ==========================================================
    def compute_features(
        self,
        include_groups: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        use_advanced: bool = False
    ):
        """
        include_groups: liste parmi ['actions', 'screens', 'temporal', 'screen_config', 'chaine']
        exclude_features: liste de colonnes à exclure après génération
        use_advanced: bool -> True pour utiliser AdvancedFeatureBuilder au lieu du BaseFeatureBuilder
        """
        builder = self.advanced_feature_builder if use_advanced else self.base_feature_builder

        dfs = [self.train_df, self.test_df]
        output_df = []

        for df in dfs:
            loc_df = builder.build_features(
                df.copy(),
                include_groups=include_groups,
                exclude_features=exclude_features
            )
            output_df.append(loc_df)

        self.train_df_computed, self.test_df_computed = output_df
        print(f"✅ Features computed: {len(self.train_df_computed.columns)} columns in training set.")

    # ==========================================================
    #   CORRELATION ANALYSIS
    # ==========================================================
    def analyze_feature_correlation(self, threshold: float = 0.85, figsize=(12, 10)):
        if self.train_df_computed is None:
            raise ValueError("Les features n'ont pas encore été calculées. Lance compute_features() d'abord.")

        num_df = self.train_df_computed.select_dtypes(include=[np.number])

        if num_df.empty:
            print("Aucune colonne numérique trouvée pour la corrélation.")
            return None

        corr_matrix = num_df.corr()

        # Heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, cbar=True)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        # Optionnel : identification des features très corrélées
        correlated_pairs = (
            (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
             .stack()
             .reset_index())
        )
        correlated_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        correlated_pairs = correlated_pairs.query("abs(Correlation) > @threshold")

        print(f"🔍 {len(correlated_pairs)} pairs of features with correlation > {threshold}")
        print(f"correlated_pairs {correlated_pairs}")
        return correlated_pairs

    # ==========================================================
    #   PREPARATION (ENCODING)
    # ==========================================================
    def prepare_data(self):
        le = LabelEncoder()
        self.train_df_computed[self.target_column] = le.fit_transform(
            self.train_df_computed[self.target_column]
        )
        self.label_encoder = le
        print("✅ Data prepared (label encoding done).")

    # ==========================================================
    #   FEATURE RELEVANCE ANALYSIS
    # ==========================================================
    def analyze_feature_relevance(self, model=None, n_splits: int = 5):
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        df = self.train_df_computed.copy()
        X = df.drop(columns=[self.target_column])
        X = X.select_dtypes(include=["number", "bool"])
        y = df[self.target_column]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        feature_scores = []

        print("🔍 Starting feature relevance analysis...")
        for feature in X.columns:
            X_single = X[[feature]]
            scores = cross_val_score(model, X_single, y, cv=kf, scoring='accuracy')
            feature_scores.append({
                'feature': feature,
                'mean_accuracy': np.mean(scores),
                'std_accuracy': np.std(scores)
            })

        feature_scores_df = pd.DataFrame(feature_scores).sort_values('mean_accuracy', ascending=False)
        print("✅ Cross-validation done.")

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_scores_df, x='mean_accuracy', y='feature', orient='h')
        plt.title('Feature relevance (k-fold mean accuracy)')
        plt.xlabel('Mean CV Accuracy')
        plt.ylabel('Feature')
        plt.show()

        model.fit(X, y)
        perm_importance = permutation_importance(model, X, y, scoring='accuracy', n_repeats=10, random_state=42)

        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=perm_df, x='importance_mean', y='feature', orient='h')
        plt.title('Permutation Importance (RandomForest)')
        plt.xlabel('Mean decrease in accuracy')
        plt.ylabel('Feature')
        plt.show()

        print("✅ Permutation importance computed.")
        return feature_scores_df, perm_df


    def train_random_forest(self, exclude_features: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42):
        """
        Entraîne un RandomForest sur 80% des données et teste sur 20%.
        exclude_features: colonnes à exclure du training.
        """
        df = self.train_df_computed.copy()
        X = df.drop(columns=[self.target_column])
        if exclude_features:
            X = X.drop(columns=[col for col in exclude_features if col in X.columns])
        X = X.select_dtypes(include=["number", "bool"])
        y = df[self.target_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        print(f"✅ RandomForest trained. Accuracy on validation set: {acc:.4f}")
        
        self.rf_model = model
        self.rf_features = X.columns.tolist()
        return model, acc

    # ==========================================================
    #   PREDICTION SUR TEST DATA
    # ==========================================================
    def predict_test_data(self, exclude_features: Optional[List[str]] = None):
        """
        Prédit la totalité de la test data avec le modèle entraîné.
        """
        if not hasattr(self, 'rf_model'):
            raise ValueError("Le modèle RandomForest n'est pas encore entraîné. Lance train_random_forest() d'abord.")

        df = self.test_df_computed.copy()
        X_test = df
        if exclude_features:
            X_test = X_test.drop(columns=[col for col in exclude_features if col in X_test.columns])
        X_test = X_test[self.rf_features]  # garder uniquement les colonnes du training
        X_test = X_test.select_dtypes(include=["number", "bool"])

        preds = self.rf_model.predict(X_test)
        df['pred_user_enc'] = preds
        if self.label_encoder is not None:
            df['pred_user'] = self.label_encoder.inverse_transform(preds)
        else:
            df['pred_user'] = preds
        self.test_df_computed = df
        return df


    def train_random_forest_full(self, exclude_features: Optional[List[str]] = None, random_state: int = 42):
        """
        Entraîne un RandomForest sur la totalité du train set.
        """
        df = self.train_df_computed.copy()
        X = df.drop(columns=[self.target_column])
        if exclude_features:
            X = X.drop(columns=[col for col in exclude_features if col in X.columns])
        X = X.select_dtypes(include=["number", "bool"])
        y = df[self.target_column]

        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        model.fit(X, y)

        print(f"✅ RandomForest trained on full train set ({X.shape[0]} samples, {X.shape[1]} features).")

        self.rf_model = model
        self.rf_features = X.columns.tolist()
        return model

    # ==========================================================
    #   GENERATION FICHIER SUBMISSION
    # ==========================================================
    def generate_submission(self, output_path: str = "data/submission.csv"):
        if 'pred_user' not in self.test_df_computed.columns:
            raise ValueError("Les prédictions n'ont pas été faites. Lance predict_test_data() d'abord.")

        submission_df = pd.DataFrame({
            'RowId': np.arange(1, len(self.test_df_computed) + 1),
            'prediction': self.test_df_computed['pred_user']
        })

        submission_df.to_csv(output_path, index=False)
        print(f"✅ Submission saved to {output_path}")
