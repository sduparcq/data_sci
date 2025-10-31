from typing import List, Literal

from .utils import loader
from .features import base_features, advanced_features


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Pipeline():
    
    def __init__(self, test_path: str, train_path: str,
                base_features_list: List[str], advanced_features_list: List[str],
                target_column: str = 'user'):
        self.target_column = target_column

        self.test_path = test_path
        self.train_path = train_path
        
        # Placeholder pour les dataframes de base
        self.test_df = None
        self.train_df = None

        # Placeholder pour les dataframes avec ttes les features computed
        self.test_df_computed = None
        self.train_df_computed = None

        self.base_features_list = base_features_list
        self.advanced_features_list = advanced_features_list

        self.base_feature_builder = base_features.BaseFeatureBuilder()
        self.advanced_feature_builder = advanced_features.AdvancedFeatureBuilder()

    def load_all_data(self):
        self.test_df = loader.load_data(path=self.test_path)
        self.train_df = loader.load_data(path=self.train_path)
        print("Data loaded successfully")

    def clear_cache(self):
        self.advanced_feature_builder.clear_cache()
        self.base_feature_builder.clear_cache()
        print("Both cache cleared")


    #region Features creation
    def compute_base_features(self):
        dfs = [self.train_df, self.test_df]
        output_df = []
        for df in dfs:
            loc_df = df
            for feature in self.base_features_list:
                if hasattr(self.base_feature_builder, feature):
                    loc_df = getattr(self.base_feature_builder, feature)(loc_df)
            output_df.append(loc_df)
        self.train_df_computed = output_df[0] 
        self.test_df_computed = output_df[1] 



    def compute_advanced_features(self):
        if self.train_df_computed is None or self.test_df_computed is None:
            print("Compute base features first")
            return
        
        dfs = [self.train_df_computed.copy(), self.test_df_computed.copy()]
        output_df = []
        for df in dfs:
            loc_df = df
            for feature in self.advanced_features_list:
                if hasattr(self.advanced_feature_builder, feature):
                    loc_df = getattr(self.advanced_feature_builder, feature)(loc_df)
            output_df.append(loc_df)
        self.train_df_computed = output_df[0]
        self.test_df_computed = output_df[1]
    
    #endregion

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        df = self.train_df_computed.copy()

        le = LabelEncoder()
        df[self.target_column] = le.fit_transform(df[self.target_column])
        self.label_encoder = le

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        X = df[numeric_cols].copy()
        y = df[self.target_column].copy()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"✅ Data prepared successfully — {X_train.shape[1]} numeric features retained.")
        return X_train, X_val, y_train, y_val

    def train_random_forest(self, **kwargs):
        X_train, X_val, y_train, y_val = self.prepare_data()

        self.model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            bootstrap=kwargs.get("bootstrap", True),
            max_samples=kwargs.get("max_samples", None),
            random_state=kwargs.get("random_state", 42)
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"🌲 Random Forest accuracy: {acc:.4f}")

        return self.model

    def train_xgboost(self, **kwargs):
        X_train, X_val, y_train, y_val = self.prepare_data()

        self.model = XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 0.8),        # sous-échantillonnage = bagging
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),  # bagging des features
            random_state=kwargs.get("random_state", 42),
            use_label_encoder=False,
            eval_metric="logloss"
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"⚡ XGBoost accuracy: {acc:.4f}")

        return self.model

    #region Analysis

    def feature_correlation(self, df=None, top_n=20, plot=True):
        if df is None:
            df = self.train_df_computed.copy()
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = df[numeric_cols].corr()

        if plot:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title("Heatmap des corrélations entre features")
            plt.show()

        corr_matrix_abs = corr_matrix.abs()
        corr_matrix_abs.values[[np.arange(len(corr_matrix))]*2] = 0  # ignorer diag
        top_corr = corr_matrix_abs.unstack().sort_values(ascending=False).drop_duplicates()
        print(f"Top {top_n} corrélations entre features :\n", top_corr.head(top_n))
        return corr_matrix, top_corr.head(top_n)

    def feature_importance_analysis(
        self, top_n: int = 20, method: str = "model", X_train=None, X_val=None, y_train=None, y_val=None
    ):
        if X_train is None or X_val is None or y_train is None or y_val is None:
            # reprendre les data déjà prepared plutot que re run prepared data ici ?
            # sinon mettre un cache peut-être utile
            X_train, X_val, y_train, y_val = self.prepare_data()

        if not hasattr(self, "model"):
            raise ValueError("Aucun modèle entraîné. Entraînez le modèle avant d'analyser les features.")

        if method == "model":
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                fi_df = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances
                }).sort_values(by="importance", ascending=False)
            else:
                raise ValueError("Le modèle ne supporte pas l'attribut feature_importances_.")

        elif method == "permutation":
            result = permutation_importance(self.model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
            fi_df = pd.DataFrame({
                "feature": X_val.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std
            }).sort_values(by="importance_mean", ascending=False)

        else:
            raise ValueError("Méthode inconnue. Choisir 'model' ou 'permutation'.")

        print(f"\nTop {top_n} features les plus importantes:")
        print(fi_df.head(top_n))

        plt.figure(figsize=(10,6))
        sns.barplot(
            x=fi_df.iloc[:top_n, 1] if method=="model" else fi_df.iloc[:top_n]["importance_mean"],
            y=fi_df.iloc[:top_n]["feature"],
            palette="viridis"
        )
        plt.title(f"Top {top_n} feature importances ({method})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

        return fi_df
    
    def feature_summary(self, df=None):
        if df is None:
            df = self.train_df_computed.copy()
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        summary = df[numeric_cols].describe().T
        summary["missing"] = df[numeric_cols].isna().sum()
        summary["inf"] = np.isinf(df[numeric_cols]).sum()
        return summary
    
    #endregion

    def predict(self, df):
        if not hasattr(self, "model"):
            raise ValueError("Aucun modèle entraîné. Utilisez train_random_forest() ou train_xgboost() d'abord.")
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        X = df[numeric_cols].copy()
        return self.model.predict(X)



