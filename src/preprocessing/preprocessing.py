import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class Preprocessor:
    """
    Handles data preprocessing: imputation and scaling.
    """
    def __init__(self, numerical_features: list, categorical_features: list):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Define transformers
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numerical_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])
            
    def fit(self, X: pd.DataFrame):
        """Fit the preprocessor on the data."""
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        X_processed = self.preprocessor.transform(X)
        # Convert back to DataFrame (if sparse matrix is returned, convert to dense)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        # Get feature names if available (for debugging/interpretability)
        try:
            num_names = self.numerical_features
            cat_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_features)
            feature_names = np.concatenate([num_names, cat_names])
            return pd.DataFrame(X_processed, columns=feature_names)
        except:
             return pd.DataFrame(X_processed)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X).transform(X)

    def save(self, filepath: str):
        """Save the fitted preprocessor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.preprocessor, filepath)

    def load(self, filepath: str):
        """Load a fitted preprocessor."""
        self.preprocessor = joblib.load(filepath)
        return self
