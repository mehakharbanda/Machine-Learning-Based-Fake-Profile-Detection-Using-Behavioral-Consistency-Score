"""
Data Preprocessing & Feature Engineering Pipeline
==================================================
Handles:
  - Missing value imputation
  - Outlier capping (IQR)
  - Feature scaling (StandardScaler / RobustScaler)
  - BCS integration as an engineered feature
  - Train / validation / test splitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# Feature columns used for modelling (excludes label and BCS sub-scores)
BASE_FEATURES = [
    'posting_frequency',
    'follower_following_ratio',
    'account_age_days',
    'avg_likes_per_post',
    'avg_comments_per_post',
    'bio_completeness',
    'profile_pic_present',
    'url_in_bio',
    'verified',
    'posting_time_variance',
    'avg_post_length',
    'hashtag_ratio',
    'mention_ratio',
    'reply_consistency',
    'content_diversity_score',
]

# BCS sub-score columns injected as extra features
BCS_FEATURES = [
    'sub_posting_regularity',
    'sub_engagement_authenticity',
    'sub_profile_completeness',
    'sub_content_quality',
    'sub_spam_signal',
    'bcs_score',
]

ALL_FEATURES = BASE_FEATURES + BCS_FEATURES
TARGET       = 'label'


# ─────────────────────────────────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, columns: list, factor: float = 3.0) -> pd.DataFrame:
    """Cap values beyond ±factor × IQR from Q1/Q3."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr    = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            df[col] = df[col].clip(lower, upper)
    return df


class ProfilePreprocessor:
    """
    End-to-end preprocessing pipeline that:
      1. Imputes missing values
      2. Caps outliers
      3. Scales features with RobustScaler (resistant to outliers)
    """

    def __init__(self):
        self.imputer  = SimpleImputer(strategy='median')
        self.scaler   = RobustScaler()
        self._fitted  = False

    def fit_transform(self, df: pd.DataFrame):
        """Fit on training data and return scaled array + feature names."""
        df = cap_outliers(df, BASE_FEATURES + ['bcs_score'])

        X = df[ALL_FEATURES].values
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df = cap_outliers(df, BASE_FEATURES + ['bcs_score'])
        X  = df[ALL_FEATURES].values
        X  = self.imputer.transform(X)
        X  = self.scaler.transform(X)
        return X

    def save(self, path: str):
        joblib.dump({'imputer': self.imputer, 'scaler': self.scaler, 'fitted': self._fitted}, path)

    @classmethod
    def load(cls, path: str):
        obj  = cls()
        data = joblib.load(path)
        obj.imputer  = data['imputer']
        obj.scaler   = data['scaler']
        obj._fitted  = data['fitted']
        return obj


# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(df: pd.DataFrame, test_size=0.20, val_size=0.10, random_state=42):
    """
    Returns train / val / test DataFrames stratified on the label.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    adjusted_val = val_size / (1 - test_size)
    train, val   = train_test_split(
        train_val, test_size=adjusted_val, stratify=train_val[TARGET], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_dataset
    from bcs_module      import compute_bcs

    os.makedirs("../data",    exist_ok=True)
    os.makedirs("../models",  exist_ok=True)

    df     = generate_dataset(2000)
    df     = compute_bcs(df)
    train, val, test = prepare_splits(df)

    prep   = ProfilePreprocessor()
    X_train = prep.fit_transform(train)
    X_val   = prep.transform(val)
    X_test  = prep.transform(test)

    prep.save("../models/preprocessor.pkl")

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print("Preprocessor saved to models/preprocessor.pkl")
