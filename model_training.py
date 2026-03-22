"""
Model Training & Evaluation
============================
Trains multiple classifiers and selects the best one.
  - Random Forest (primary, as described in project)
  - Gradient Boosting
  - Logistic Regression (baseline)
  - Support Vector Machine

Saves the best model + metrics report.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                        f1_score, roc_auc_score, confusion_matrix,
                                        classification_report)
from sklearn.model_selection   import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from data_generator  import generate_dataset
from bcs_module      import compute_bcs
from preprocessing   import ProfilePreprocessor, prepare_splits, TARGET


# ─────────────────────────────────────────────────────────────────────────────
# Model zoo
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1_score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

from preprocessing import ALL_FEATURES

def get_feature_importance(model, feature_names=ALL_FEATURES):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return dict(sorted(zip(feature_names, importances.tolist()),
                            key=lambda x: x[1], reverse=True))
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        return dict(sorted(zip(feature_names, importances.tolist()),
                            key=lambda x: x[1], reverse=True))
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(n_samples=2000, save_dir="../models"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(" FAKE PROFILE DETECTION — MODEL TRAINING")
    print("=" * 60)

    # ── Data preparation ──────────────────────────────────────────
    print("\n[1/4] Generating & preprocessing data ...")
    df           = generate_dataset(n_samples)
    df           = compute_bcs(df)
    train, val, test = prepare_splits(df)

    prep         = ProfilePreprocessor()
    X_train      = prep.fit_transform(train)
    X_val        = prep.transform(val)
    X_test       = prep.transform(test)
    y_train      = train[TARGET].values
    y_val        = val[TARGET].values
    y_test       = test[TARGET].values

    prep.save(os.path.join(save_dir, "preprocessor.pkl"))
    print(f"   Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    # ── Cross-validation ──────────────────────────────────────────
    print("\n[2/4] 5-fold cross-validation on training set ...")
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, model in MODELS.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"   {name:<25} CV F1 = {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Train all models ──────────────────────────────────────────
    print("\n[3/4] Training all models on full train set ...")
    results = {}
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        metrics            = evaluate_model(model, X_test, y_test)
        metrics['cv_f1']   = round(cv_scores[name], 4)
        results[name]      = metrics
        trained[name]      = model
        print(f"   {name:<25} Test AUC={metrics['roc_auc']}  F1={metrics['f1_score']}  Acc={metrics['accuracy']}")

    # ── Select best model ─────────────────────────────────────────
    best_name  = max(results, key=lambda n: results[n]['roc_auc'])
    best_model = trained[best_name]
    print(f"\n   ✅  Best model: {best_name}  (AUC={results[best_name]['roc_auc']})")

    # ── Save artefacts ────────────────────────────────────────────
    print("\n[4/4] Saving model artefacts ...")
    joblib.dump(best_model, os.path.join(save_dir, "best_model.pkl"))

    feature_imp = get_feature_importance(best_model)
    report = {
        'best_model':        best_name,
        'all_results':       results,
        'feature_importance': feature_imp,
        'dataset_info': {
            'total': n_samples,
            'train': len(X_train),
            'val':   len(X_val),
            'test':  len(X_test),
        }
    }
    with open(os.path.join(save_dir, "report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   Saved: {save_dir}/best_model.pkl")
    print(f"   Saved: {save_dir}/report.json")
    print("\n" + "=" * 60)
    return best_model, prep, report


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_profile(profile_dict: dict, model, preprocessor: ProfilePreprocessor,
                    threshold=0.50):
    """
    Predict a single profile.

    profile_dict : dict with all BASE_FEATURES keys
    Returns      : dict with prediction, probability, and BCS
    """
    from bcs_module import compute_bcs, bcs_label
    df   = pd.DataFrame([profile_dict])
    df   = compute_bcs(df)
    X    = preprocessor.transform(df)
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= threshold)
    return {
        'prediction':  'Fake' if pred else 'Genuine',
        'probability': round(float(prob), 4),
        'bcs_score':   round(float(df['bcs_score'].iloc[0]), 2),
        'bcs_label':   bcs_label(float(df['bcs_score'].iloc[0])),
    }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, prep, report = train_and_evaluate()

    # Quick demo prediction
    sample = {
        'posting_frequency': 18.0, 'follower_following_ratio': 0.05,
        'account_age_days': 45,    'avg_likes_per_post': 3,
        'avg_comments_per_post': 1,'bio_completeness': 0.1,
        'profile_pic_present': 0,  'url_in_bio': 1,
        'verified': 0,             'posting_time_variance': 0.8,
        'avg_post_length': 20,     'hashtag_ratio': 12,
        'mention_ratio': 6,        'reply_consistency': 0.1,
        'content_diversity_score': 0.1,
    }
    result = predict_profile(sample, model, prep)
    print("\nSample prediction (suspicious profile):", result)
