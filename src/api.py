"""
Flask REST API — Fake Profile Detection System
===============================================
Endpoints:
  POST /api/predict          — single profile prediction
  POST /api/predict/batch    — batch prediction (list of profiles)
  GET  /api/health           — health check
  GET  /api/model/info       — model metadata & metrics
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from src.bcsModule    import compute_bcs, bcs_label, WEIGHTS
from src.preprocessing import ProfilePreprocessor, BASE_FEATURES

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

model        = None
preprocessor = None
report       = None


def load_artefacts():
    global model, preprocessor, report
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    prep_path  = os.path.join(MODEL_DIR, 'preprocessor.pkl')
    rep_path   = os.path.join(MODEL_DIR, 'report.json')

    if not os.path.exists(model_path):
        print("  [API] Model not found — running training pipeline ...")
        from src.model_training import train_and_evaluate
        model, preprocessor, report = train_and_evaluate(save_dir=MODEL_DIR)
        return

    model        = joblib.load(model_path)
    preprocessor = ProfilePreprocessor.load(prep_path)
    with open(rep_path) as f:
        report   = json.load(f)
    print(f"  [API] Loaded model: {report['best_model']}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

FIELD_DEFAULTS = {
    'posting_frequency':        2.0,
    'follower_following_ratio':  1.0,
    'account_age_days':         365,
    'avg_likes_per_post':       50.0,
    'avg_comments_per_post':    5.0,
    'bio_completeness':         0.5,
    'profile_pic_present':       1,
    'url_in_bio':                0,
    'verified':                  0,
    'posting_time_variance':    6.0,
    'avg_post_length':         120.0,
    'hashtag_ratio':             3.0,
    'mention_ratio':             1.0,
    'reply_consistency':        0.5,
    'content_diversity_score':  0.5,
}

def validate_and_fill(data: dict) -> dict:
    """Fill missing fields with defaults and cast types."""
    filled = {}
    for field, default in FIELD_DEFAULTS.items():
        val = data.get(field, default)
        filled[field] = float(val)
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction logic
# ─────────────────────────────────────────────────────────────────────────────

def _predict_single(profile: dict, threshold: float = 0.50) -> dict:
    profile_filled = validate_and_fill(profile)
    df = pd.DataFrame([profile_filled])
    df = compute_bcs(df)
    X  = preprocessor.transform(df)

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    bcs  = float(df['bcs_score'].iloc[0])

    # Risk level
    if prob < 0.30:
        risk = 'Low'
    elif prob < 0.60:
        risk = 'Medium'
    else:
        risk = 'High'

    return {
        'prediction':   'Fake' if pred else 'Genuine',
        'is_fake':       bool(pred),
        'probability':   round(prob, 4),
        'risk_level':    risk,
        'bcs_score':     round(bcs, 2),
        'bcs_label':     bcs_label(bcs),
        'sub_scores': {
            'posting_regularity':      round(float(df['sub_posting_regularity'].iloc[0]), 3),
            'engagement_authenticity': round(float(df['sub_engagement_authenticity'].iloc[0]), 3),
            'profile_completeness':    round(float(df['sub_profile_completeness'].iloc[0]), 3),
            'content_quality':         round(float(df['sub_content_quality'].iloc[0]), 3),
            'spam_signal':             round(float(df['sub_spam_signal'].iloc[0]), 3),
        },
        'bcs_weights': WEIGHTS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200


@app.route('/api/model/info', methods=['GET'])
def model_info():
    if report is None:
        return jsonify({'error': 'Model not loaded'}), 503
    best     = report['best_model']
    metrics  = report['all_results'][best]
    top_feat = list(report['feature_importance'].items())[:10]
    return jsonify({
        'best_model':     best,
        'metrics':        metrics,
        'top_features':   dict(top_feat),
        'all_models':     {k: v for k, v in report['all_results'].items()},
        'dataset_info':   report.get('dataset_info', {}),
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not ready'}), 503

    data      = request.get_json(force=True)
    threshold = float(data.pop('threshold', 0.50))

    try:
        result = _predict_single(data, threshold)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    if model is None:
        return jsonify({'error': 'Model not ready'}), 503

    data      = request.get_json(force=True)
    profiles  = data.get('profiles', [])
    threshold = float(data.get('threshold', 0.50))

    if not isinstance(profiles, list) or len(profiles) == 0:
        return jsonify({'error': 'profiles must be a non-empty list'}), 400
    if len(profiles) > 500:
        return jsonify({'error': 'Max 500 profiles per batch'}), 400

    results = []
    for i, profile in enumerate(profiles):
        try:
            r = _predict_single(profile, threshold)
            r['index'] = i
            results.append(r)
        except Exception as e:
            results.append({'index': i, 'error': str(e)})

    summary = {
        'total':   len(results),
        'fake':    sum(1 for r in results if r.get('is_fake')),
        'genuine': sum(1 for r in results if not r.get('is_fake') and 'error' not in r),
        'errors':  sum(1 for r in results if 'error' in r),
    }
    return jsonify({'summary': summary, 'results': results}), 200


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_artefacts()
    print("\n[API] Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)