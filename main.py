"""
main.py — Fake Profile Detection System
========================================
Entry point:
  python main.py train       — train models & save artefacts
  python main.py evaluate    — evaluate + generate all plots
  python main.py predict     — interactive CLI predictor
  python main.py api         — start Flask REST API
  python main.py demo        — full end-to-end demo
"""

import os
import sys
import argparse

# ── ensure src/ is importable ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)

os.makedirs(os.path.join(BASE_DIR, 'data'),            exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'),          exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'outputs/figures'), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_train(n_samples=2000):
    from model_training import train_and_evaluate
    model, prep, report = train_and_evaluate(
        n_samples=n_samples,
        save_dir=os.path.join(BASE_DIR, 'models')
    )
    return model, prep, report


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate():
    import json, joblib
    import pandas as pd
    from data_generator  import generate_dataset
    from bcs_module      import compute_bcs
    from preprocessing   import ProfilePreprocessor, prepare_splits, TARGET
    from model_training  import MODELS
    from visualization   import generate_all_plots

    model_dir = os.path.join(BASE_DIR, 'models')
    report_path = os.path.join(model_dir, 'report.json')

    if not os.path.exists(report_path):
        print("[!] No trained model found. Running training first ...")
        run_train()

    with open(report_path) as f:
        report = json.load(f)

    best_model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    prep       = ProfilePreprocessor.load(os.path.join(model_dir, 'preprocessor.pkl'))

    df          = generate_dataset(2000)
    df          = compute_bcs(df)
    _, _, test  = prepare_splits(df)
    X_test      = prep.transform(test)
    y_test      = test[TARGET].values

    # Re-train all models for ROC comparison
    from preprocessing import ALL_FEATURES
    _, _, train_df = prepare_splits(df)   # note: using same seed so splits match
    train_df, _, _ = prepare_splits(df)
    X_train        = prep.fit_transform(train_df)

    trained_models = {}
    for name, m in MODELS.items():
        m.fit(X_train, train_df[TARGET].values)
        trained_models[name] = m
    trained_models[report['best_model']] = best_model   # override with saved best

    generate_all_plots(
        trained_models, X_test, y_test, df, report,
        out_dir=os.path.join(BASE_DIR, 'outputs/figures')
    )
    print("\n✅  All evaluation plots saved to outputs/figures/")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT (interactive CLI)
# ─────────────────────────────────────────────────────────────────────────────

def run_predict():
    import json, joblib
    from preprocessing  import ProfilePreprocessor
    from model_training import predict_profile

    model_dir  = os.path.join(BASE_DIR, 'models')
    model      = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    prep       = ProfilePreprocessor.load(os.path.join(model_dir, 'preprocessor.pkl'))

    print("\n" + "═" * 55)
    print(" FAKE PROFILE DETECTION — Interactive Predictor")
    print("═" * 55)
    print("Enter profile values (press Enter to use default):\n")

    defaults = {
        'posting_frequency':        2.0,
        'follower_following_ratio':  1.0,
        'account_age_days':         365,
        'avg_likes_per_post':       50.0,
        'avg_comments_per_post':    5.0,
        'bio_completeness':         0.5,
        'profile_pic_present':       1.0,
        'url_in_bio':                0.0,
        'verified':                  0.0,
        'posting_time_variance':    6.0,
        'avg_post_length':         120.0,
        'hashtag_ratio':             3.0,
        'mention_ratio':             1.0,
        'reply_consistency':        0.5,
        'content_diversity_score':  0.5,
    }

    profile = {}
    for field, default in defaults.items():
        val = input(f"  {field:<30} [{default}]: ").strip()
        profile[field] = float(val) if val else default

    result = predict_profile(profile, model, prep)
    print("\n" + "─" * 45)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']:.4f}")
    print(f"  BCS Score   : {result['bcs_score']} / 100  ({result['bcs_label']})")
    print("─" * 45 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

def run_api():
    from api import app, load_artefacts
    load_artefacts()
    app.run(debug=False, host='0.0.0.0', port=5000)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "═" * 60)
    print("  FAKE PROFILE DETECTION — Full Demo")
    print("═" * 60)

    model, prep, report = run_train(n_samples=1000)
    run_evaluate()

    from model_training import predict_profile

    samples = [
        {   # Genuine-looking profile
            'posting_frequency': 2.3, 'follower_following_ratio': 2.5,
            'account_age_days': 900,  'avg_likes_per_post': 120,
            'avg_comments_per_post': 15, 'bio_completeness': 0.85,
            'profile_pic_present': 1, 'url_in_bio': 0, 'verified': 0,
            'posting_time_variance': 7.5, 'avg_post_length': 150,
            'hashtag_ratio': 2.5, 'mention_ratio': 0.8,
            'reply_consistency': 0.75, 'content_diversity_score': 0.8,
        },
        {   # Obvious bot
            'posting_frequency': 22, 'follower_following_ratio': 0.03,
            'account_age_days': 14,  'avg_likes_per_post': 1,
            'avg_comments_per_post': 0, 'bio_completeness': 0.05,
            'profile_pic_present': 0, 'url_in_bio': 1, 'verified': 0,
            'posting_time_variance': 0.3, 'avg_post_length': 18,
            'hashtag_ratio': 14, 'mention_ratio': 7,
            'reply_consistency': 0.05, 'content_diversity_score': 0.05,
        },
    ]

    labels = ["👤 Genuine-looking profile", "🤖 Obvious bot profile"]
    print("\n── Sample Predictions ──────────────────────────────────")
    for label, sample in zip(labels, samples):
        result = predict_profile(sample, model, prep)
        print(f"\n{label}")
        print(f"  → {result['prediction']:10}  prob={result['probability']:.3f}  BCS={result['bcs_score']}/100  ({result['bcs_label']})")

    print("\n✅  Demo complete. Check outputs/figures/ for plots.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake Profile Detection System')
    parser.add_argument('mode', nargs='?', default='demo',
                        choices=['train', 'evaluate', 'predict', 'api', 'demo'],
                        help='Execution mode (default: demo)')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of samples to generate (default: 2000)')
    args = parser.parse_args()

    if   args.mode == 'train':    run_train(args.samples)
    elif args.mode == 'evaluate': run_evaluate()
    elif args.mode == 'predict':  run_predict()
    elif args.mode == 'api':      run_api()
    else:                         run_demo()