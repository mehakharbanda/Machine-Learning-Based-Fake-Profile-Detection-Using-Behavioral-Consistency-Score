"""
Synthetic Dataset Generator for Fake Profile Detection
Generates realistic social media profile behavioral data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

def generate_dataset(n_samples=2000):
    """
    Generate synthetic social media profile dataset with behavioral features.
    
    Features:
    - posting_frequency        : avg posts per day
    - follower_following_ratio : followers / following
    - account_age_days         : how old the account is
    - avg_likes_per_post       : average likes received
    - avg_comments_per_post    : average comments received
    - bio_completeness         : 0–1 score for profile bio
    - profile_pic_present      : 0 or 1
    - url_in_bio               : 0 or 1
    - verified                 : 0 or 1
    - posting_time_variance    : variance in posting times (hrs)
    - avg_post_length          : average characters per post
    - hashtag_ratio            : hashtags per post
    - mention_ratio            : @mentions per post
    - reply_consistency        : how consistently they reply
    - content_diversity_score  : uniqueness of content
    """

    n_fake = n_samples // 2
    n_real = n_samples - n_fake

    # --- Real profiles ---
    real = pd.DataFrame({
        'posting_frequency':       np.random.normal(2.5, 1.2, n_real).clip(0.1, 20),
        'follower_following_ratio': np.random.lognormal(0.5, 0.8, n_real).clip(0.01, 50),
        'account_age_days':        np.random.normal(800, 400, n_real).clip(30, 3650),
        'avg_likes_per_post':      np.random.lognormal(3.5, 1.0, n_real).clip(0, 10000),
        'avg_comments_per_post':   np.random.lognormal(1.5, 1.0, n_real).clip(0, 1000),
        'bio_completeness':        np.random.beta(5, 2, n_real),
        'profile_pic_present':     np.random.binomial(1, 0.92, n_real),
        'url_in_bio':              np.random.binomial(1, 0.35, n_real),
        'verified':                np.random.binomial(1, 0.08, n_real),
        'posting_time_variance':   np.random.normal(6, 3, n_real).clip(0.5, 24),
        'avg_post_length':         np.random.normal(120, 50, n_real).clip(10, 500),
        'hashtag_ratio':           np.random.beta(2, 5, n_real) * 10,
        'mention_ratio':           np.random.beta(2, 8, n_real) * 5,
        'reply_consistency':       np.random.beta(4, 2, n_real),
        'content_diversity_score': np.random.beta(4, 2, n_real),
        'label': 0  # genuine
    })

    # --- Fake profiles ---
    fake = pd.DataFrame({
        'posting_frequency':       np.random.choice(
                                       np.concatenate([
                                           np.random.normal(20, 5, n_fake // 2),   # bots post constantly
                                           np.random.normal(0.2, 0.1, n_fake // 2) # dormant fakes
                                       ])
                                   ).clip(0, 50),
        'follower_following_ratio': np.random.lognormal(-0.5, 1.2, n_fake).clip(0.001, 5),
        'account_age_days':        np.random.normal(120, 80, n_fake).clip(1, 500),
        'avg_likes_per_post':      np.random.lognormal(1.0, 1.5, n_fake).clip(0, 500),
        'avg_comments_per_post':   np.random.lognormal(0.2, 0.8, n_fake).clip(0, 50),
        'bio_completeness':        np.random.beta(1.5, 5, n_fake),
        'profile_pic_present':     np.random.binomial(1, 0.55, n_fake),
        'url_in_bio':              np.random.binomial(1, 0.65, n_fake),
        'verified':                np.random.binomial(1, 0.005, n_fake),
        'posting_time_variance':   np.random.normal(1.5, 1.0, n_fake).clip(0, 5),  # robotic
        'avg_post_length':         np.random.normal(30, 20, n_fake).clip(1, 200),
        'hashtag_ratio':           np.random.beta(5, 2, n_fake) * 15,
        'mention_ratio':           np.random.beta(5, 2, n_fake) * 8,
        'reply_consistency':       np.random.beta(1.5, 5, n_fake),
        'content_diversity_score': np.random.beta(1.5, 5, n_fake),
        'label': 1  # fake
    })

    df = pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=42)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/profiles_dataset.csv", index=False)
    print(f"Dataset saved: {len(df)} records | Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")
    print(df.describe())