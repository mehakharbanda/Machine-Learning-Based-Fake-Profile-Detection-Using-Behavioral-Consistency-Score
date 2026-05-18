"""
Behavioral Consistency Score (BCS) Module
==========================================
Computes a normalised score (0–100) that reflects how consistently
"human-like" a social media profile's behaviour is.

A high BCS  → likely genuine
A low  BCS  → likely fake / bot
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Individual sub-score helpers
# ─────────────────────────────────────────────────────────────────────────────

def _posting_regularity_score(posting_frequency: np.ndarray,
                               posting_time_variance: np.ndarray) -> np.ndarray:
    """
    Genuine users post irregularly but within a reasonable daily range.
    Bots post at extreme frequencies with very low variance (robotic clock).
    Score penalises both extremes.
    """
    freq_norm  = np.clip(posting_frequency, 0, 30) / 30.0   # 0..1
    # Sweet-spot for frequency is ~1-5 posts/day
    freq_score = 1 - np.abs(freq_norm - 0.15)               # peak around 4.5 posts/day

    # High variance in posting time is human-like
    var_norm   = np.clip(posting_time_variance, 0, 24) / 24.0
    var_score  = var_norm                                    # higher variance = more human

    return (0.5 * freq_score + 0.5 * var_score).clip(0, 1)


def _engagement_authenticity_score(avg_likes: np.ndarray,
                                    avg_comments: np.ndarray,
                                    follower_ratio: np.ndarray) -> np.ndarray:
    """
    Real users receive proportionate engagement relative to their follower ratio.
    Fake accounts often have near-zero engagement despite inflated metrics.
    """
    # log-scale normalisation to handle wide range
    likes_norm    = np.log1p(avg_likes)    / np.log1p(10000)
    comments_norm = np.log1p(avg_comments) / np.log1p(1000)
    ratio_norm    = np.log1p(follower_ratio.clip(0, 50)) / np.log1p(50)

    # Consistent engagement relative to followers
    engagement  = (likes_norm + comments_norm) / 2
    consistency = 1 - np.abs(engagement - ratio_norm * 0.8).clip(0, 1)

    return ((engagement * 0.6 + consistency * 0.4)).clip(0, 1)


def _profile_completeness_score(bio_completeness: np.ndarray,
                                  profile_pic: np.ndarray,
                                  account_age_days: np.ndarray,
                                  verified: np.ndarray) -> np.ndarray:
    """
    Genuine profiles tend to be complete and older.
    """
    age_norm = np.log1p(account_age_days.clip(0, 3650)) / np.log1p(3650)
    score    = (bio_completeness * 0.35 +
                profile_pic      * 0.25 +
                age_norm         * 0.30 +
                verified         * 0.10)
    return score.clip(0, 1)


def _content_quality_score(avg_post_length: np.ndarray,
                             content_diversity: np.ndarray,
                             reply_consistency: np.ndarray) -> np.ndarray:
    """
    Genuine users write longer, diverse content and reply consistently.
    """
    length_norm = np.clip(avg_post_length, 0, 500) / 500.0
    # Ideal post length: 80-200 chars
    length_score = 1 - np.abs(length_norm - 0.28)

    score = (length_score      * 0.30 +
             content_diversity * 0.40 +
             reply_consistency * 0.30)
    return score.clip(0, 1)


def _spam_signal_score(hashtag_ratio: np.ndarray,
                        mention_ratio: np.ndarray,
                        url_in_bio: np.ndarray) -> np.ndarray:
    """
    Very high hashtag/mention usage and suspicious bio links → bot-like.
    Returns an INVERTED score (1 = clean, 0 = spammy).
    """
    hash_norm    = np.clip(hashtag_ratio, 0, 15) / 15.0
    mention_norm = np.clip(mention_ratio, 0, 8)  / 8.0

    spam_signal = (hash_norm * 0.40 +
                   mention_norm * 0.40 +
                   url_in_bio   * 0.20)
    return (1 - spam_signal).clip(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Master BCS computation
# ─────────────────────────────────────────────────────────────────────────────

# Component weights — must sum to 1.0
WEIGHTS = {
    'posting_regularity':      0.20,
    'engagement_authenticity': 0.25,
    'profile_completeness':    0.20,
    'content_quality':         0.20,
    'spam_signal':             0.15,
}


def compute_bcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame with the behavioural feature columns and returns
    the same DataFrame with additional BCS columns.

    Returns
    -------
    df : pd.DataFrame
        Original columns + 5 sub-scores + 'bcs_raw' (0–1) + 'bcs_score' (0–100)
    """
    df = df.copy()

    df['sub_posting_regularity'] = _posting_regularity_score(
        df['posting_frequency'].values,
        df['posting_time_variance'].values
    )
    df['sub_engagement_authenticity'] = _engagement_authenticity_score(
        df['avg_likes_per_post'].values,
        df['avg_comments_per_post'].values,
        df['follower_following_ratio'].values
    )
    df['sub_profile_completeness'] = _profile_completeness_score(
        df['bio_completeness'].values,
        df['profile_pic_present'].values,
        df['account_age_days'].values,
        df['verified'].values
    )
    df['sub_content_quality'] = _content_quality_score(
        df['avg_post_length'].values,
        df['content_diversity_score'].values,
        df['reply_consistency'].values
    )
    df['sub_spam_signal'] = _spam_signal_score(
        df['hashtag_ratio'].values,
        df['mention_ratio'].values,
        df['url_in_bio'].values
    )

    df['bcs_raw'] = (
        df['sub_posting_regularity']      * WEIGHTS['posting_regularity']     +
        df['sub_engagement_authenticity'] * WEIGHTS['engagement_authenticity'] +
        df['sub_profile_completeness']    * WEIGHTS['profile_completeness']   +
        df['sub_content_quality']         * WEIGHTS['content_quality']        +
        df['sub_spam_signal']             * WEIGHTS['spam_signal']
    )

    df['bcs_score'] = (df['bcs_raw'] * 100).round(2)  # 0–100 for readability

    return df


def bcs_label(score: float) -> str:
    """Human-readable risk label based on BCS score."""
    if score >= 70:
        return "Genuine"
    elif score >= 45:
        return "Suspicious"
    else:
        return "Likely Fake"


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_dataset

    df = generate_dataset(500)
    df = compute_bcs(df)

    print("\n=== BCS Statistics ===")
    print(df[['bcs_score', 'label']].groupby('label').describe())
    print("\nSample rows:")
    print(df[['bcs_score', 'sub_posting_regularity', 'sub_engagement_authenticity',
              'sub_profile_completeness', 'sub_content_quality', 'sub_spam_signal', 'label']].head(10))