# Work : Copyright Submission
# Title : Machine Learning–Based Fake Profile Detection Using Behavioral Consistency Score
# Authors : Shriya Seth, Mehak Kharbanda, Sargam Narang
# Roll Numbers : 2201992353, 2210990574, 2210992259
# Current Status : Unpublished

A full-stack machine learning system that detects fake social media profiles by computing a **Behavioral Consistency Score (BCS)** from user activity patterns.

---

## Project Structure

```
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── src/
│   ├── api.py               # Flask REST API
│   ├── bcsModule.py         # Behavioral Consistency Score computation
│   ├── data_generator.py    # Synthetic dataset generation
│   ├── model_training.py    # ML model training & evaluation
│   ├── preprocessing.py     # Data preprocessing pipeline
│   └── visualization.py     # Charts & evaluation plots
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx          # Main React dashboard
        └── main.jsx         # React entry point
```

---

## Tech Stack

**Backend:** Python, Flask, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

**Frontend:** React, Vite

**ML Models:** Random Forest, Gradient Boosting, Logistic Regression, SVM

---

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## Running the Project

Open **two terminals** and run both at the same time.

**Terminal 1 — Start the Backend API:**
```bash
python main.py api
```
Backend runs at: `http://localhost:5000`

**Terminal 2 — Start the Frontend:**
```bash
cd frontend
npm run dev
```
Frontend runs at: `http://localhost:5173`

Open `http://localhost:5173` in your browser to use the dashboard.

---

## Other Run Modes

```bash
python main.py demo        # Full demo: train + evaluate + sample predictions
python main.py train       # Train all models and save
python main.py evaluate    # Generate evaluation plots
python main.py predict     # Interactive CLI predictor
```

---

## How It Works

The system computes a **Behavioral Consistency Score (BCS)** for each profile using 5 weighted sub-scores:

| Sub-Score | Weight | Description |
|---|---|---|
| Posting Regularity | 20% | Posting frequency and time variance |
| Engagement Authenticity | 25% | Likes/comments vs follower ratio |
| Profile Completeness | 20% | Bio, profile pic, account age, verified |
| Content Quality | 20% | Post length, diversity, reply consistency |
| Anti-Spam Signal | 15% | Hashtag/mention ratio, suspicious links |

**BCS Thresholds:**
- ≥ 70 → Genuine
- 45–70 → Suspicious
- < 45 → Likely Fake

---

## Features (15 Behavioral Attributes)

`posting_frequency`, `follower_following_ratio`, `account_age_days`, `avg_likes_per_post`, `avg_comments_per_post`, `bio_completeness`, `profile_pic_present`, `url_in_bio`, `verified`, `posting_time_variance`, `avg_post_length`, `hashtag_ratio`, `mention_ratio`, `reply_consistency`, `content_diversity_score`

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/model/info` | Model metrics and feature importance |
| POST | `/api/predict` | Predict a single profile |
| POST | `/api/predict/batch` | Predict multiple profiles |

**Example prediction request:**
```json
POST http://localhost:5000/api/predict
{
  "posting_frequency": 2.5,
  "follower_following_ratio": 2.0,
  "account_age_days": 400,
  "bio_completeness": 0.8,
  "profile_pic_present": 1
}
```

**Example response:**
```json
{
  "prediction": "Genuine",
  "is_fake": false,
  "probability": 0.12,
  "bcs_score": 74.3,
  "bcs_label": "Genuine",
  "risk_level": "Low"
}
```

---

## Evaluation Plots

After running `python main.py evaluate`, plots are saved to `outputs/figures/`:

- `01_confusion_matrix.png`
- `02_roc_curves.png`
- `03_feature_importance.png`
- `04_bcs_distribution.png`
- `05_model_comparison.png`

---

## Dashboard

The React frontend provides an interactive dashboard with:
- **INPUT tab** — Adjust 15 behavioral sliders and toggles
- **RESULT tab** — Live BCS gauge, radar chart, and sub-score breakdown
- **ABOUT tab** — Project description and methodology

---

## Abstract

This project develops a machine learning-based system to detect fake social media profiles using a Behavioral Consistency Score (BCS). Unlike rule-based methods, the system analyses user behaviour patterns including posting frequency, engagement rates, and content diversity to identify suspicious accounts. A Random Forest classifier trained on these features achieves high detection accuracy with explainable sub-scores for each prediction.

---

## Author

**Final Semester Project**
Department of Computer Science
=======
