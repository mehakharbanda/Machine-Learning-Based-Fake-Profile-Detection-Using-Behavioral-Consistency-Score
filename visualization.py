"""
Visualization & Reporting Module
=================================
Generates:
  - Confusion matrix heatmap
  - ROC curve
  - Feature importance bar chart
  - BCS score distribution
  - Model comparison bar chart
All figures saved to /outputs/figures/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE  = {'genuine': '#2ecc71', 'fake': '#e74c3c', 'neutral': '#3498db'}
BG_COLOR = '#0f1117'
FG_COLOR = '#e8eaf0'

def _dark_style():
    plt.rcParams.update({
        'figure.facecolor':  BG_COLOR,
        'axes.facecolor':    '#1a1d27',
        'axes.edgecolor':    '#2d3047',
        'axes.labelcolor':   FG_COLOR,
        'xtick.color':       FG_COLOR,
        'ytick.color':       FG_COLOR,
        'text.color':        FG_COLOR,
        'grid.color':        '#2d3047',
        'grid.linestyle':    '--',
        'font.family':       'DejaVu Sans',
        'font.size':         11,
    })


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, model_name: str, save_path: str):
    _dark_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr  = np.array(cm)
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine', 'Fake'],
                yticklabels=['Genuine', 'Fake'],
                linewidths=1, linecolor='#2d3047',
                ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual',    fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# ── 2. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc_curves(models_dict: dict, X_test, y_test, save_path: str):
    """models_dict = {name: fitted_model}"""
    _dark_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for (name, model), color in zip(models_dict.items(), colors):
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name}  (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4, label='Random')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='white')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate');  ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# ── 3. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(importance_dict: dict, save_path: str, top_n=15):
    _dark_style()
    items  = list(importance_dict.items())[:top_n]
    names  = [i[0].replace('_', ' ').title() for i in items]
    vals   = [i[1] for i in items]
    colors = [PALETTE['genuine'] if 'bcs' in items[i][0].lower() or 'sub_' in items[i][0]
              else PALETTE['neutral'] for i in range(len(items))]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor='none', height=0.7)

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, color=FG_COLOR)

    legend_patches = [
        mpatches.Patch(color=PALETTE['genuine'], label='BCS / Sub-score features'),
        mpatches.Patch(color=PALETTE['neutral'], label='Raw profile features'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    ax.set_title('Feature Importance (Top 15)', fontsize=13)
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# ── 4. BCS Distribution ───────────────────────────────────────────────────────

def plot_bcs_distribution(df: pd.DataFrame, save_path: str):
    _dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # KDE plot
    ax = axes[0]
    for label, color, name in [(0, PALETTE['genuine'], 'Genuine'), (1, PALETTE['fake'], 'Fake')]:
        subset = df[df['label'] == label]['bcs_score']
        ax.hist(subset, bins=40, alpha=0.55, color=color, edgecolor='none', density=True, label=name)
        subset.plot.kde(ax=ax, color=color, lw=2)
    ax.axvline(45, color='yellow', lw=1.5, ls='--', alpha=0.7, label='Suspicious threshold (45)')
    ax.axvline(70, color='white',  lw=1.5, ls='--', alpha=0.7, label='Genuine threshold (70)')
    ax.set_title('BCS Score Distribution', fontsize=13)
    ax.set_xlabel('Behavioral Consistency Score (0–100)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    data_g = df[df['label'] == 0]['bcs_score']
    data_f = df[df['label'] == 1]['bcs_score']
    bp = ax2.boxplot([data_g, data_f], labels=['Genuine', 'Fake'],
                     patch_artist=True, notch=True,
                     medianprops=dict(color='white', lw=2))
    bp['boxes'][0].set_facecolor(PALETTE['genuine'] + '88')
    bp['boxes'][1].set_facecolor(PALETTE['fake']    + '88')
    ax2.set_title('BCS Score Box Plot', fontsize=13)
    ax2.set_ylabel('BCS Score')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Behavioral Consistency Score Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# ── 5. Model Comparison ───────────────────────────────────────────────────────

def plot_model_comparison(results: dict, save_path: str):
    _dark_style()
    metrics   = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    models    = list(results.keys())
    x         = np.arange(len(metrics))
    width     = 0.18
    colors    = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [results[model_name].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=color,
                      alpha=0.85, edgecolor='none')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, color=FG_COLOR)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title('Model Comparison — Performance Metrics', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(models_dict: dict, X_test, y_test, df_full: pd.DataFrame,
                        report: dict, out_dir: str = "../outputs/figures"):
    os.makedirs(out_dir, exist_ok=True)

    best_name  = report['best_model']
    best_model = models_dict[best_name]

    print("\n[Visualization] Generating plots ...")
    plot_confusion_matrix(
        report['all_results'][best_name]['confusion_matrix'],
        best_name,
        os.path.join(out_dir, '01_confusion_matrix.png')
    )
    plot_roc_curves(
        models_dict, X_test, y_test,
        os.path.join(out_dir, '02_roc_curves.png')
    )
    plot_feature_importance(
        report['feature_importance'],
        os.path.join(out_dir, '03_feature_importance.png')
    )
    plot_bcs_distribution(
        df_full,
        os.path.join(out_dir, '04_bcs_distribution.png')
    )
    plot_model_comparison(
        report['all_results'],
        os.path.join(out_dir, '05_model_comparison.png')
    )
    print("[Visualization] All plots generated.")


if __name__ == "__main__":
    print("Run model_training.py first to generate models, then visualize.")