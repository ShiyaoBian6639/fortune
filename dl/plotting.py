"""
Visualization functions for training results and evaluation.
"""

import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

from .config import get_class_names, NUM_CLASSES


def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    class_names = get_class_names()
    n_classes = cm.shape[0]

    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig_size = max(10, n_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})

    thresh = cm_normalized.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:
                ax.text(j + 0.5, i + 0.5, f'{cm[i, j]}',
                       ha="center", va="center", fontsize=7,
                       color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def plot_class_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot class distribution comparison."""
    class_names = get_class_names()
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, n_classes))

    # Actual
    ax1 = axes[0]
    unique, counts = np.unique(y_true, return_counts=True)
    count_dict = dict(zip(unique, counts))
    bars1 = ax1.bar(range(n_classes), [count_dict.get(i, 0) for i in range(n_classes)],
                    color=colors, edgecolor='black')
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Count')
    ax1.set_title('Actual Class Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Predicted
    ax2 = axes[1]
    unique, counts = np.unique(y_pred, return_counts=True)
    count_dict = dict(zip(unique, counts))
    bars2 = ax2.bar(range(n_classes), [count_dict.get(i, 0) for i in range(n_classes)],
                    color=colors, edgecolor='black')
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Count')
    ax2.set_title('Predicted Class Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Class distribution plot saved to: {save_path}")


def plot_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot per-class precision, recall, and F1 scores."""
    class_names = get_class_names()
    n_classes = len(class_names)

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Pad if needed
    while len(precision) < n_classes:
        precision = np.append(precision, 0)
        recall = np.append(recall, 0)
        f1 = np.append(f1, 0)

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    ax.bar(x, recall, width, label='Recall', color='darkorange')
    ax.bar(x + width, f1, width, label='F1 Score', color='green')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to: {save_path}")


def plot_metrics_summary(metrics: Dict[str, float], save_path: str):
    """Plot bar chart of evaluation metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metric_names)))

    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='navy', linewidth=1.5)

    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=1/NUM_CLASSES, color='red', linestyle='--',
               label=f'Random Baseline ({1/NUM_CLASSES:.3f})', alpha=0.7)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics summary plot saved to: {save_path}")


def plot_combined_summary(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: str
):
    """Generate combined summary figure."""
    class_names = get_class_names()

    fig = plt.figure(figsize=(20, 12))

    # Training history
    ax1 = fig.add_subplot(2, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Confusion matrix
    ax3 = fig.add_subplot(2, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=ax3,
                xticklabels=class_names, yticklabels=class_names)
    ax3.set_title('Confusion Matrix', fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    plt.setp(ax3.get_yticklabels(), fontsize=6)

    # Class distribution
    ax4 = fig.add_subplot(2, 3, 4)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.35
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    true_counts = [counts_true[list(unique_true).index(i)] if i in unique_true else 0 for i in range(n_classes)]
    pred_counts = [counts_pred[list(unique_pred).index(i)] if i in unique_pred else 0 for i in range(n_classes)]
    ax4.bar(x - width/2, true_counts, width, label='Actual', alpha=0.7)
    ax4.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=45, ha='right', fontsize=6)
    ax4.set_title('Class Distribution', fontweight='bold')
    ax4.legend()

    # Per-class accuracy
    ax5 = fig.add_subplot(2, 3, 5)
    per_class_acc = []
    for i in range(n_classes):
        if i < cm.shape[0]:
            total = cm[i].sum()
            correct = cm[i, i] if i < cm.shape[1] else 0
            per_class_acc.append(correct / total if total > 0 else 0)
        else:
            per_class_acc.append(0)
    colors = plt.cm.RdYlGn(np.array(per_class_acc))
    ax5.bar(x, per_class_acc, color=colors, edgecolor='black')
    ax5.set_xticks(x)
    ax5.set_xticklabels(class_names, rotation=45, ha='right', fontsize=6)
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Per-Class Accuracy', fontweight='bold')
    ax5.set_ylim([0, 1.0])
    ax5.grid(True, alpha=0.3, axis='y')

    # Overall metrics
    ax6 = fig.add_subplot(2, 3, 6)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    bars = ax6.bar(metrics.keys(), metrics.values(), color='steelblue', edgecolor='navy')
    for bar, val in zip(bars, metrics.values()):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax6.axhline(y=1/n_classes, color='red', linestyle='--', label='Random', alpha=0.7)
    ax6.set_ylabel('Score')
    ax6.set_title('Overall Metrics (Weighted)', fontweight='bold')
    ax6.set_ylim([0, 1.0])
    ax6.legend()
    ax6.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'Stock Price Change Prediction ({NUM_CLASSES} Classes) - Summary',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined summary plot saved to: {save_path}")


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, save_path: str):
    """
    Plot One-vs-Rest ROC curves for every class plus macro- and micro-averages.

    Args:
        y_true:  (N,) integer class labels
        y_probs: (N, C) predicted probabilities (post-softmax / calibrated)
    """
    class_names = get_class_names()
    n_classes   = len(class_names)

    y_bin = label_binarize(y_true, classes=range(n_classes))   # (N, C)
    # If only one class present in y_true, label_binarize returns (N,1) — guard
    if y_bin.shape[1] != n_classes:
        y_bin = np.eye(n_classes)[y_true.astype(int)]

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class OvR curves
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:          # class absent from test split
            fpr[i] = np.array([0., 1.])
            tpr[i] = np.array([0., 0.])
            roc_auc[i] = 0.0
        else:
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average: flatten all classes
    fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Macro-average: interpolate per-class curves to a shared FPR axis
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro']    = all_fpr
    tpr['macro']    = mean_tpr
    roc_auc['macro'] = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (color, name) in enumerate(zip(colors, class_names)):
        ax.plot(fpr[i], tpr[i], color=color, lw=1.2, alpha=0.65,
                label=f'{name}  AUC={roc_auc[i]:.3f}')

    ax.plot(fpr['micro'], tpr['micro'], 'k--', lw=2,
            label=f'Micro-avg  AUC={roc_auc["micro"]:.3f}')
    ax.plot(fpr['macro'], tpr['macro'], 'k-',  lw=2,
            label=f'Macro-avg  AUC={roc_auc["macro"]:.3f}')
    ax.plot([0, 1], [0, 1], color='grey', linestyle=':', lw=1, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('ROC Curves — One-vs-Rest (7-class)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to: {save_path}")


def plot_precision_recall_curves(y_true: np.ndarray, y_probs: np.ndarray, save_path: str):
    """
    Plot One-vs-Rest Precision-Recall curves for every class plus the micro-average.

    Args:
        y_true:  (N,) integer class labels
        y_probs: (N, C) predicted probabilities (post-softmax / calibrated)
    """
    class_names = get_class_names()
    n_classes   = len(class_names)

    y_bin = label_binarize(y_true, classes=range(n_classes))
    if y_bin.shape[1] != n_classes:
        y_bin = np.eye(n_classes)[y_true.astype(int)]

    precision_c, recall_c, avg_prec = {}, {}, {}

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            precision_c[i] = np.array([1., 0.])
            recall_c[i]    = np.array([0., 0.])
            avg_prec[i]    = 0.0
        else:
            precision_c[i], recall_c[i], _ = precision_recall_curve(
                y_bin[:, i], y_probs[:, i]
            )
            avg_prec[i] = average_precision_score(y_bin[:, i], y_probs[:, i])

    # Micro-average PR curve
    precision_c['micro'], recall_c['micro'], _ = precision_recall_curve(
        y_bin.ravel(), y_probs.ravel()
    )
    avg_prec['micro'] = average_precision_score(y_bin, y_probs, average='micro')

    # Macro-average AP (scalar only — no single macro PR curve is well-defined)
    avg_prec['macro'] = average_precision_score(y_bin, y_probs, average='macro')

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (color, name) in enumerate(zip(colors, class_names)):
        ax.plot(recall_c[i], precision_c[i], color=color, lw=1.2, alpha=0.65,
                label=f'{name}  AP={avg_prec[i]:.3f}')

    ax.plot(recall_c['micro'], precision_c['micro'], 'k-', lw=2,
            label=f'Micro-avg  AP={avg_prec["micro"]:.3f}')

    # Baseline: overall positive rate across all OvR problems
    baseline = float(y_bin.mean())
    ax.axhline(y=baseline, color='grey', linestyle=':', lw=1,
               label=f'No-skill baseline ({baseline:.3f})')

    # Macro AP annotation
    ax.text(0.02, 0.04, f'Macro-avg AP = {avg_prec["macro"]:.3f}',
            transform=ax.transAxes, fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall',    fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — One-vs-Rest (7-class)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curves saved to: {save_path}")


def plot_tft_variable_importance(
    weights:    np.ndarray,
    feat_names: List[str],
    save_path:  str,
    title:      str,
    top_n:      int = 30,
):
    """
    Horizontal bar chart of VSN variable importance weights.

    Args:
        weights:    (num_features,) mean VSN softmax weights across all samples/timesteps
        feat_names: feature name strings matching weights length
        save_path:  output PNG path
        title:      subplot title
        top_n:      show at most this many features (by descending weight)
    """
    n = min(top_n, len(weights))
    order = np.argsort(weights)[::-1][:n]
    w     = weights[order]
    names = [feat_names[i] for i in order]

    # Reverse so highest bar is at the top
    w     = w[::-1]
    names = names[::-1]

    colors = plt.cm.RdYlGn(w / (w.max() + 1e-10))

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    bars = ax.barh(range(n), w, color=colors, edgecolor='grey', linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Mean VSN Weight', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # Value labels
    for bar, val in zip(bars, w):
        ax.text(bar.get_width() + w.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"TFT variable importance plot saved to: {save_path}")


def plot_tft_attention_heatmap(
    attn_weights: np.ndarray,
    save_path:    str,
    seq_len:      int = 30,
    dec_len:      int = 5,
):
    """
    Heatmap of averaged InterpretableMultiHeadAttention weights (35×35).

    Args:
        attn_weights: (seq_len+dec_len, seq_len+dec_len) averaged attention matrix
        save_path:    output PNG path
        seq_len:      encoder length (30 past timesteps)
        dec_len:      decoder length (5 future positions)
    """
    total = seq_len + dec_len

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn_weights, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Encoder/decoder boundary
    ax.axvline(x=seq_len - 0.5, color='red',    linestyle='--', linewidth=1.5, label='Enc/Dec boundary')
    ax.axhline(y=seq_len - 0.5, color='red',    linestyle='--', linewidth=1.5)

    # Prediction horizon positions (day3/4/5 → decoder positions 2,3,4)
    for pos in [seq_len + 2, seq_len + 3, seq_len + 4]:
        if pos < total:
            ax.axvline(x=pos - 0.5, color='yellow', linestyle=':', linewidth=1.0)
            ax.axhline(y=pos - 0.5, color='yellow', linestyle=':', linewidth=1.0)

    # Axis labels: t1–t30 for encoder, d1–d5 for decoder
    enc_labels = [f't{i+1}' for i in range(seq_len)]
    dec_labels = [f'd{i+1}' for i in range(dec_len)]
    all_labels = enc_labels + dec_labels

    tick_step = max(1, total // 20)
    ticks = list(range(0, total, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([all_labels[t] for t in ticks], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([all_labels[t] for t in ticks], fontsize=7)

    ax.set_xlabel('Key Position', fontsize=11)
    ax.set_ylabel('Query Position', fontsize=11)
    ax.set_title('TFT Interpretable Multi-Head Attention (averaged over test set)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"TFT attention heatmap saved to: {save_path}")


def plot_all_results(
    history:    Dict[str, List[float]],
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_probs:    np.ndarray,
    save_dir:   str,
    tft_interp: dict = None,
    model_type: str  = 'transformer',
):
    """Generate all visualization plots."""
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "-" * 40)
    print("Generating Visualizations")
    print("-" * 40)

    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    plot_confusion_matrix(y_true, y_pred, os.path.join(save_dir, 'confusion_matrix.png'))
    plot_class_distribution(y_true, y_pred, os.path.join(save_dir, 'class_distribution.png'))
    plot_per_class_metrics(y_true, y_pred, os.path.join(save_dir, 'per_class_metrics.png'))

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision\n(weighted)': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall\n(weighted)': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score\n(weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    plot_metrics_summary(metrics, os.path.join(save_dir, 'metrics_summary.png'))

    # ROC and Precision-Recall curves require class probabilities
    if y_probs is not None and y_probs.ndim == 2 and y_probs.shape[1] == NUM_CLASSES:
        plot_roc_curves(y_true, y_probs, os.path.join(save_dir, 'roc_curves.png'))
        plot_precision_recall_curves(y_true, y_probs, os.path.join(save_dir, 'precision_recall_curves.png'))

    plot_combined_summary(history, y_true, y_pred, y_probs, os.path.join(save_dir, 'combined_summary.png'))

    # TFT-specific interpretability plots
    if tft_interp is not None:
        from .config import OBSERVED_PAST_FEATURE_COLUMNS, KNOWN_FUTURE_FEATURE_COLUMNS
        plot_tft_variable_importance(
            weights    = tft_interp['enc_vsn'],
            feat_names = OBSERVED_PAST_FEATURE_COLUMNS,
            save_path  = os.path.join(save_dir, 'tft_enc_variable_importance.png'),
            title      = 'TFT Encoder Variable Importance (Observed Past, top 30)',
            top_n      = 30,
        )
        plot_tft_variable_importance(
            weights    = tft_interp['dec_vsn'],
            feat_names = KNOWN_FUTURE_FEATURE_COLUMNS,
            save_path  = os.path.join(save_dir, 'tft_dec_variable_importance.png'),
            title      = 'TFT Decoder Variable Importance (Known Future, all 27)',
            top_n      = len(KNOWN_FUTURE_FEATURE_COLUMNS),
        )
        plot_tft_attention_heatmap(
            attn_weights = tft_interp['attn'],
            save_path    = os.path.join(save_dir, 'tft_attention_heatmap.png'),
        )
