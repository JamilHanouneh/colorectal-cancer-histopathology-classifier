"""Visualization module."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path
from typing import Dict
import pandas as pd


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(history: Dict, save_path: str) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        annot_kws={'fontsize': 14}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix plot saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ ROC curve plot saved to {save_path}")


def create_metrics_table(metrics: Dict, save_path: str) -> None:
    """
    Create and save metrics table.
    
    Args:
        metrics: Evaluation metrics dictionary
        save_path: Path to save table
    """
    # Select main metrics
    main_metrics = {
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall/Sensitivity': f"{metrics['recall']:.4f}",
        'Specificity': f"{metrics['specificity']:.4f}",
        'F1 Score': f"{metrics['f1_score']:.4f}",
        'ROC AUC': f"{metrics['roc_auc']:.4f}"
    }
    
    df = pd.DataFrame(list(main_metrics.items()), columns=['Metric', 'Value'])
    df.to_csv(save_path, index=False)
    
    print(f"✓ Metrics table saved to {save_path}")


def generate_all_visualizations(
    history: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict,
    config: dict
) -> None:
    """
    Generate all visualizations.
    
    Args:
        history: Training history
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        metrics: Evaluation metrics
        config: Configuration dictionary
    """
    figures_dir = Path(config['evaluation']['figures_dir'])
    tables_dir = Path(config['evaluation']['tables_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Normal', 'Adenocarcinoma']
    
    # Training history
    plot_training_history(history, figures_dir / 'training_history.png')
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names,
                         figures_dir / 'confusion_matrix.png')
    
    # ROC curve
    plot_roc_curve(y_true, y_prob, figures_dir / 'roc_curve.png')
    
    # Metrics table
    create_metrics_table(metrics, tables_dir / 'performance_metrics.csv')
