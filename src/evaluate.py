"""Evaluation module."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from typing import Dict, Tuple
import pandas as pd
from pathlib import Path

from src.utils import save_json


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (predictions, labels, probabilities)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probabilities)
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    except:
        roc_auc = 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ['Normal', 'Adenocarcinoma']
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        Classification report as string
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )


def save_evaluation_results(
    metrics: Dict,
    config: dict,
    classification_report: str
) -> None:
    """
    Save evaluation results.
    
    Args:
        metrics: Evaluation metrics
        config: Configuration dictionary
        classification_report: Classification report string
    """
    metrics_dir = Path(config['evaluation']['metrics_dir'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    save_json(metrics, metrics_dir / 'test_metrics.json')
    
    # Save classification report
    with open(metrics_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report)
    
    # Save metrics as CSV
    df = pd.DataFrame([metrics])
    df.to_csv(metrics_dir / 'test_metrics.csv', index=False)
    
    print(f"\n✓ Evaluation results saved to {metrics_dir}")
