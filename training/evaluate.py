"""
Model evaluation and metrics calculation.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: List[str] = None,
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC calculation)
        class_names: Names of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    if class_names:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f"{class_name}_precision"] = precision_per_class[i]
            metrics[f"{class_name}_recall"] = recall_per_class[i]
            metrics[f"{class_name}_f1"] = f1_per_class[i]
    
    # AUC-ROC (if probabilities provided)
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                # Binary classification
                metrics["auc"] = roc_auc_score(y_true, y_proba)
            else:
                # Multi-class: use one-vs-rest
                metrics["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            metrics["auc"] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    class_names: List[str] = None,
    save_dir: str = None,
) -> Dict:
    """
    Evaluate model on test set and generate comprehensive reports.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        class_names: Names of classes
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation results
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probas = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probas, class_names)
    
    # Generate classification report
    if class_names:
        report = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
        metrics["classification_report"] = report
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_json = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        with open(save_dir / "evaluation_metrics.json", "w") as f:
            import json
            json.dump(metrics_json, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names or [f"Class {i}" for i in range(len(metrics["confusion_matrix"]))],
            save_path=save_dir / "confusion_matrix.png",
        )
        
        # Plot ROC curve (if binary classification)
        if all_probas.shape[1] == 2:
            plot_roc_curve(
                all_labels, all_probas[:, 1],
                save_path=save_dir / "roc_curve.png",
            )
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if metrics.get("auc") is not None:
        print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print("=" * 50)
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = None,
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()

