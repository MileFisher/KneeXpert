"""
Training script for knee disease classification models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, Optional
from .config import model_config, training_config, data_config


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = None,
    device: str = None,
    save_dir: str = None,
) -> Dict:
    """
    Train a model with validation monitoring.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on ("cuda" or "cpu")
        save_dir: Directory to save checkpoints
        
    Returns:
        Training history dictionary
    """
    if num_epochs is None:
        num_epochs = model_config.num_epochs
    if device is None:
        device = training_config.device
    if save_dir is None:
        save_dir = model_config.checkpoint_dir
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Mixed precision training
    scaler = GradScaler() if training_config.use_mixed_precision else None
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    
    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training on {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                if training_config.gradient_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if training_config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * train_correct / train_total:.2f}%"
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device, scaler)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch + 1
            
            if model_config.save_best_only:
                checkpoint_path = Path(save_dir) / "best_model.pth"
            else:
                checkpoint_path = Path(save_dir) / f"epoch_{epoch+1}_best.pth"
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "history": history,
            }, checkpoint_path)
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= model_config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    # Save final model and history
    final_checkpoint = Path(save_dir) / "final_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, final_checkpoint)
    
    history_path = Path(save_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    return history


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> tuple:
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        scaler: Optional gradient scaler for mixed precision
        
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

