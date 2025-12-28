"""
Main training script for KneeXpert models.
"""
import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import ResNetModel, EfficientNetModel, VisionTransformerModel
from training.config import model_config, data_config, training_config, augmentation_config
from training.dataset import create_dataloaders
from training.train import train_model
from training.evaluate import evaluate_model
from preprocessing.utils import create_data_splits, load_presplit_data


def load_data_splits(data_dir: str):
    """Load data splits from directory or create them."""
    data_path = Path(data_dir)

    # If pre-split directories exist, use them directly
    if all((data_path / split).exists() for split in ["train", "val", "test"]):
        print("Detected pre-split data (train/val/test). Loading directly...")
        return load_presplit_data(data_dir)

    # Otherwise, fall back to auto-splitting
    splits_dir = data_path / "splits"
    if not splits_dir.exists() or not list(splits_dir.glob("*.txt")):
        print("Creating data splits...")
        splits = create_data_splits(
            data_dir,
            str(splits_dir),
            train_ratio=data_config.train_split,
            val_ratio=data_config.val_split,
            test_ratio=data_config.test_split,
            random_seed=training_config.seed,
        )
    else:
        print("Loading existing data splits...")
        splits = {}
        for split_name in ["train", "val", "test"]:
            split_file = splits_dir / f"{split_name}.txt"
            if split_file.exists():
                paths_labels = []
                with open(split_file, "r") as f:
                    for line in f:
                        path, label = line.strip().split("\t")
                        paths_labels.append((path, label))
                splits[split_name] = paths_labels

    # Convert to lists
    train_data = splits.get("train", [])
    val_data = splits.get("val", [])
    test_data = splits.get("test", [])

    # Create class mapping
    all_labels = set([label for _, label in train_data + val_data + test_data])
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_labels))}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    train_paths = [path for path, _ in train_data]
    train_labels = [class_to_idx[label] for _, label in train_data]
    val_paths = [path for path, _ in val_data]
    val_labels = [class_to_idx[label] for _, label in val_data]
    test_paths = [path for path, _ in test_data] if test_data else None
    test_labels = [class_to_idx[label] for _, label in test_data] if test_data else None

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, list(idx_to_class.values())


def main():
    parser = argparse.ArgumentParser(description="Train KneeXpert model")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_base"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed images",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    
    args = parser.parse_args()
    
    # Update config from args
    if args.epochs:
        model_config.num_epochs = args.epochs
    if args.batch_size:
        data_config.batch_size = args.batch_size
    if args.lr:
        model_config.learning_rate = args.lr
    
    # Set random seed
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    # Resolve runtime device (fall back to CPU if CUDA unavailable)
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("KneeXpert Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {model_config.num_epochs}")
    print(f"Batch size: {data_config.batch_size}")
    print(f"Learning rate: {model_config.learning_rate}")
    print(f"Device: {runtime_device}")
    print("=" * 60)
    
    # Load data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = load_data_splits(
        args.data_dir
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    if test_paths:
        print(f"  Test samples: {len(test_paths)}")
    print(f"  Classes: {class_names}")
    print()
    
    # Create data loaders
    if test_paths:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            class_names=class_names,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            image_size=(data_config.image_size, data_config.image_size),
            augmentation_config={
                "enable_augmentation": augmentation_config.enable_augmentation,
                "rotation_range": augmentation_config.rotation_range,
                "brightness_range": augmentation_config.brightness_range,
                "contrast_range": augmentation_config.contrast_range,
            },
        )
    else:
        train_loader, val_loader = create_dataloaders(
            train_paths, train_labels,
            val_paths, val_labels,
            class_names=class_names,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            image_size=(data_config.image_size, data_config.image_size),
            augmentation_config={
                "enable_augmentation": augmentation_config.enable_augmentation,
                "rotation_range": augmentation_config.rotation_range,
                "brightness_range": augmentation_config.brightness_range,
                "contrast_range": augmentation_config.contrast_range,
            },
        )
        test_loader = None
    
    # Create model
    if args.model == "resnet50":
        model = ResNetModel(
            num_classes=len(class_names),
            model_name="resnet50",
            pretrained=model_config.pretrained,
            dropout=model_config.dropout,
            freeze_backbone=model_config.freeze_backbone,
        )
    elif args.model == "efficientnet_b0":
        model = EfficientNetModel(
            num_classes=len(class_names),
            model_name="efficientnet_b0",
            pretrained=model_config.pretrained,
            dropout=model_config.dropout,
            freeze_backbone=model_config.freeze_backbone,
        )
    elif args.model == "vit_base":
        model = VisionTransformerModel(
            num_classes=len(class_names),
            model_name="vit_base_patch16_224",
            pretrained=model_config.pretrained,
            dropout=model_config.dropout,
            freeze_backbone=model_config.freeze_backbone,
        )
    
    # Train model
    save_dir = Path(model_config.checkpoint_dir) / args.model
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=model_config.num_epochs,
        device=runtime_device,
        save_dir=str(save_dir),
    )
    
    # Evaluate on test set if available
    if test_loader:
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)
        
        # Load best model
        best_model_path = save_dir / "best_model.pth"
        checkpoint = torch.load(best_model_path, map_location=runtime_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate
        metrics = evaluate_model(
            model,
            test_loader,
            device=runtime_device,
            class_names=class_names,
            save_dir=str(save_dir),
        )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print(f"Best model saved to: {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

