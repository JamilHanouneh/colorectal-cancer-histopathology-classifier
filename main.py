"""Main pipeline script for colorectal cancer classification."""
import torch 
from pathlib import Path

from src.config import Config
from src.utils import set_seed, setup_logging, create_directories, load_json
from src.data_loader import get_data_loaders
from src.model import create_model
from src.train import train_model
from src.evaluate import (
    evaluate_model, calculate_metrics,
    generate_classification_report, save_evaluation_results
)
from src.visualization import generate_all_visualizations


def main():
    """Main execution function."""
    print("=" * 80)
    print("Colorectal Cancer Classification Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = Config('config.yaml')
    
    # Set random seed
    set_seed(config.seed)
    print(f"\n✓ Random seed set to {config.seed}")
    
    # Create directories FIRST (before logging)
    create_directories(config.config)
    print("✓ Directories created")
    
    # Setup logging AFTER directories are created
    logger = setup_logging('outputs/logs/pipeline.log')
    logger.info("Starting pipeline execution")
    
    # Check for dataset - UPDATED FOR NCT-CRC-HE-100K
    raw_dir = Path(config['data']['raw_dir'])
    dataset_dir = raw_dir / 'NCT-CRC-HE-100K'
    norm_dir = dataset_dir / 'NORM'
    tum_dir = dataset_dir / 'TUM'
    
    if not dataset_dir.exists():
        print("\n⚠️  Dataset not found!")
        print("Please download the NCT-CRC-HE-100K dataset from:")
        print("https://zenodo.org/records/1214456")
        print(f"And extract it to: {raw_dir}")
        print("\nExpected structure:")
        print(f"  {dataset_dir}/")
        print(f"    ├── NORM/  (normal colon mucosa images)")
        print(f"    └── TUM/   (adenocarcinoma images)")
        return
    
    # Check if the required class folders exist
    if not norm_dir.exists() or not tum_dir.exists():
        print("\n⚠️  Required class folders not found!")
        print(f"Looking for:")
        print(f"  NORM folder: {norm_dir} - {'✓ Found' if norm_dir.exists() else '✗ Missing'}")
        print(f"  TUM folder: {tum_dir} - {'✓ Found' if tum_dir.exists() else '✗ Missing'}")
        print("\nCurrent structure:")
        print(f"  {dataset_dir}/ contains:")
        for item in dataset_dir.iterdir():
            print(f"    - {item.name}")
        return
    
    # Count images
    norm_images = list(norm_dir.glob('*.tif'))
    tum_images = list(tum_dir.glob('*.tif'))
    
    print(f"\n✓ Dataset found at {dataset_dir}")
    print(f"  - NORM (normal): {len(norm_images):,} images")
    print(f"  - TUM (cancer): {len(tum_images):,} images")
    print(f"  - Total: {len(norm_images) + len(tum_images):,} images")
    
    # Load data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    train_loader, val_loader, test_loader = get_data_loaders(
        config.config, config.seed
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)
    model = create_model(config.config)
    model = model.to(config.device)
    print(f"✓ Model architecture: {config['model']['architecture']}")
    print(f"✓ Device: {config.device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n" + "=" * 80)
    print("Training Model")
    print("=" * 80)
    model, history = train_model(
        model, train_loader, val_loader, config.config, config.device
    )
    print("\n✓ Training completed")
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("Evaluating Model")
    print("=" * 80)
    y_pred, y_true, y_prob = evaluate_model(model, test_loader, config.device)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    
    # Generate classification report
    report = generate_classification_report(y_true, y_pred)
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save evaluation results
    save_evaluation_results(metrics, config.config, report)
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    generate_all_visualizations(
        history, y_true, y_pred, y_prob, metrics, config.config
    )
    
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\nResults saved to: outputs/")
    print("  - Trained model: outputs/models/best_model.pth")
    print("  - Metrics: outputs/metrics/")
    print("  - Figures: outputs/figures/")
    print("  - Tables: outputs/tables/")
    print("  - Logs: outputs/logs/pipeline.log")


if __name__ == '__main__':
    main()
