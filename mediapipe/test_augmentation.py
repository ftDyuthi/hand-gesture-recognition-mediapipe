"""
Test script to verify augmentation isn't destroying the training signal
"""
import numpy as np
import torch
from mediapipe_dataset import MediaPipeDataset
import matplotlib.pyplot as plt

def test_augmentation_quality(features_file):
    """
    Load dataset and check:
    1. How often augmentation is applied
    2. Visual comparison of original vs augmented
    3. Statistics of transformations
    """
    print("=" * 60)
    print("AUGMENTATION QUALITY TEST")
    print("=" * 60)
    
    # Load with and without augmentation
    dataset_no_aug = MediaPipeDataset(features_file, split='train', augment=False)
    dataset_aug = MediaPipeDataset(features_file, split='train', augment=True)
    
    print(f"\nDataset size: {len(dataset_no_aug)} samples")
    
    # Test multiple samples
    num_tests = 100
    aug_applied_count = 0
    
    print(f"\nTesting {num_tests} samples to measure augmentation frequency...")
    
    differences = []
    for i in range(num_tests):
        # Get same sample with and without augmentation
        idx = np.random.randint(0, len(dataset_no_aug))
        
        # Original
        orig_features, _, _ = dataset_no_aug[idx]
        
        # Augmented (try 10 times to see if augmentation is applied)
        aug_applied_this_sample = False
        for _ in range(10):
            aug_features, _, _ = dataset_aug[idx]
            diff = torch.abs(orig_features - aug_features).mean().item()
            
            if diff > 0.001:  # Threshold for detecting augmentation
                aug_applied_this_sample = True
                differences.append(diff)
                break
        
        if aug_applied_this_sample:
            aug_applied_count += 1
    
    aug_frequency = aug_applied_count / num_tests * 100
    print(f"\n✅ Augmentation applied to: {aug_frequency:.1f}% of samples")
    print(f"   Expected: ~5% (with 10 tries per sample)")
    
    if differences:
        print(f"\n📊 Augmentation Statistics (when applied):")
        print(f"   Mean difference: {np.mean(differences):.6f}")
        print(f"   Std difference: {np.std(differences):.6f}")
        print(f"   Max difference: {np.max(differences):.6f}")
        print(f"   Min difference: {np.min(differences):.6f}")
    
    # Visual test - compare one sample
    print(f"\n📈 Generating visual comparison...")
    idx = 0
    orig_features, label, vid = dataset_no_aug[idx]
    aug_features, _, _ = dataset_aug[idx]
    
    # Try multiple times to get augmented version
    for attempt in range(50):
        aug_features, _, _ = dataset_aug[idx]
        if torch.abs(orig_features - aug_features).mean() > 0.001:
            break
    
    orig_np = orig_features.numpy()
    aug_np = aug_features.numpy()
    
    # Plot first few features over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Position features (first D/3)
    D = orig_np.shape[1] // 3
    
    for i, ax in enumerate(axes.flat):
        if i < 4:
            feat_idx = i * (D // 4)
            ax.plot(orig_np[:, feat_idx], label='Original', linewidth=2, alpha=0.7)
            ax.plot(aug_np[:, feat_idx], label='Augmented', linewidth=2, alpha=0.7, linestyle='--')
            ax.set_title(f'Feature {feat_idx}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('augmentation_comparison.png', dpi=150)
    print(f"   Saved visualization to: augmentation_comparison.png")
    
    # Check training/validation split
    print(f"\n📊 Dataset Split:")
    print(f"   Training samples: {len(dataset_no_aug)}")
    val_dataset = MediaPipeDataset(features_file, split='val', augment=False)
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Number of classes: {dataset_no_aug.num_classes}")
    
    # Check feature dimensions
    sample_features, _, _ = dataset_no_aug[0]
    print(f"\n📊 Feature Dimensions:")
    print(f"   Sequence length: {sample_features.shape[0]}")
    print(f"   Feature dimension: {sample_features.shape[1]}")
    print(f"   Expected: 64 x 477 (position + velocity + acceleration)")
    
    # Check for NaN or Inf
    has_nan = torch.isnan(sample_features).any()
    has_inf = torch.isinf(sample_features).any()
    print(f"\n✅ Data Quality:")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Value range: [{sample_features.min():.3f}, {sample_features.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if aug_frequency > 10:
        print("⚠️  Augmentation frequency is HIGH. Consider reducing further.")
    elif aug_frequency < 3:
        print("✅ Augmentation frequency is good (minimal).")
    
    if sample_features.max() > 10 or sample_features.min() < -10:
        print("⚠️  Feature values are large. Check normalization.")
    else:
        print("✅ Feature values are in reasonable range.")
    
    if has_nan or has_inf:
        print("❌ Data has NaN/Inf values! Fix preprocessing!")
    else:
        print("✅ No NaN/Inf values detected.")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, 
                       default="../../../mediapipe_features/mediapipe_features_top100.pkl")
    args = parser.parse_args()
    
    test_augmentation_quality(args.features_file)
