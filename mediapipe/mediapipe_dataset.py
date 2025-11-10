import pickle
import numpy as np
import torch
import torch.utils.data as data_utl
from torch.utils.data import Dataset, DataLoader


class MediaPipeDataset(data_utl.Dataset):
    """
    Dataset for MediaPipe landmarks with added motion features (velocity + acceleration)
    and keypoint normalization.
    
    FIXES:
    - Reduced augmentation probabilities (was too aggressive)
    - Augment position first, then recompute velocity/acceleration
    - Removed time reversal (breaks temporal semantics)
    - Gentler transformation ranges
    """
    def __init__(self, features_file, split='train', augment=True, seq_len=64):
        with open(features_file, 'rb') as f:
            self.features_dict = pickle.load(f)
        
        self.data = []
        self.num_classes = 0
        for vid, info in self.features_dict.items():
            if split == 'train':
                if info['subset'] in ['train', 'val']:
                    self.data.append((vid, info['features'], info['label']))
            elif split == 'val':
                if info['subset'] == 'test':
                    self.data.append((vid, info['features'], info['label']))
            self.num_classes = max(self.num_classes, info['label'] + 1)
        
        self.augment = augment
        self.seq_len = seq_len
        print(f"{split.upper()} dataset: {len(self.data)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        vid, features, label = self.data[index]
        # features: numpy array shape (T, D) where D is typically 159 in your setup
        # 1) Normalize keypoints (per-sample, per-axis)
        features = self.normalize_keypoints(features)
        # 2) Add motion features (velocity and acceleration)
        features = self.add_motion_features(features)
        
        if self.augment:
            features = self.augment_features(features)
        
        # ensure fixed length at the end (pad/truncate)
        features = torch.from_numpy(self.pad_or_truncate(features, self.seq_len).copy()).float()
        label = torch.tensor(label, dtype=torch.long)
        return features, label, vid

    def normalize_keypoints(self, features, eps=1e-6):
        """Normalize keypoints per-frame by centering and scaling"""
        D = features.shape[1]
        if D % 3 != 0:
            # fallback: center by mean of all dims
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + eps
            return (features - mean) / std
        
        pts = features.reshape(features.shape[0], -1, 3)
        # center per-frame using mean of all keypoints
        mean = pts.mean(axis=1, keepdims=True)  # (T,1,3)
        std = pts.std(axis=1, keepdims=True) + eps
        pts = (pts - mean) / std
        return pts.reshape(features.shape)

    def add_motion_features(self, features):
        """Add velocity (delta) and acceleration (delta2) along time axis"""
        # features shape: (T, D)
        vel = np.diff(features, axis=0, prepend=features[0:1])
        acc = np.diff(vel, axis=0, prepend=vel[0:1])
        combined = np.concatenate([features, vel, acc], axis=1)
        return combined

    def augment_features(self, features):
        """
        Apply augmentation to POSITION features only, then recompute motion.
        
        MAJOR CHANGES:
        - Reduced all augmentation probabilities (0.3-0.4 instead of 0.6-0.9)
        - Gentler transformation ranges
        - Removed time reversal (breaks temporal semantics)
        - Apply augmentations to position, then recompute velocity/acceleration
        """
        # features shape: (T, D*3) where first D is position, next D is vel, last D is acc
        D = features.shape[1] // 3
        pos = features[:, :D].copy()  # Extract position features
        
        # 1. Random time shift - REDUCED from 0.9 to 0.3
        if np.random.random() < 0.3:
            shift = np.random.randint(-3, 4)  # Reduced from -5, 5
            pos = np.roll(pos, shift, axis=0)
        
        # 2. Gaussian noise - REDUCED from 0.9 to 0.4
        if np.random.random() < 0.4:
            noise = np.random.normal(0, 0.01, pos.shape)  # Reduced from 0.02
            pos = pos + noise
        
        # 3. Scaling - REDUCED from 0.9 to 0.4
        if np.random.random() < 0.4:
            scale = np.random.uniform(0.95, 1.05)  # Gentler: was 0.9, 1.1
            pos = pos * scale

        # 4. Hand flip - REDUCED from 0.6 to 0.3
        if np.random.random() < 0.3:
            # Assumes layout: left hand (0:63), right hand (63:126), face/pose (126:)
            if D >= 126 and D % 3 == 0:
                left_hand = pos[:, 0:63].copy()
                right_hand = pos[:, 63:126].copy()
                pos[:, 0:63] = right_hand
                pos[:, 63:126] = left_hand
                # Flip x-coordinates (assuming x, y, z ordering)
                pos[:, 0::3] = 1.0 - pos[:, 0::3]

        # 5. Rotation - REDUCED from 0.5 to 0.3
        if np.random.random() < 0.3:
            angle = np.random.uniform(-np.pi/24, np.pi/24)  # Gentler: was -pi/16, pi/16
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0, 0, 1]
            ])
            # Apply rotation to position channels
            if D % 3 == 0:
                for i in range(pos.shape[0]):
                    coords = pos[i].reshape(-1, 3)
                    rotated = coords @ rot_matrix.T
                    pos[i] = rotated.flatten()

        # 6. Temporal warp - REDUCED from 0.4 to 0.2
        if np.random.random() < 0.2:
            factor = np.random.uniform(0.95, 1.05)  # Gentler: was 0.9, 1.1
            new_len = max(1, int(pos.shape[0] * factor))
            idxs = np.linspace(0, pos.shape[0] - 1, new_len).astype(int)
            pos = pos[idxs]

        # 7. Time reverse - REMOVED (breaks temporal semantics for sign language)
        # Sign language has temporal dependencies that shouldn't be reversed

        # 8. Random crop - REDUCED from 0.3 to 0.2
        if np.random.random() < 0.2:
            crop = np.random.randint(1, 3)  # Reduced from 1, 4
            if pos.shape[0] > 2 * crop:
                pos = pos[crop:-crop]
        
        # RECOMPUTE motion features from augmented position
        vel = np.diff(pos, axis=0, prepend=pos[0:1])
        acc = np.diff(vel, axis=0, prepend=vel[0:1])
        features = np.concatenate([pos, vel, acc], axis=1)
        
        # Final: ensure fixed length (pad_or_truncate)
        features = self.pad_or_truncate(features, self.seq_len)
        return features

    @staticmethod
    def pad_or_truncate(features, length):
        """Pad or truncate sequence to fixed length"""
        if features.shape[0] > length:
            return features[:length]
        elif features.shape[0] < length:
            padding = np.tile(features[-1], (length - features.shape[0], 1))
            return np.vstack([features, padding])
        return features

def get_dataloaders(features_file, batch_size=32, num_workers=4):
    """Create train and validation dataloaders"""
    train_dataset = MediaPipeDataset(features_file, split='train', augment=True)
    val_dataset = MediaPipeDataset(features_file, split='val', augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, train_dataset.num_classes
