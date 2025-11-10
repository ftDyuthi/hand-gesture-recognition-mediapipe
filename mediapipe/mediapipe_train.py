import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
import argparse
from mediapipe_dataset import get_dataloaders

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

class TemporalBlock(nn.Module):
    """1D temporal convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SignLanguageModel(nn.Module):
    """
    Larger Hybrid Model: Temporal CNN + Transformer + GRU
    
    Modified:
    - Hidden size 1536
    - Transformer encoder layers increased to 5
    """
    def __init__(self, input_size=159*3, hidden_size=2048, num_classes=25, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.temp_conv1 = TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=1, dropout=dropout)
        self.temp_conv2 = TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=2, dropout=dropout)
        self.temp_conv3 = TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=4, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        # Increase layers from 3 to 5
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        self.gru = nn.GRU(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x_conv = x.transpose(1, 2)
        x_conv = self.temp_conv1(x_conv)
        x_conv = self.temp_conv2(x_conv)
        x_conv = self.temp_conv3(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_trans = self.transformer(x_conv)
        x_gru, _ = self.gru(x_trans)
        x_attn, _ = self.attention(x_gru, x_gru, x_gru)
        x_attn = self.norm(x_attn + x_gru)
        avg_pool = torch.mean(x_attn, dim=1)
        max_pool, _ = torch.max(x_attn, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        output = self.classifier(pooled)
        return output


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for features, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(features)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / len(dataloader), 100 * total_correct / total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for features, labels, _ in tqdm(dataloader, desc="Validation", leave=False):
            features, labels = features.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    return total_loss / len(dataloader), 100 * total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str,
                        default="../../../mediapipe_features/mediapipe_features_top25.pkl")
    parser.add_argument("--save_dir", type=str,
                        default="../../../checkpoints_mediapipe_top25_v4/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower LR
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable minimal augmentation (5%)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data with augmentation option
    train_loader, val_loader, num_classes = get_dataloaders(args.features_file, args.batch_size, num_workers=args.num_workers)
    # Enable minimal augmentation at dataset level via a flag in mediapipe_dataset.py if needed
    
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[2]
    print(f"Input dimension: {input_size}")
    print(f"Number of classes: {num_classes}")

    model = SignLanguageModel(
        input_size=input_size,
        hidden_size=2048,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0
    epochs_no_improve = 0
    patience = 60

    print("=" * 70)
    print("ENHANCED MODEL TRAINING:")
    print(f"  ✓ Hidden size: 2046")
    print(f"  ✓ Transformer layers: 5")
    print(f"  ✓ Minimal augmentation enabled: {args.augment}")
    print(f"  ✓ Lower learning rate: {args.lr}")
    print(f"  ✓ Training samples: {len(train_loader.dataset)}")
    print(f"  ✓ Validation samples: {len(val_loader.dataset)}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs} [LR: {current_lr:.6f}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_path = os.path.join(args.save_dir, f"best_{num_classes}cls_val{val_acc:.2f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'num_classes': num_classes
            }, save_path)
            print(f"✅ Saved best model: {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️  Early stopping after no improvement for {patience} epochs")
            break

        if epoch % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, ckpt_path)

    print("=" * 70)
    print(f"FINAL BEST VALIDATION ACCURACY: {best_val_acc:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
