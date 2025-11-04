#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
import argparse

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


class KeyPointSequenceClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_sequence_classifier/exported_model.pt',
        device='cpu'
    ):
        self.device = torch.device(device)
        # Prefer loading a TorchScript module, but fall back to other helpers
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        except RuntimeError as e:
            # Common cause: the file is a state_dict or non-TorchScript checkpoint
            try:
                loaded = torch.load(model_path, map_location=self.device)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model at '{model_path}' as TorchScript (error: {e}) "
                    f"and torch.load also failed (error: {e2}).\n"
                    "Please ensure the model file is a TorchScript archive created with torch.jit.save()."
                )

            # If torch.load returns an nn.Module instance, use it
            if isinstance(loaded, torch.nn.Module):
                self.model = loaded.to(self.device)
                self.model.eval()
            elif isinstance(loaded, dict):
                # Simple and explicit path: assume `loaded` is a state_dict or a dict containing the state_dict
                # Prefer common nested keys, otherwise treat the dict as the state_dict directly.
                if "state_dict" in loaded:
                    sd = loaded["state_dict"]
                elif "model_state_dict" in loaded:
                    sd = loaded["model_state_dict"]
                else:
                    sd = loaded

                # Instantiate the model with sensible defaults and load the state dict
                model_instance = SignLanguageModel(input_size=159 * 3, hidden_size=2048, num_classes=25)

                # Strip 'module.' prefix that can appear when saved from DataParallel
                new_sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }

                model_instance.load_state_dict(new_sd)
                model_instance.to(self.device)
                model_instance.eval()
                self.model = model_instance
                # Try to set feature_dim from the model if possible
                try:
                    self.feature_dim = int(model_instance.input_proj[0].in_features)
                except Exception:
                    self.feature_dim = None
                # sequence length isn't encoded in the model architecture; leave as None
                self.sequence_length = None
            else:
                raise RuntimeError(
                    f"torch.load returned an unsupported object type: {type(loaded)}. "
                    "Provide a TorchScript module or a pickled nn.Module."
                )

    def __call__(self, sequence):
        """
        sequence: list/np.ndarray of shape (T, D) where T is variable sequence length
        """
        x = np.array(sequence, dtype=np.float32)

        # Check only feature dimension, allow variable sequence length
        if x.ndim != 2:
            raise ValueError(f"Input must be 2D (sequence_length, features), but got shape {x.shape}")

        # Add batch dimension → (1, T, D)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)  # shape [1, num_classes]
            pred = torch.argmax(output, dim=1).item()

        return pred