#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np


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
                # state_dict was provided — cannot reconstruct without model class
                raise RuntimeError(
                    f"The file at '{model_path}' appears to be a state_dict (torch.save(state_dict)). "
                    "KeypointSequenceClassifier expects a TorchScript model (saved with torch.jit.save) or a pickled nn.Module.\n"
                    "If you only have a state_dict, you must instantiate the model class and load the state_dict before saving a TorchScript version."
                )
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