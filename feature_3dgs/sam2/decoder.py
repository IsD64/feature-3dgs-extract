import torch
import torch.nn as nn
from typing import Dict, Any

from feature_3dgs.decoder import AbstractTrainableDecoder

class SAM2Decoder(AbstractTrainableDecoder):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 16, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Linear layer to map to desired output channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, features: Dict[str, Any]) -> torch.Tensor:
        """
        Decode features to output dimension.

        Args:
            features: Dict containing 'features' key with tensor of shape (C, H, W)

        Returns:
            Decoded features of shape (out_channels, H, W)
        """
        feat_map = features["features"]  # (C, H, W)

        # Add batch dimension
        feat_map = feat_map.unsqueeze(0)  # (1, C, H, W)

        b, c, h, w = feat_map.shape
        feat_flat = feat_map.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Apply linear transformation
        decoded = self.linear(feat_flat)  # (B, H*W, out_channels)

        # Reshape back to spatial
        decoded = decoded.permute(0, 2, 1).view(b, self.out_channels, h, w)  # (B, out_channels, H, W)

        return decoded.squeeze(0)  # Remove batch dim

    @property
    def out_channels(self) -> int:
        return self.out_channels