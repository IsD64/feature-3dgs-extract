import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F

from .extractor import padding

from feature_3dgs.decoder import AbstractTrainableDecoder

class SAM2Decoder(AbstractTrainableDecoder):
    def __init__(self, *args, patch_size: int = 16, **configs):
        super().__init__(*args, **configs)
        self.patch_size = patch_size


    def decode_feature_map(self, feature_map):
        P = self.patch_size
        x = padding(feature_map, P)  # (C_enc, H', W')
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, P, P) / (P * P)
        return F.conv2d(x.unsqueeze(0), weight, self.linear.bias,stride=P).squeeze(0)

    def encode_feature_map(self, feature_map: torch.Tensor, camera) -> torch.Tensor:
        x = self.encode_feature_pixels(feature_map)           # (C_enc, H_p, W_p)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode='bilinear', align_corners=True).squeeze(0)