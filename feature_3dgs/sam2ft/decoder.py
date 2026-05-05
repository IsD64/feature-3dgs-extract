import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F

from .extractor import padding, resize_and_pad_to_square

from feature_3dgs.decoder import LinearDecoder

class SAM2Decoder(LinearDecoder):
    def __init__(self, *args, patch_size: int = 16, **configs):
        super().__init__(*args, **configs)
        self.patch_size = patch_size


    def decode_feature_map(self, feature_map):
        P = self.patch_size
        print(f"decoder input feature map has shape {feature_map.shape}")
        x = resize_and_pad_to_square(feature_map, 1024)  # (C_enc, H', W')
        print(f"padded input has shape {x.shape}")
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, P, P) / (P * P)
        print(f"weight has shape")
        decoded = F.conv2d(x.unsqueeze(0), weight, self.linear.bias,stride=P).squeeze(0)
        print(f"decoded has shape {decoded.shape}")
        return decoded

    def encode_feature_map(self, feature_map: torch.Tensor, camera) -> torch.Tensor:
        x = self.encode_feature_pixels(feature_map)           # (C_enc, H_p, W_p)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode='bilinear', align_corners=True).squeeze(0)