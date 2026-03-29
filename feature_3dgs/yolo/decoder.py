import torch
import torch.nn as nn
from torch.nn import functional as F

from feature_3dgs.decoder import LinearDecoder
from .extractor import padding

class YOLODecoder(LinearDecoder):

    def __init__(self, *args, stride_size: int, resolution: int, **configs):
        super().__init__(*args, **configs)
        self.stride_size = stride_size
        self.resolution = resolution

    def decode_feature_map(self, feature_map):
        S = self.stride_size
        x = padding(feature_map, stride=S, resolution=self.resolution)  # (C_feat, H', W')
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, S, S) / (S * S)
        return F.conv2d(x.unsqueeze(0), weight, self.linear.bias, stride=S).squeeze(0)

    def encode_feature_map(self, feature_map, camera):
        x = self.encode_feature_pixels(feature_map)           # (C_enc, H_p, W_p)
        return F.interpolate(x.unsqueeze(0), size=(camera.image_height, camera.image_width), mode='bilinear', align_corners=True).squeeze(0)
