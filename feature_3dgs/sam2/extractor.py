import torch
import numpy as np
from typing import Dict, Any
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2

from feature_3dgs.extractor import AbstractFeatureExtractor

from sam2.sam2_image_predictor import SAM2ImagePredictor

def padding(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad image so that H and W are multiples of patch_size."""
    _, h, w = image.shape  # (C, H, W)
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image

TRANSFORMS_MEAN = [0.485, 0.456, 0.406]
TRANSFORMS_STD = [0.229, 0.224, 0.225]

class SAM2Extractor(AbstractFeatureExtractor):

    def __init__(self, model: nn.Module, patch_size: int = 16) -> torch.Tensor:
        self.model = model
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.patch_size = patch_size

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Extract features from input image.

        Args:
            image: Input image tensor in (C, H, W) format, values in [0, 1]

        Returns:
            Dict with 'features' key containing the feature map
        """
        # Convert to (H, W, C) required by SAM2
        x = (image * 255).astype(np.uint8).numpy().transpose(1, 2, 0)  
        x = TF.normalize(x, mean=TRANSFORMS_MEAN, std=TRANSFORMS_STD)
        x = padding(x, self.patch_size)

        self.predictor.set_image(x)

        # Shape: (1, C, H', W')
        # According to documentation, should be (1, 256, 64, 64)
        embedding = self.predictor.get_image_embedding() 

        features = embedding.squeeze(0)  # (C, H', W')
        return features

    def to(self, device) -> 'SAM2Extractor':
        self.model.to(device)
        return self