import torch
import numpy as np
from typing import Dict, Any, Optional
import cv2

from feature_3dgs.extractor import AbstractFeatureExtractor

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import Sam2ImagePredictor

class SAM2Extractor(AbstractFeatureExtractor):
    def __init__(self, version: str = "sam2_hiera_base_plus", checkpoint_dir: str = "checkpoints"):
        super().__init__()
        self.version = version
        self.checkpoint_dir = checkpoint_dir

        # Load SAM2 model
        # checkpoint_path = os.path.join(checkpoint_dir, f"{version}.pt")  # Adjust path
        # self.sam2_model = build_sam2(version, checkpoint_path)
        # self.predictor = Sam2ImagePredictor(self.sam2_model)

        # Placeholder
        self.predictor = None
        self.stride = 16  # SAM2 stride

    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Extract features from input image.

        Args:
            image: Input image tensor in (C, H, W) format, values in [0, 1]

        Returns:
            Dict with 'features' key containing the feature map
        """
        # Convert to numpy for SAM2 (expects HWC, uint8)
        image_np = self._tensor_to_numpy(image)

        # Set image and get embedding
        # self.predictor.set_image(image_np)
        # embedding = self.predictor.get_image_embedding()  # Shape: (1, C, H', W')

        # features = embedding.squeeze(0)  # (C, H', W')

        # Placeholder: return dummy features
        _, h, w = image.shape
        c = 256  # SAM2 feature channels
        h_feat = h // self.stride
        w_feat = w // self.stride
        features = torch.randn(c, h_feat, w_feat)
        return {"features": features}

    def _tensor_to_numpy(self, image: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor (C, H, W) [0,1] to numpy (H, W, C) [0,255]
        """
        # Transpose to (H, W, C)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Scale to [0, 255]
        image_np = (image_np * 255).astype(np.uint8)
        return image_np

    @property
    def out_channels(self) -> int:
        return 256  # SAM2 feature dimension

    @property
    def stride(self) -> int:
        return self.stride