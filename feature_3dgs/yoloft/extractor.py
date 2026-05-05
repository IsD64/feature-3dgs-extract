import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox

from feature_3dgs import AbstractFeatureExtractor

def padding(image: torch.Tensor, stride: int, resolution: int) -> torch.Tensor:
    """Pad image so that H and W are multiples of stride.
    
    Args:
        image: (C, H, W) tensor in [0, 1] range
        stride: stride for padding
        resolution: target resolution (square)
    
    Returns:
        (H, W, C) tensor after letterbox padding
    """
    # Convert (C, H, W) to (H, W, C) for LetterBox
    origin_device = image.device
    image_hwc = image.permute(1, 2, 0).cpu().detach().numpy()  # (H, W, C)
    image_hwc = (image_hwc * 255).astype('uint8')  # Convert [0,1] float to [0,255] uint8

    letterbox = LetterBox(resolution, stride=stride, auto=False)
    image_padded = letterbox(image=image_hwc)  # (H, W, C) uint8

    # Convert back to (C, H, W) float [0, 1]
    image_padded = torch.from_numpy(image_padded).permute(2, 0, 1).float() / 255.0
    return image_padded.to(origin_device)


class YOLOExtractor(AbstractFeatureExtractor):
    def __init__(self, model: nn.Module, stride: int, resolution: int):
        self.model = model
        self.stride_size = stride
        self.resolution = resolution
        self.model.eval()
        self.spatial_features = {}

        def hook_fn(module, input, output):
            self.spatial_features["backbone"] = output.detach()

        for m in self.model.model.model:
            if m.__class__.__name__ == "SPPF":
                self._hook_handle = m.register_forward_hook(hook_fn)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        x = image
        x_padded = padding(x, self.stride_size, self.resolution).to("cuda" if torch.cuda.is_available() else "cpu")

        feats = self.model.predict(x_padded.unsqueeze(0), imgsz = self.resolution)
        # # feats = self.model.embed(x_padded.unsqueeze(0))
        # backbone = self.model
        # backbone.eval()
        # print(backbone)
        # print(x_padded.unsqueeze(0).shape)
        # feats = backbone(x_padded.unsqueeze(0))[0]
        # for i, m in enumerate(self.model.model.model):
        #         print(i, type(m))
        return self.spatial_features["backbone"].squeeze(0).clone()  # (D, H_p, W_p)

    def to(self, device) -> 'YOLOExtractor':
        self.model.to(device)
        return self
