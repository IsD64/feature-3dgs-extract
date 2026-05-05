import torch
import torch.nn.functional as F
from gaussian_splatting import Camera

from feature_3dgs.decoder import LinearDecoder

from .extractor import preprocess_image_tensor


class TTT3RLinearAvgDecoder(LinearDecoder):
    def __init__(
        self,
        *args,
        patch_size: int,
        input_resolution: int = 512,
        state_size: int = 768,
        state_width: int = 28,
        square_ok: bool = False,
        **configs,
    ):
        super().__init__(*args, **configs)
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.state_size = state_size
        self.state_width = state_width
        self.square_ok = square_ok

    def decode_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Project into TTT3R's fixed square recurrent-state grid."""
        P = self.patch_size
        x = preprocess_image_tensor(
            feature_map,
            size=self.input_resolution,
            square_ok=self.square_ok,
        )
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.state_width * P, self.state_width * P),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        weight = self.linear.weight[:, :, None, None].expand(-1, -1, P, P) / (P * P)
        decoded = F.conv2d(x.unsqueeze(0), weight, self.linear.bias, stride=P).squeeze(0)
        if self.state_size < self.state_width * self.state_width:
            decoded = decoded.reshape(decoded.shape[0], -1)
            decoded[:, self.state_size:] = 0
            decoded = decoded.reshape(decoded.shape[0], self.state_width, self.state_width)
        return decoded

    def encode_feature_map(
        self,
        feature_map: torch.Tensor,
        camera: Camera,
    ) -> torch.Tensor:
        x = self.encode_feature_pixels(feature_map)
        return F.interpolate(
            x.unsqueeze(0),
            size=(camera.image_height, camera.image_width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
