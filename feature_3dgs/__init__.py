import warnings

from .gaussian_model import SemanticGaussianModel, CameraTrainableSemanticGaussianModel
from .decoder import AbstractSemanticDecoder, AbstractTrainableDecoder, LinearDecoder
from .extractor import AbstractFeatureExtractor, FeatureCameraDataset, TrainableFeatureCameraDataset
from .registry import register_extractor_decoder, get_available_extractor_decoders, build_extractor_decoder


def _register_optional_backend(module_name: str) -> None:
    try:
        __import__(f"{__name__}.{module_name}")
    except Exception as exc:
        warnings.warn(
            f"Skipping optional backend '{module_name}': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


for _backend in ("dinov3ft", "yoloft", "sam2ft", "ttt3rft"):
    _register_optional_backend(_backend)
