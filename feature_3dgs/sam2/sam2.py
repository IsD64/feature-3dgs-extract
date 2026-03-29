import os
from typing import Tuple

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import SAM2Extractor
from .decoder import SAM2Decoder

# SAM2 models
MODEL_SAM2_TINY = "sam2_hiera_tiny"
MODEL_SAM2_SMALL = "sam2_hiera_small"
MODEL_SAM2_BASE_PLUS = "sam2_hiera_base_plus"
MODEL_SAM2_LARGE = "sam2_hiera_large"

MODELS = [
    MODEL_SAM2_TINY,
    MODEL_SAM2_SMALL,
    MODEL_SAM2_BASE_PLUS,
    MODEL_SAM2_LARGE,
]

# Model name -> factory function (placeholder, need to import from sam2)
MODEL_TO_FACTORY = {
    MODEL_SAM2_TINY: None,  # sam2_hiera_tiny
    MODEL_SAM2_SMALL: None,  # sam2_hiera_small
    MODEL_SAM2_BASE_PLUS: None,  # sam2_hiera_base_plus
    MODEL_SAM2_LARGE: None,  # sam2_hiera_large
}

# Feature dimensions for each backbone (SAM2 uses 256 channels like SAM)
FEATURE_DIMS = {
    MODEL_SAM2_TINY: 256,
    MODEL_SAM2_SMALL: 256,
    MODEL_SAM2_BASE_PLUS: 256,
    MODEL_SAM2_LARGE: 256,
}

# Stride for SAM2 (image encoder stride, typically 16 like ViT)
STRIDE = 16

# Model name -> config file
MODEL_TO_CONFIG = {
    MODEL_SAM2_TINY: "sam2_hiera_t.yaml",
    MODEL_SAM2_SMALL: "sam2_hiera_s.yaml",
    MODEL_SAM2_BASE_PLUS: "sam2_hiera_b+.yaml",
    MODEL_SAM2_LARGE: "sam2_hiera_l.yaml",
}

# Model name -> checkpoint filename
MODEL_TO_CHECKPOINT = {
    MODEL_SAM2_TINY: "sam2_hiera_tiny.pt",
    MODEL_SAM2_SMALL: "sam2_hiera_small.pt",
    MODEL_SAM2_BASE_PLUS: "sam2_hiera_base_plus.pt",
    MODEL_SAM2_LARGE: "sam2_hiera_large.pt",
}

def SAM2ExtractorFactory(version: str = "sam2_hiera_base_plus", checkpoint_dir: str = "checkpoints") -> SAM2Extractor:
    assert version in MODELS, f"SAM2 version '{version}' not supported. Choose from: {MODELS}"
    return SAM2Extractor(version=version, checkpoint_dir=checkpoint_dir)

def build_factory(version: str):
    def factory(embed_dim: int, checkpoint_dir="checkpoints", **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = SAM2ExtractorFactory(version, checkpoint_dir=checkpoint_dir)
        decoder = SAM2Decoder(
            in_channels=embed_dim,
            out_channels=FEATURE_DIMS[version],
            stride=STRIDE,
            **configs
        )
        return extractor, decoder
    return factory

for version in MODELS:
    register_extractor_decoder(version, build_factory(version))