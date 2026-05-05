import os
from typing import Tuple

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from sam2.build_sam import build_sam2

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
    MODEL_SAM2_TINY: None,
    MODEL_SAM2_SMALL: None,
    MODEL_SAM2_BASE_PLUS: None,
    MODEL_SAM2_LARGE: None,
}

# Feature dimensions for each backbone (SAM2 uses 256 channels like SAM)
FEATURE_DIMS = {
    MODEL_SAM2_TINY: 256,
    MODEL_SAM2_SMALL: 256,
    MODEL_SAM2_BASE_PLUS: 256,
    MODEL_SAM2_LARGE: 256,
}

# Stride for SAM2 (image encoder stride, typically 16 like ViT)
SAM2_PATCH_SIZE = 16


# Model name -> checkpoint filename
MODEL_TO_CHECKPOINT = {
    MODEL_SAM2_TINY: "sam2.1_hiera_tiny.pt",
    MODEL_SAM2_SMALL: "sam2.1_hiera_small.pt",
    MODEL_SAM2_BASE_PLUS: "sam2.1_hiera_base_plus.pt",
    MODEL_SAM2_LARGE: "sam2.1_hiera_large.pt",
}

MODEL_TO_CONFIG = {
    MODEL_SAM2_TINY: "configs/sam2.1/sam2.1_hiera_t.yaml",
    MODEL_SAM2_SMALL: "configs/sam2.1/sam2.1_hiera_s.yaml",
    MODEL_SAM2_BASE_PLUS: "configs/sam2.1/sam2.1_hiera_b+.yaml",
    MODEL_SAM2_LARGE: "configs/sam2.1/sam2.1_hiera_l.yaml",
}

def SAM2FeatureExtractor(version: str = "sam2_hiera_base_plus", checkpoint_dir: str = "checkpoints") -> SAM2Extractor:
    assert version in MODELS, f"SAM2 version '{version}' not supported. Choose from: {MODELS}"
    local_path = os.path.join(checkpoint_dir, MODEL_TO_CHECKPOINT[version])
    sam2_model = build_sam2(config_file=MODEL_TO_CONFIG[version], ckpt_path=local_path)
    return SAM2Extractor(model=sam2_model, patch_size=SAM2_PATCH_SIZE)

def build_factory(version: str):
    def factory(embed_dim: int, checkpoint_dir="checkpoints", **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = SAM2FeatureExtractor(version, checkpoint_dir=checkpoint_dir)
        decoder = SAM2Decoder(
            in_channels=embed_dim,
            out_channels=FEATURE_DIMS[version],
            patch_size=SAM2_PATCH_SIZE,
            **configs
        )
        return extractor, decoder
    return factory

for version in MODELS:
    register_extractor_decoder(version, build_factory(version))