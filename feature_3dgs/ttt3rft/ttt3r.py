from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import torch

from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.registry import register_extractor_decoder

from .decoder import TTT3RLinearAvgDecoder
from .extractor import PATCH_SIZE, TTT3RExtractor, ensure_ttt3r_on_path

if TYPE_CHECKING:
    from dust3r.model import ARCroco3DStereo


MODEL_TTT3R = "ttt3r"
DEFAULT_CHECKPOINT = "checkpoints/cut3r_512_dpt_4_64.pth"
DEFAULT_INPUT_RESOLUTION = 512


def _resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_trusted_ttt3r_model(checkpoint: Path, device: str):
    from dust3r import model as dust3r_model

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed",
        "PatchEmbedDust3R",
    )
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True",
            "landscape_only=False",
        )
    if "landscape_only=False" not in args:
        raise ValueError(f"Unexpected TTT3R model args: {args}")

    model = eval(args, dust3r_model.__dict__)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device)


def load_TTT3R(
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_update_type: str = "ttt3r",
    device: str | None = None,
) -> "ARCroco3DStereo":
    ensure_ttt3r_on_path()

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"TTT3R checkpoint not found at {checkpoint_path}. "
            "Download `cut3r_512_dpt_4_64.pth` into the checkpoints directory."
        )

    resolved_device = _resolve_device(device)
    model = _load_trusted_ttt3r_model(checkpoint_path, resolved_device)
    model.config.model_update_type = model_update_type
    model.eval()
    return model


def TTT3RFeatureExtractor(
    checkpoint: str = DEFAULT_CHECKPOINT,
    input_resolution: int = DEFAULT_INPUT_RESOLUTION,
    model_update_type: str = "ttt3r",
    device: str | None = None,
) -> TTT3RExtractor:
    model = load_TTT3R(
        checkpoint=checkpoint,
        model_update_type=model_update_type,
        device=device,
    )
    return TTT3RExtractor(
        model=model,
        resolution=input_resolution,
        patch_size=PATCH_SIZE,
    )


def build_factory():
    def factory(
        embed_dim: int,
        checkpoint: str = DEFAULT_CHECKPOINT,
        img_load_resolution: int = DEFAULT_INPUT_RESOLUTION,
        model_update_type: str = "ttt3r",
        device: str | None = None,
        **configs,
    ) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = TTT3RFeatureExtractor(
            checkpoint=checkpoint,
            input_resolution=img_load_resolution,
            model_update_type=model_update_type,
            device=device,
        )
        decoder = TTT3RLinearAvgDecoder(
            in_channels=embed_dim,
            out_channels=extractor.feature_dim,
            patch_size=extractor.patch_size,
            input_resolution=img_load_resolution,
            state_size=extractor.state_size,
            state_width=extractor.state_width,
            **configs,
        )
        return extractor, decoder

    return factory


register_extractor_decoder(MODEL_TTT3R, build_factory())
