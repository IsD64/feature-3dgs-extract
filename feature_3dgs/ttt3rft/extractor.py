from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from feature_3dgs.extractor import AbstractFeatureExtractor

if TYPE_CHECKING:
    from dust3r.model import ARCroco3DStereo


PATCH_SIZE = 16


def ensure_ttt3r_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2] / "submodules" / "TTT3R"
    src_root = repo_root / "src"
    if not src_root.is_dir():
        raise ImportError(
            f"TTT3R source was not found at {src_root}. "
            "Clone the upstream repo into submodules/TTT3R first."
        )
    for path in (repo_root, src_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


def preprocess_image_tensor(
    image: torch.Tensor,
    size: int,
    square_ok: bool = False,
) -> torch.Tensor:
    """Mirror TTT3R's image resize + center-crop preprocessing in torch space."""
    if image.ndim != 3:
        raise ValueError(f"Expected a (C, H, W) tensor, got shape {tuple(image.shape)}")

    _, in_h, in_w = image.shape
    if size == 224:
        target_long_edge = round(size * max(in_w / in_h, in_h / in_w))
    else:
        target_long_edge = size

    scale = target_long_edge / max(in_h, in_w)
    out_h = max(1, int(round(in_h * scale)))
    out_w = max(1, int(round(in_w * scale)))

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    cx, cy = out_w // 2, out_h // 2
    if size == 224:
        half = min(cx, cy)
        left, right = cx - half, cx + half
        top, bottom = cy - half, cy + half
    else:
        half_w = ((2 * cx) // PATCH_SIZE) * (PATCH_SIZE // 2)
        half_h = ((2 * cy) // PATCH_SIZE) * (PATCH_SIZE // 2)
        if not square_ok and out_w == out_h:
            half_h = int(3 * half_w / 4)
        left, right = cx - half_w, cx + half_w
        top, bottom = cy - half_h, cy + half_h

    return resized[:, top:bottom, left:right].contiguous()


def prepare_input_ttt3r(
    images: Iterable[torch.Tensor],
    size: int,
    square_ok: bool = False,
    revisit: int = 1,
    update: bool = True,
    reset_interval: int = 10000,
) -> list[dict[str, torch.Tensor | int | str]]:
    prepared_images = []
    for image in images:
        img = preprocess_image_tensor(image=image, size=size, square_ok=square_ok)
        img = ((img - 0.5) / 0.5).contiguous()
        prepared_images.append(
            dict(
                img=img.unsqueeze(0),
                true_shape=np.int32([img.shape[-2:]]),
                idx=len(prepared_images),
                instance=str(len(prepared_images)),
            )
        )

    views = []
    for i, image in enumerate(prepared_images):
        view = {
            "img": image["img"],
            "ray_map": torch.full(
                (
                    image["img"].shape[0],
                    6,
                    image["img"].shape[-2],
                    image["img"].shape[-1],
                ),
                torch.nan,
                dtype=image["img"].dtype,
            ),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        views.append(view)
        if (i + 1) % reset_interval == 0:
            overlap_view = deepcopy(view)
            overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap_view)

    if revisit > 1:
        revisited_views = []
        for revisit_idx in range(revisit):
            for view_idx, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = revisit_idx * len(views) + view_idx
                new_view["instance"] = str(revisit_idx * len(views) + view_idx)
                if revisit_idx > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                revisited_views.append(new_view)
        return revisited_views

    return views


def state_feature_to_map(
    state_feat: torch.Tensor,
    state_pos: torch.Tensor,
) -> torch.Tensor:
    if state_feat.ndim != 3 or state_pos.ndim != 3:
        raise ValueError(
            "TTT3R state tensors must have shape (B, N, C) and (B, N, 2); "
            f"got {tuple(state_feat.shape)} and {tuple(state_pos.shape)}"
        )
    if state_feat.shape[0] != 1 or state_pos.shape[0] != 1:
        raise ValueError(
            "TTT3RExtractor currently expects a single-view batch per step; "
            f"got batch size {state_feat.shape[0]}"
        )

    tokens = state_feat[0]
    positions = state_pos[0].to(torch.long)
    grid_h = int(positions[:, 0].max().item()) + 1
    grid_w = int(positions[:, 1].max().item()) + 1
    feature_map = tokens.new_zeros(tokens.shape[-1], grid_h, grid_w)
    feature_map[:, positions[:, 0], positions[:, 1]] = tokens.transpose(0, 1)
    return feature_map.contiguous()


class TTT3RExtractor(AbstractFeatureExtractor):
    """TTT3R recurrent-state feature extractor.

    The upstream model maintains a 2D latent state grid. This wrapper exposes
    that state after each view as a dense feature map shaped like the paired
    semantic decoder output: ``(C_feat, H_p, W_p)``.
    """

    def __init__(
        self,
        model: "ARCroco3DStereo",
        resolution: int = 512,
        patch_size: int = PATCH_SIZE,
        square_ok: bool = False,
    ):
        self.model = model
        self.input_resolution = resolution
        self.patch_size = patch_size
        self.square_ok = square_ok
        self.feature_dim = int(self.model.dec_embed_dim)
        self.state_size = int(self.model.state_size)
        self.state_width = int(self.state_size**0.5)
        if self.state_width % 2 == 1:
            self.state_width += 1
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TTT3R requires multiple images. Use extract_all() instead."
        )

    @torch.no_grad()
    def extract_all(self, images: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        image_list = list(images)
        if not image_list:
            return iter(())

        device = image_list[0].device
        inputs = prepare_input_ttt3r(
            image_list,
            size=self.input_resolution,
            square_ok=self.square_ok,
        )
        _, _, state_args = self.model.forward_recurrent(
            inputs,
            device=device,
            ret_state=True,
        )

        feature_maps = []
        for state_feat, state_pos, _, _, _ in state_args[1:]:
            feature_maps.append(state_feature_to_map(state_feat, state_pos))
        return iter(feature_maps)

    def to(self, device) -> "TTT3RExtractor":
        self.model.to(device)
        return self
