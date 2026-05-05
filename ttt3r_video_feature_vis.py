from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from feature_3dgs.ttt3rft.ttt3r import TTT3RFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PCA-colored TTT3R feature visualization video.",
    )
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--frame-stride", default=8, type=int)
    parser.add_argument("--max-frames", default=12, type=int)
    parser.add_argument("--input-resolution", default=512, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    return parser.parse_args()


def load_video_frames(
    video_path: str,
    frame_stride: int,
    max_frames: int,
) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_bgr: list[np.ndarray] = []
    frame_index = 0

    while len(frames_bgr) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_stride == 0:
            frames_bgr.append(frame)
        frame_index += 1

    cap.release()
    if not frames_bgr:
        raise ValueError(f"No frames were extracted from {video_path}")
    return frames_bgr, fps / frame_stride


def bgr_to_torch(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return tensor.contiguous()


def project_feature_maps(feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
    stacked = torch.cat(
        [feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0]) for feature_map in feature_maps],
        dim=0,
    )
    pca = PCA(n_components=3)
    pca.fit(stacked.cpu().numpy())

    projected_maps: list[torch.Tensor] = []
    for feature_map in feature_maps:
        h, w = feature_map.shape[1:]
        flat = feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0]).cpu().numpy()
        projected = pca.transform(flat)
        projected_maps.append(torch.from_numpy(projected).float().reshape(h, w, 3).permute(2, 0, 1))

    mins = torch.stack([proj.amin(dim=(1, 2)) for proj in projected_maps]).amin(dim=0)
    maxs = torch.stack([proj.amax(dim=(1, 2)) for proj in projected_maps]).amax(dim=0)
    denom = torch.clamp(maxs - mins, min=1e-6)

    normalized = []
    for proj in projected_maps:
        normalized.append(((proj - mins[:, None, None]) / denom[:, None, None]).clamp(0, 1))
    return normalized


def make_panel(
    frame_bgr: np.ndarray,
    feature_rgb: torch.Tensor,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    feature_up = F.interpolate(
        feature_rgb.unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    feature_bgr = (
        feature_up.permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255.0
    ).astype(np.uint8)

    panel = np.concatenate([frame_bgr, feature_bgr], axis=1)
    cv2.putText(panel, "Input Frame", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "TTT3R PCA Features", (w + 24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()

    frames_bgr, out_fps = load_video_frames(
        video_path=args.video,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    frame_tensors = [bgr_to_torch(frame) for frame in frames_bgr]

    extractor = TTT3RFeatureExtractor(
        input_resolution=args.input_resolution,
        device=args.device,
    )
    with torch.no_grad():
        feature_maps = list(extractor.extract_all(frame_tensors))

    projected_maps = project_feature_maps(feature_maps)
    panels = [make_panel(frame, proj) for frame, proj in zip(frames_bgr, projected_maps)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = panels[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for panel in panels:
        writer.write(panel)
    writer.release()

    print(f"Saved TTT3R feature visualization to {output_path}")
    print(f"Frames: {len(panels)}, output FPS: {out_fps:.2f}")


if __name__ == "__main__":
    main()
