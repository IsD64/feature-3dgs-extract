import os
from typing import Tuple
from ultralytics import YOLO

from feature_3dgs.extractor import AbstractFeatureExtractor
from feature_3dgs.decoder import AbstractTrainableDecoder
from feature_3dgs.registry import register_extractor_decoder

from .extractor import YOLOExtractor
from .decoder import YOLODecoder

INTERNAL_RESOLUTION = 640  # YOLO's default input size (letterbox-padded to square)
STRIDE_SIZE = 32  # YOLO's default stride size (downsampling factor)

MODEL_YOLOV8SEG = "yolov8seg"
MODEL_YOLO11SEG = "yolo11seg"
MODEL_YOLO26SEG = "yolo26seg"
MODEL_YOLOV8SEGX = "yolov8segx"
MODEL_YOLO11SEGX = "yolo11segx"
MODEL_YOLO26SEGX = "yolo26segx"

MODELS = [
    MODEL_YOLOV8SEG,
    MODEL_YOLO11SEG,
    MODEL_YOLO26SEG,
    MODEL_YOLOV8SEGX,
    MODEL_YOLO11SEGX,
    MODEL_YOLO26SEGX
]

MODEL_TO_FILENAME = {
    MODEL_YOLOV8SEG: "yolov8l-seg.pt",
    MODEL_YOLO11SEG: "yolo11l-seg.pt",
    MODEL_YOLO26SEG: "yolo26l-seg.pt",
    MODEL_YOLOV8SEGX: "yolov8x-seg.pt",
    MODEL_YOLO11SEGX: "yolo11x-seg.pt",
    MODEL_YOLO26SEGX: "yolo26x-seg.pt",
}

FEATURE_DIMS = {
    # TODO: confirm these dimensions by inspecting the actual model outputs
    MODEL_YOLOV8SEG: 512,
    MODEL_YOLO11SEG: 512,
    MODEL_YOLO26SEG: 512,
    MODEL_YOLOV8SEGX: 640,
    MODEL_YOLO11SEGX: 768,
    MODEL_YOLO26SEGX: 768,
}

# # https://github.com/ultralytics/assets/releases
# YOLO_VERSIONS = [
#     # YOLO26 (Ultralytics, Jan 14 2026) :contentReference[oaicite:0]{index=0}
#     "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",

#     # YOLO12 (Ultralytics) :contentReference[oaicite:1]{index=1}
#     "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",

#     # YOLO11 (Ultralytics, Sep 10 2024) :contentReference[oaicite:2]{index=2}
#     "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",

#     # YOLOv10 (THU-MIG / Ultralytics integration; includes special "b" balanced) :contentReference[oaicite:3]{index=3}
#     "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",

#     # YOLOv9 :contentReference[oaicite:4]{index=4}
#     "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",

#     # YOLOv8 :contentReference[oaicite:5]{index=5}
#     "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",

#     # YOLOv5u (Ultralytics modernized variants) :contentReference[oaicite:6]{index=6}
#     "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu",
#     "yolov5n6u", "yolov5s6u", "yolov5m6u", "yolov5l6u", "yolov5x6u",

#     # YOLOv3u (Ultralytics variants) :contentReference[oaicite:7]{index=7}
#     "yolov3-tinyu", "yolov3u", "yolov3-sppu",

#     # YOLO-NAS (not YOLOv* lineage, but Ultralytics-supported) :contentReference[oaicite:8]{index=8}
#     "yolo_nas_s", "yolo_nas_m", "yolo_nas_l",
# ]

# TODO build regstry to cover all these versions, or at least the most recent ones (YOLOv8+)

def YOLOFeatureExtractor(model_name: str, checkpoint_dir: str) -> YOLOExtractor:
    if model_name not in MODELS:
        raise ValueError(f"Unsupported YOLO model name: {model_name}")
    model_path = os.path.join(checkpoint_dir, MODEL_TO_FILENAME[model_name])
    model = YOLO(model_path)
    return YOLOExtractor(model=model, stride=STRIDE_SIZE, resolution=INTERNAL_RESOLUTION)

def build_factory(version: str):
    def factory(embed_dim: int, checkpoint_dir: str, **configs) -> Tuple[AbstractFeatureExtractor, AbstractTrainableDecoder]:
        extractor = YOLOFeatureExtractor(model_name=version, checkpoint_dir=checkpoint_dir)
        decoder = YOLODecoder(
            in_channels=embed_dim,
            out_channels=embed_dim,  # identity mapping; no dimensionality change
            **configs,
        )
        return extractor, decoder
    return factory

for model_name in MODELS:
    register_extractor_decoder(model_name, build_factory(model_name))