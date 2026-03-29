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
    MODEL_YOLOV8SEG: 512,
    MODEL_YOLO11SEG: 512,
    MODEL_YOLO26SEG: 512,
    MODEL_YOLOV8SEGX: 640,
    MODEL_YOLO11SEGX: 768,
    MODEL_YOLO26SEGX: 768,
}

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
            out_channels=FEATURE_DIMS[version],  # identity mapping; no dimensionality change
            **configs,
        )
        return extractor, decoder
    return factory

for model_name in MODELS:
    register_extractor_decoder(model_name, build_factory(model_name))