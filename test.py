import PIL
import torch
from ultralytics import YOLO
from feature_3dgs.yolo.registry import MODEL_TO_FILENAME, MODEL_YOLOV8SEG, YOLOFeatureExtractor
from gaussian_splatting.camera import read_image
print(torch.cuda.is_available())
print(torch.version.cuda)



model = YOLOFeatureExtractor(model_name=MODEL_YOLOV8SEG, checkpoint_dir="./checkpoints")
picture: torch.Tensor = read_image("truck.jpg")
print(picture.shape)  # should be (C, H, W)
features = model(picture)
# print(picture.permute(1, 2, 0).shape)
# print(picture.permute(1, 2, 0).permute(2, 0, 1).shape)