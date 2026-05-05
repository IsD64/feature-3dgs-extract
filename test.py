import PIL
import torch
from ultralytics import YOLO
from feature_3dgs.yoloft.registry import MODEL_TO_FILENAME, MODEL_YOLOV8SEG, YOLOFeatureExtractor
from feature_3dgs.dinov3ft.vit import DINOv3ViTExtractor
from gaussian_splatting.camera import read_image
print(torch.cuda.is_available())
print(torch.version.cuda)



model = YOLOFeatureExtractor(model_name=MODEL_YOLOV8SEG, checkpoint_dir="./checkpoints")
dinomodel = DINOv3ViTExtractor(version="dinov3_vitl16", checkpoint_dir="./checkpoints")
picture: torch.Tensor = read_image("truck.jpg")
# print(picture.shape)  # should be (C, H, W)
features2 = dinomodel(picture)
features = model(picture)
print(features.shape)
print(features2.shape)
# for result in features:
#     embeddings = result.embeddings
#     print(embeddings.shape)  # should be (D, H_p, W_p)
# print(picture.permute(1, 2, 0).shape)
# print(picture.permute(1, 2, 0).permute(2, 0, 1).shape)