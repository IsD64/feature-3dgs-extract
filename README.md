# Prerequisites

* [Python] v3.11 tested to run and recommended
* [Pytorch] >= v2.4 recommended
* [CUDA Toolkit] 12.4 recommended, match with PyTorch version
* [gsplat](https://github.com/nerfstudio-project/gsplat)
* [tkinter] for interactive UI after training

# Install

```shell
pip install --upgrade git+https://github.com/facebookresearch/dinov3@main
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
pip install --upgrade git+https://github.com/facebookresearch/sam2@main
pip install --upgrade ultralytics

pip install --upgrade git+https://github.com/yindaheng98/feature-3dgs.git@main --no-build-isolation
# or
git clone --recursive https://github.com/yindaheng98/feature-3dgs.git
cd feature-3dgs
pip install --target . --upgrade . --no-deps --no-build-isolation
```

# Download Checkpoints
In order to run the semantic extractors needed for this repository, you need to download the checkpoint weights of the used models. Please download them and put them in the checkpoints folder:
```
checkpoints
├── dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
├── dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
├── sam2.1_hiera_base_plus.pt
├── sam2.1_hiera_large.pt
├── sam2.1_hiera_small.pt
├── sam2.1_hiera_tiny.pt
├── yolo11l-seg.pt
├── yolo11m-seg.pt
├── yolo11n-pose.pt
├── yolo11s-seg.pt
├── yolo11x-seg.pt
├── yolo26l-seg.pt
├── yolo26m-seg.pt
├── yolo26n-seg.pt
├── yolo26s-seg.pt
├── yolo26x-seg.pt
├── yolov8l.pt
├── yolov8l-seg.pt
├── yolov8m.pt
├── yolov8n.pt
├── yolov8s.pt
├── yolov8x.pt
└── yolov8x-seg.pt
```
Some checkpoints (e.g.SAM) is not openly available, you need to request for access to download in kaggle or their official website.

# Command-Line Usage
Our test assumes that a pretrained standard 3d gaussian splatting already exists. To train the semantic decoder and add semantic features to 3d gaussians, first put the pretrained model in the data/ folder, then run the following commands. The data/ folder should be organized in this order:
```
data/{scene_name}
├── cameras.json
├── cfg_args
├── images
│   └── *.jpg
├── input.ply
├── point_cloud
│   └── iteration_*
│       └── point_cloud.ply
└── sparse
    └── 0
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        └── project.ini
```
## Train

```shell
python feature_3dgs/train.py \
    --name {data/semantic_model_name} --embed_dim 48 \
    -s data/{scene_name} \
    -d output/{scene_name}-semantic-{semantic_model_abbreviation} \
    -i 0 \
    --mode camera \
    -l {data/semantic_model_name/path_to_pretrained_pointcloud} \
    --no_load_semantic --empty_cache_every_step \
    -oposition_lr_max_steps=1000 \
    -osemantic_decoder_lr_max_steps=1000
```

## Render

```shell
python feature_3dgs/render.py \
    --name {semantic_model_name} --embed_dim 48 \
    -s data/{scene_name} \
    -d output/{scene_name}-semantic-{semantic_model_abbreviation} \
    -i 0 \
    --load_camera output/{scene_name}-semantic-{semantic_model_abbreviation}/cameras.json
```

Rendered feature maps are PCA-projected to RGB and saved alongside ground-truth feature visualisations. This render function also collects evaluation metrics and store them at output/{scene_name}-semantic-{semantic_model_abbreviation}/ours_0

## Interactive Viewer

```shell
python feature_3dgs/viewer.py \
    --name {semantic_model_name} --embed_dim 48 \
    -s data/truck \
    -d output/{scene_name}-semantic-{semantic_model_abbreviation} \
    -i 0 --port 8080
```

## Interactive Object Extractor
please put a photo inside outputs/{scene_name}-semantic-{semantic_model_abbreviation}/images for the extractor to correctly identify model path. You can then choose that photo inside the GUI and click any object to extract it to ./filtered.ply and ./filtered_out.ply
```shell
python extract_item_gaussian.py

```
Opens an interactive viewer that renders PCA-colorized semantic feature maps in real time from free-viewpoint camera controls.
