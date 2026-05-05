import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from gaussian_splatting.utils import PILtoTorch
import os
import torch

def query_features(folder_path, image, location):
    from feature_3dgs.dinov3ft import DINOv3LinearAvgDecoder, DINOv3ViTExtractor
    from feature_3dgs.sam2ft import SAM2Decoder, SAM2FeatureExtractor
    from feature_3dgs.yoloft import YOLODecoder, YOLOFeatureExtractor
    decoder_path = os.path.join(folder_path, "point_cloud/iteration_0/point_cloud.ply.decoder.pt")
    if "dino" in decoder_path:
        extractor = DINOv3ViTExtractor(version="dinov3_vitl16", checkpoint_dir="checkpoints").to("cuda")
        decoder = DINOv3LinearAvgDecoder(
            in_channels=48,
            out_channels=1024,
            patch_size=16
        )
    elif "sam" in decoder_path:
        extractor = SAM2FeatureExtractor("sam2_hiera_large", checkpoint_dir="checkpoints").to("cuda")
        decoder = SAM2Decoder(
            in_channels=48,
            out_channels=256,
            patch_size=16
        )
    elif "yolo26x" in decoder_path:
        extractor = YOLOFeatureExtractor("yolo26segx", checkpoint_dir="checkpoints").to("cuda")
        decoder = YOLODecoder(
            in_channels=48,
            out_channels = 768,
            stride_size=16,
            resolution=640
        )
    elif "yolo26" in decoder_path:
        extractor = YOLOFeatureExtractor("yolo26seg", checkpoint_dir="checkpoints").to("cuda")
        decoder = YOLODecoder(
            in_channels=48,
            out_channels = 512,
            stride_size=16,
            resolution=640
        )
    else:
        raise IOError("Incorrect input folder, must be dinov3_vit16, sam2_hiera_large, yolo26l-seg or yolo26x-seg models")

    decoder.load(decoder_path)
    decoder = decoder.to("cuda")
    image_torch = PILtoTorch(image).to("cuda")
    features = extractor(image_torch).to("cuda")
    print(features.is_inference())
    # print(f"feature device:{features.device}, decoder device: {decoder.linear.device}")
    encoded_features = decoder.encode_feature_pixels(features)  # (C_enc, H, W)
    # print(features.shape)

    query_position_x = int(location[0] * encoded_features.shape[2])
    query_position_y = int(location[1] * encoded_features.shape[1])
    queried_feature = encoded_features[:, query_position_y, query_position_x]  # (C_enc,)

    print(f"Queried feature at {location} (pixel ({query_position_x}, {query_position_y})): {queried_feature.shape}")

    return queried_feature

def query_ply(folder_path, encoded_features, threshold=0.6):
    import plyfile
    from plyfile import PlyData, PlyElement

    ply_path = os.path.join(folder_path, "point_cloud/iteration_0/point_cloud.ply")
    ply = plyfile.PlyData.read(ply_path)
    ply_np = ply["vertex"].data

    # print(ply.elements)

    ply_feature_path = os.path.join(folder_path, "point_cloud/iteration_0/point_cloud.ply.semantic.pt")
    ply_features_tensor = torch.load(ply_feature_path)

    cosine_similarities = torch.nn.functional.cosine_similarity(encoded_features.unsqueeze(0).to("cuda"), ply_features_tensor, dim=1)
    print(ply_features_tensor.shape)
    mask = cosine_similarities > threshold
    mask_np = mask.detach().cpu().numpy()
    filtered = ply_np[mask_np]
    filtered_out = ply_np[~mask_np]

    new_ply = PlyData([PlyElement.describe(filtered, "vertex")])
    new_ply.write("filtered.ply")
    new_ply_out = PlyData([PlyElement.describe(filtered_out, "vertex")])
    new_ply_out.write("filtered_out.ply")

    return

if __name__ == "__main__":
    input_size = ()
    extract_location = ()
    root = tk.Tk()

    file_path = filedialog.askopenfilename(initialdir="./outputs", title="Select an image to extract from", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    img = Image.open(file_path)
    tk_img = ImageTk.PhotoImage(img)

    canvas = tk.Canvas(root, width=img.width, height=img.height)
    input_size = (img.width, img.height)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    root.geometry(f"{img.width}x{img.height}")
    root.resizable(False, False)
    canvas.pack()

    def click(event):
        print(f"Clicked at: ({event.x}, {event.y})")
        extract_location = (event.x / img.width, event.y / img.height)  # Normalized coordinates
        messagebox.showinfo(f"Location Selected",
                            f"You selected {extract_location}, extracting the object from your selected location...")
        base_path = os.path.dirname(os.path.dirname(file_path))
        features = query_features(base_path, img, extract_location)
        query_ply(base_path, features)
        messagebox.showinfo("Extraction Complete", "Extraction complete! Check the current directory for 'filtered.ply' and 'filtered_out.ply'.")
        root.destroy()

    canvas.bind("<Button-1>", click)

    root.mainloop()
