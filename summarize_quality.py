import csv
import sys
import os

def summarize_quality_metrics(csv_path):

    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} does not exist")
        return

    psnr_pca_values = []
    ssim_pca_values = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Strip whitespace from keys to handle malformed headers
                    clean_row = {k.strip(): v for k, v in row.items()}
                    psnr_pca_values.append(float(clean_row['psnr_pca']))
                    ssim_pca_values.append(float(clean_row['ssim_pca']))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row: {row}. Error: {e}")
                    continue

    except Exception as e:
        print(f"Error reading file {csv_path}: {e}")
        return

    if not psnr_pca_values:
        print("Error: No valid PSNR_PCA values found")
        return

    if not ssim_pca_values:
        print("Error: No valid SSIM_PCA values found")
        return

    avg_psnr_pca = sum(psnr_pca_values) / len(psnr_pca_values)
    avg_ssim_pca = sum(ssim_pca_values) / len(ssim_pca_values)

    print(f"File: {csv_path}")
    print(f"Number of samples: {len(psnr_pca_values)}")
    print(f"Average PSNR(PCA): {avg_psnr_pca:.4f}")
    print(f"Average SSIM(PCA): {avg_ssim_pca:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise NotADirectoryError("Please enter the path to the output scene folder!")
    scene_path = sys.argv[1]
    csv_path = os.path.join(scene_path, "ours_0/quality_semantic.csv")
    # print(csv_path)
    summarize_quality_metrics(csv_path)