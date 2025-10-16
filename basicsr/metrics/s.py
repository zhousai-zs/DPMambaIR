import os
from PIL import Image
import numpy as np
from tqdm import tqdm
def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order."""
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def to_y_channel(img):
    """Convert image to Y channel of YCbCr."""
    img = img.convert('YCbCr')
    return np.array(img)[:, :, 0]

def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(Image.fromarray(img1.astype(np.uint8)))
        img2 = to_y_channel(Image.fromarray(img2.astype(np.uint8)))

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))

def calculate_psnr_difference(gt_folder, restore1_folder, restore2_folder):
    differences = []
    psnr1_total = 0
    psnr2_total = 0
    count = 0

    for filename in tqdm(os.listdir(gt_folder)):
        if 'gt' in filename:
            continue

        gt_path = os.path.join(gt_folder, filename)
        restore1_path = os.path.join(restore1_folder, filename)
        restore2_path = os.path.join(restore2_folder, filename)

        if os.path.exists(restore1_path) and os.path.exists(restore2_path):
            gt_image = np.array(Image.open(gt_path))
            restore1_image = np.array(Image.open(restore1_path))
            restore2_image = np.array(Image.open(restore2_path))

            psnr1 = calculate_psnr(gt_image, restore1_image)
            psnr2 = calculate_psnr(gt_image, restore2_image)

            psnr1_total += psnr1
            psnr2_total += psnr2
            count += 1

            difference = psnr1 - psnr2
            differences.append((filename, psnr1, psnr2, difference))

    # Calculate averages
    psnr1_avg = psnr1_total / count if count > 0 else 0
    psnr2_avg = psnr2_total / count if count > 0 else 0

    # Sort by difference in descending order and get the top 10
    differences.sort(key=lambda x: x[3], reverse=True)
    top_10_differences = differences[:10]

    return top_10_differences, psnr1_avg, psnr2_avg

# 使用示例
gt_folder = '/zs/Data/AllInOneDataset/fog/test/gt'
# restore1_folder = '/zs/Restormer/results/test_DPMambaIR_jpeg_archived_20250331_011534/visualization/rain'
restore1_folder = '/zs/Restormer/results/test_DPMambaIR_jpeg/visualization/fog'
restore2_folder = '/zs/Restormer/results/test_OneRestore/visualization/fog'

top_10, psnr1_avg, psnr2_avg = calculate_psnr_difference(gt_folder, restore1_folder, restore2_folder)

for filename, psnr1, psnr2, diff in top_10:
    print(f"Filename: {filename}, Restore1 PSNR: {psnr1:.2f}, Restore2 PSNR: {psnr2:.2f}, Difference: {diff:.2f}")

print(f"Average PSNR for Restore1: {psnr1_avg:.2f}")
print(f"Average PSNR for Restore2: {psnr2_avg:.2f}")