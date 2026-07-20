import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import frangi
import torch
from torchvision import transforms
import random
from tqdm import tqdm

# 设置随机种子
np.random.seed(42)
random.seed(42)

# 数据集和图像数量
datasets = ['DRIVE', 'CHASEDB1', 'HRF']
num_images_per_dataset = 3

# 图像增强方法
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """应用CLAHE增强"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_local_contrast_normalization(image, kernel_size=15):
    """应用局部对比度归一化"""
    # 转换为浮点数
    image_float = image.astype(np.float32) / 255.0
    
    # 计算局部均值
    blur = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), 0)
    
    # 计算局部标准差
    squared_diff = cv2.GaussianBlur(np.square(image_float - blur), 
                                   (kernel_size, kernel_size), 0)
    local_std = np.sqrt(np.maximum(squared_diff, 0)) + 1e-8
    
    # 归一化
    normalized = (image_float - blur) / local_std
    
    # 缩放到[0,1]
    min_val = normalized.min()
    max_val = normalized.max()
    normalized = (normalized - min_val) / (max_val - min_val + 1e-8)
    
    return (normalized * 255).astype(np.uint8)

def apply_frangi_filter(image, sigma_range=(1, 3), num_sigma=3):
    """应用Frangi滤波器增强细小血管"""
    # 归一化到[0,1]
    norm_img = image / 255.0
    # 应用Frangi滤波器
    sigmas = np.linspace(sigma_range[0], sigma_range[1], num_sigma)
    filtered = frangi(norm_img, sigmas=sigmas, black_ridges=False)
    # 归一化滤波结果
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-8)
    return (filtered * 255).astype(np.uint8)

def apply_combined_enhancement(image):
    """应用组合增强"""
    # 应用CLAHE
    clahe_img = apply_clahe(image)
    
    # 应用局部对比度归一化
    lcn_img = apply_local_contrast_normalization(image)
    
    # 应用Frangi滤波器
    frangi_img = apply_frangi_filter(image)
    
    # 融合不同的增强结果 (0.5*原始CLAHE增强 + 0.25*LCN + 0.25*Frangi)
    enhanced = 0.5 * (clahe_img/255.0) + 0.25 * (lcn_img/255.0) + 0.25 * (frangi_img/255.0)
    return (enhanced * 255).astype(np.uint8)

def enhance_mask(mask):
    """增强掩码的可视化效果"""
    # 确保掩码是二值图像
    binary_mask = (mask > 127).astype(np.uint8) * 255
    # 应用形态学操作使血管更清晰
    kernel = np.ones((3,3), np.uint8)
    enhanced_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return enhanced_mask

def visualize_dataset(dataset_name):
    """可视化指定数据集的图像增强效果"""
    print(f"Processing {dataset_name} dataset...")
    
    # 设置图像路径
    images_dir = f'preprocessed_data/{dataset_name}/train/images'
    masks_dir = f'preprocessed_data/{dataset_name}/train/masks'
    
    # 检查路径是否存在，如果不存在则尝试使用测试集
    if not os.path.exists(images_dir):
        print(f"Training set path does not exist, trying alternative path...")
        # 尝试本地路径
        images_dir = f'./data/{dataset_name}/train/images'
        masks_dir = f'./data/{dataset_name}/train/masks'
        
        # 如果本地路径也不存在，尝试使用测试集
        if not os.path.exists(images_dir):
            images_dir = f'./data/{dataset_name}/test/images'
            masks_dir = f'./data/{dataset_name}/test/masks'
    
    # 确保目录存在
    if not os.path.exists(images_dir):
        print(f"Error: Unable to find image directory for {dataset_name}")
        return
    
    # 获取图像列表
    image_files = sorted(os.listdir(images_dir))
    
    # 随机选择图像
    if len(image_files) <= num_images_per_dataset:
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, num_images_per_dataset)
    
    # 创建结果目录
    os.makedirs('visualization_results', exist_ok=True)
    
    # 处理每张选定的图像
    for img_file in selected_images:
        img_path = os.path.join(images_dir, img_file)
        
        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Unable to read image: {img_path}")
            continue
        
        # 应用图像增强
        enhanced_img = apply_combined_enhancement(image)
        
        # 读取掩码
        mask_file = img_file.replace('.jpg', '.png').replace('.JPG', '.png').replace('.tif', '.png')
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Unable to find mask file: {mask_path}")
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Unable to read mask: {mask_path}")
            continue
            
        # 增强掩码
        enhanced_mask = enhance_mask(mask)
        
        # 可视化结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(142)
        plt.title('Enhanced Image')
        plt.imshow(enhanced_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(143)
        plt.title('Original Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(144)
        plt.title('Enhanced Mask')
        plt.imshow(enhanced_mask, cmap='gray')
        plt.axis('off')
        
        # 保存图像
        plt.suptitle(f"{dataset_name} - {img_file}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"visualization_results/{dataset_name}_{img_file.split('.')[0]}_enhancement.png", 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved visualization result for {dataset_name} - {img_file}")

def main():
    print("Starting visualization of enhancement effects...")
    
    # 处理每个数据集
    for dataset in datasets:
        visualize_dataset(dataset)
    
    print("Visualization completed! Results saved in 'visualization_results' directory")

if __name__ == "__main__":
    main() 