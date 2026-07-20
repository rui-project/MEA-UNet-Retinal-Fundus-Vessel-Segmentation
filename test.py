import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast
from tqdm import tqdm
import glob
import re

from train import MEAUNet, RetinalVesselDataset, calculate_dice

# 获取原始图像和掩码的函数
def get_original_image_mask_path(dataset_name, img_name):

    base_name = os.path.basename(img_name)
    
    if dataset_name == "DRIVE":

        img_id = re.search(r'(\d+)', base_name).group(1)
        
      
        if int(img_id) <= 20:  
            orig_img_path = f"DRIVE/test/images/{img_id}_test.tif"
            mask_path = f"DRIVE/test/1st_manual/{img_id}_manual1.gif"
        else:  
            orig_img_path = f"DRIVE/training/images/{img_id}_training.tif"
            mask_path = f"DRIVE/training/1st_manual/{img_id}_manual1.gif"
            
    elif dataset_name == "CHASEDB1":

        img_id = re.search(r'(\d+[LR])', base_name).group(1)
        orig_img_path = f"CHASEDB1/Image_{img_id}.jpg"
        mask_path = f"CHASEDB1/Image_{img_id}_1stHO.png"
        
    elif dataset_name == "HRF":
 
        img_parts = re.search(r'(\d+)_([hgd]r)', base_name)
        if img_parts:
            img_id = img_parts.group(1)
            img_type = img_parts.group(2)
            orig_img_path = f"HRF/images/{img_id}_{img_type}.jpg"
            mask_path = f"HRF/mask/{img_id}_{img_type}_mask.tif"
        else:
           
            return None, None
    else:
        return None, None
        
    return orig_img_path, mask_path

# 评估函数
def evaluate_model(model, test_loader, device, save_dir=None, save_predictions=False, dataset_name=None):
    model.eval()
    predictions = []
    targets = []
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluate the test set")):
            images = images.to(device)
            outputs = model(images)
            pred = torch.sigmoid(outputs).cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)
            masks_np = masks.numpy()
            
            # 计算每个批次的Dice系数
            batch_dice = calculate_dice(outputs.cpu().numpy(), masks_np)
            total_dice += batch_dice
            
            # 计算IoU 
            intersection = np.logical_and(pred_binary, masks_np)
            union = np.logical_or(pred_binary, masks_np)
            batch_iou = np.sum(intersection) / (np.sum(union) + 1e-6)  # 添加平滑项避免除零
            total_iou += batch_iou
            
    
            predictions.extend(pred_binary.flatten())
            targets.extend(masks_np.flatten())
            num_batches += 1
            
            # 保存预测结果
            if save_predictions and save_dir is not None and dataset_name is not None:
                test_images_dir = os.path.join('preprocessed_data', dataset_name, 'test', 'images')
                image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png')])
                
                for j in range(images.size(0)):
                    img_idx = i * test_loader.batch_size + j
                    if img_idx < len(image_files):
                        img_name = image_files[img_idx]
                        
              
                        orig_img_path, mask_path = get_original_image_mask_path(dataset_name, img_name)
                        
                        if orig_img_path and os.path.exists(orig_img_path):
                         
                            if dataset_name == "DRIVE":
                                orig_image = cv2.imread(orig_img_path)
                                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                            else:
                                orig_image = cv2.imread(orig_img_path)
                                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                                
                        
                            if mask_path and os.path.exists(mask_path):
                                if mask_path.endswith('.gif'):
                                 
                                    import imageio
                                    true_mask = imageio.imread(mask_path)
                                else:
                                    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            else:
                              
                                true_mask = masks_np[j][0] * 255
                                true_mask = true_mask.astype(np.uint8)
                        else:
                          
                            orig_image = images[j].cpu().numpy()[0]
                            orig_image = (orig_image * 0.5 + 0.5) * 255
                            orig_image = orig_image.astype(np.uint8)
                            
                         
                            true_mask = masks_np[j][0] * 255
                            true_mask = true_mask.astype(np.uint8)
                        
               
                        pred_mask = pred_binary[j][0] * 255
                        pred_mask = pred_mask.astype(np.uint8)
                        
                 
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        axes[0].imshow(orig_image)
                        axes[0].set_title('Original Image', fontsize=12)
                        axes[0].axis('off')
                        
                        axes[1].imshow(true_mask, cmap='gray')
                        axes[1].set_title('Ground Truth', fontsize=12)
                        axes[1].axis('off')
                        
                        axes[2].imshow(pred_mask, cmap='gray')
                        axes[2].set_title('Prediction', fontsize=12)
                        axes[2].axis('off')
                        
                        plt.tight_layout(pad=1.5)
                        plt.savefig(os.path.join(save_dir, f'result_{i}_{j}.png'), 
                                  bbox_inches='tight', dpi=150)
                        plt.close()
    
    # 计算平均Dice和IoU
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
   
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice_score': avg_dice,
        'iou_score': avg_iou
    }

def main():
   
    data_dir = 'preprocessed_data'  # 数据集根目录
    batch_size = 4  # 批次大小
    target_size = (512, 512)  # 图像目标尺寸 原为576
    save_predictions = True  # 是否保存预测结果
    output_dir = 'test_results'  # 输出结果保存目录
    
    # 要评估的数据集和对应的模型
    datasets_models = {
        'DRIVE': 'best_model_DRIVE.pth',
        'CHASEDB1': 'best_model_CHASEDB1.pth',
        'HRF': 'best_model_HRF.pth'
    }
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Equipment Used: {device}')
    
    # 创建输出目录
    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试转换
    test_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    # 循环测试每个数据集及其对应的模型
    for dataset_name, model_path in datasets_models.items():
        print(f'\nloading {dataset_name} model and testing')
        
        try:
            # 加载模型
            model = MEAUNet().to(device)
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                print("Model weights loaded successfully.")
            
            # 创建测试数据加载器
            test_dataset = RetinalVesselDataset(
                os.path.join(data_dir, dataset_name, 'test', 'images'),
                os.path.join(data_dir, dataset_name, 'test', 'masks'),
                transform=test_transform,
                target_size=target_size,
                is_training=False
            )
            
            test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                   shuffle=False, num_workers=0)
            
            print(f"Test set size: {len(test_dataset)}")
            
            # 创建保存目录
            save_dir = None
            if save_predictions:
                save_dir = os.path.join(output_dir, dataset_name)
                os.makedirs(save_dir, exist_ok=True)
            
            # 评估模型
            metrics = evaluate_model(model, test_loader, device, save_dir, save_predictions, dataset_name)
            
            print(f'\n{dataset_name} Dataset evaluation metrics:')
            for metric, value in metrics.items():
                print(f'{metric}: {value:.4f}')
            
            # 保存指标结果到文件
            if save_predictions:
                with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
                    for metric, value in metrics.items():
                        f.write(f'{metric}: {value:.4f}\n')
                
                print(f"Metrics saved to {os.path.join(save_dir, 'metrics.txt')}")
            
        except Exception as e:
            print(f"Evaluating {dataset_name} occurred error: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main() 