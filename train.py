import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# CBAM注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



# 多尺度边缘感知注意力模块
class MEAModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(MEAModule, self).__init__()
        self.in_channels = in_channels
        
        # 多尺度特征提取
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        
        # 边缘检测分支
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        

        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        
        # 添加1x1卷积用于边缘检测
        self.edge_reduce = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 多尺度特征提取
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        multi_scale_feat = feat1 + feat3 + feat5
        
        # 边缘特征提取
        edge_feat = self.edge_conv(x)
        edge_single = self.edge_reduce(edge_feat)
        edge_x = F.conv2d(edge_single, self.sobel_x, padding=1)
        edge_y = F.conv2d(edge_single, self.sobel_y, padding=1)
        edge_attention = torch.abs(edge_x) + torch.abs(edge_y)
        
        # 通道注意力
        avg_out = self.fc(self.avg_pool(multi_scale_feat).view(batch_size, -1))
        max_out = self.fc(self.max_pool(multi_scale_feat).view(batch_size, -1))
        channel_attention = self.sigmoid(avg_out + max_out).view(batch_size, self.in_channels, 1, 1)
        
        # 空间注意力
        avg_out = torch.mean(multi_scale_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(multi_scale_feat, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_feat))
        
        # 特征融合
        attended_feat = multi_scale_feat * channel_attention * spatial_attention
        edge_enhanced_feat = edge_feat * spatial_attention
        
        # 最终特征融合
        output = self.fusion_conv(torch.cat([attended_feat, edge_enhanced_feat], dim=1))
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mea = MEAModule(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.mea(x)
        return x


class MEAUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MEAUNet, self).__init__()
        
        # 编码器 - 使用MEA注意力机制
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        
        # 解码器 - 使用MEA注意力机制的跳跃连接
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)  # 512 + 512 = 1024 输入通道
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)   # 256 + 256 = 512 输入通道
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)   # 128 + 128 = 256 输入通道
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)    # 64 + 64 = 128 输入通道
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 编码路径 - 多尺度特征提取
        x1 = self.inc(x)        # 64 channels with MEA
        x2 = self.down1(x1)     # 128 channels with MEA
        x3 = self.down2(x2)     # 256 channels with MEA
        x4 = self.down3(x3)     # 512 channels with MEA
        x5 = self.down4(x4)     # 1024 channels with MEA
        
        # 解码路径 - 特征融合和边缘增强
        x = self.up1(x5)        # 512
        if x4.shape != x.shape:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x4, x], dim=1)  # 512 + 512 = 1024
        x = self.up_conv1(x)    # 512 with MEA
        
        x = self.up2(x)         # 256
        if x3.shape != x.shape:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)  # 256 + 256 = 512
        x = self.up_conv2(x)    # 256 with MEA
        
        x = self.up3(x)         # 128
        if x2.shape != x.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)  # 128 + 128 = 256
        x = self.up_conv3(x)    # 128 with MEA
        
        x = self.up4(x)         # 64
        if x1.shape != x.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)  # 64 + 64 = 128
        x = self.up_conv4(x)    # 64 with MEA
        
        x = self.outc(x)
        return x

# 数据集类
class RetinalVesselDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512), is_training=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.is_training = is_training
        
    
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        # 读取图像和掩码
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图像和掩码具有相同的尺寸
        if image is None:
            raise ValueError(f"Unable to read the image: {img_path}")
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")
        
        # 应用CLAHE增强
        image = self.clahe.apply(image)
            
        # 调整图像和掩码到相同的目标尺寸
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = (mask > 127).float()
        
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        if torch.isnan(pred).any():
            print("Warning: NaN detected in predictions during DiceLoss calculation.")
            return torch.tensor([1.0], device=pred.device, requires_grad=True)
        
        pred = torch.sigmoid(pred)
  
        if pred.shape != target.shape:
            if len(target.shape) == 3: 
                target = target.unsqueeze(1)  
                
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()

        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        if (pred_sum + target_sum) < 1e-6:
            return torch.tensor([0.0], device=pred.device, requires_grad=True)

        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        return torch.clamp(1 - dice, 0.0, 1.0)

# 组合损失：BCE + Dice
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
    def forward(self, pred, target):
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("Warning: NaN value detected in portfolio loss calculation.")
            return torch.tensor([1.0], device=pred.device, requires_grad=True)

        if pred.shape != target.shape:
  
            if len(target.shape) == 3:  
                target = target.unsqueeze(1)  

        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)

        if torch.isnan(dice_loss).any():
            return bce_loss
        if torch.isnan(bce_loss).any():
            return dice_loss

        combined_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        
        return combined_loss

# 计算Dice系数
def calculate_dice(pred, target):
    if np.isnan(pred).any() or np.isinf(pred).any():
        print("Warning: NaN or Inf values detected in predictions.")
        return float('nan')     
    pred = 1.0 / (1.0 + np.exp(-pred)) 
    pred = (pred > 0.5).astype(np.float32)
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum + target_sum < 1e-6:
        return 1.0  
    smooth = 1.0
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice

def get_transforms(target_size):
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(  
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-45, 45),
            p=0.5
        ),
        
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=1
            ),
            A.GridDistortion(p=1),
            A.OpticalDistortion(
                distort_limit=1,
                p=1
            ),
        ], p=0.3),
 
        A.OneOf([
            A.GaussNoise(
                per_channel=True,
                mean=0,
                std=10,
                p=1
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
        ], p=0.2),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=1
            ),
        ], p=0.3),
        

        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, dataset_name, patience=15):
    best_val_dice = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    patience_counter = 0
    

    scaler = GradScaler()
    

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  
        factor=0.5,  
        patience=5,  
        min_lr=1e-6
    )
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        valid_batches = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN/Inf loss detected in training batch; skipping this batch.")
                    continue
 
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # 计算训练Dice
                with torch.no_grad():
                    outputs_np = outputs.detach().cpu().numpy()
                    masks_np = masks.cpu().numpy()
   
                    if np.isnan(outputs_np).any():
                        print(f"Warning: Model output contains NaN values.")
                        continue
                        
                    batch_dice = calculate_dice(outputs_np, masks_np)

                    if not np.isnan(batch_dice):
                        train_dice += batch_dice
                        train_loss += loss.item()
                        valid_batches += 1
                        
                pbar.set_postfix({'loss': loss.item(), 'dice': batch_dice})
        
       
        if valid_batches > 0:
            train_loss = train_loss / valid_batches
            train_dice = train_dice / valid_batches
        else:
            print("Warning: No valid training batches in this round.")
            train_loss = float('nan')
            train_dice = 0.0
            
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        valid_val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
               
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue
                
                outputs_np = outputs.cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                
                if np.isnan(outputs_np).any():
                    continue
                    
                current_dice = calculate_dice(outputs_np, masks_np)
                
                if not np.isnan(current_dice):
                    val_loss += loss.item()
                    val_dice += current_dice
                    valid_val_batches += 1
        
        # 确保至少有一个有效批次
        if valid_val_batches > 0:
            val_loss = val_loss / valid_val_batches
            val_dice = val_dice / valid_val_batches
        else:
            print("Warning: There are no valid verification batches for this round.")
            val_loss = float('nan')
            val_dice = 0.0
            
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        

        previous_lr = optimizer.param_groups[0]['lr']
        if not np.isnan(val_dice):
            scheduler.step(val_dice) 
        
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f'Learning rate reduced: {previous_lr:.6f} -> {current_lr:.6f}')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 保存最佳模型和早停
        if not np.isnan(val_dice) and val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'best_val_dice': best_val_dice
            }, f'best_model_{dataset_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                print(f'Best validation Dice: {best_val_dice:.4f} at epoch {best_epoch+1}')
                break

# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            
   
            if torch.isnan(outputs).any():
                print("Warning: Model output NaN detected during evaluation; skipping this batch.")
                continue
            
        
            pred = torch.sigmoid(outputs).cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)
            masks_np = masks.numpy()
            
            batch_dice = calculate_dice(outputs.cpu().numpy(), masks_np)
            if np.isnan(batch_dice):
                print("Warning: Dice score detected as NaN; skipping this batch.")
                continue
                
            total_dice += batch_dice
            

            intersection = np.logical_and(pred_binary, masks_np)
            union = np.logical_or(pred_binary, masks_np)
            union_sum = np.sum(union)
            if union_sum < 1e-6:
                batch_iou = 1.0  
            else:
                batch_iou = np.sum(intersection) / union_sum
                
            total_iou += batch_iou
            
            # 收集预测结果用于计算其他指标
            valid_pred = pred_binary.flatten()
            valid_targets = masks_np.flatten()
            
            # 只添加非NaN值
            if not np.isnan(valid_pred).any() and not np.isnan(valid_targets).any():
                predictions.extend(valid_pred)
                targets.extend(valid_targets)
                num_batches += 1
    

    if num_batches == 0:
        print("Warning: No valid batches were evaluated; all metrics will be 0.")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'dice_score': 0.0,
            'iou_score': 0.0
        }
    
    # 计算平均Dice和IoU
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
    # 计算其他指标，添加异常处理
    try:
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        f1 = f1_score(targets, predictions)
    except Exception as e:
        print(f"Warning: Error calculating evaluation metrics：{str(e)}")
        accuracy = precision = recall = f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice_score': avg_dice,
        'iou_score': avg_iou
    }


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) - 1:
            if idx < self.cumulative_lengths[dataset_idx + 1]:
                break
            dataset_idx += 1

        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

def main():

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
 
    if device.type == 'cuda':
        print("Enable mixed-precision training.")
    else:
        print("Warning: Currently using CPU; mixed-precision training cannot be enabled.")
    
    # 训练参数
    batch_size = 4
    num_epochs = 200
    learning_rate = 0.0005  
    patience = 30
    target_size = (512, 512)
    
    # 获取数据增强转换
    train_transform, val_transform = get_transforms(target_size)
    
    no_aug_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    # 数据集列表
    datasets = ['DRIVE', 'CHASEDB1', 'HRF']
    
    # 对每个数据集进行训练
    for dataset_name in datasets:
        print(f'\nTraining on {dataset_name} dataset')
        
        try:
            # 创建原始训练数据集（不带增强）
            train_dataset_original = RetinalVesselDataset(
                f'preprocessed_data/{dataset_name}/train/images',
                f'preprocessed_data/{dataset_name}/train/masks',
                transform=no_aug_transform,
                target_size=target_size,
                is_training=True
            )
            
            # 创建增强训练数据集
            train_dataset_augmented = RetinalVesselDataset(
                f'preprocessed_data/{dataset_name}/train/images',
                f'preprocessed_data/{dataset_name}/train/masks',
                transform=train_transform,
                target_size=target_size,
                is_training=True
            )
            
            # 合并原始和增强的训练数据集
            combined_train_dataset = CombinedDataset([train_dataset_original, train_dataset_augmented])
            
            val_dataset = RetinalVesselDataset(
                f'preprocessed_data/{dataset_name}/test/images',
                f'preprocessed_data/{dataset_name}/test/masks',
                transform=val_transform,
                target_size=target_size,
                is_training=False
            )
            
            # 显示数据集信息
            print(f"Original training set size: {len(train_dataset_original)}")
            print(f"Enhance the size of the training set: {len(train_dataset_augmented)}")
            print(f"Training set size after merging: {len(combined_train_dataset)}")
            print(f"Validation set size: {len(val_dataset)}")
            
            train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 创建MEA-UNet模型
            model = MEAUNet().to(device)
            
            # 使用组合损失函数
            criterion = CombinedLoss(bce_weight=0.5)

            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=1e-5  
            )
            
            # 训练模型
            train_model(model, train_loader, val_loader, criterion, optimizer, 
                    num_epochs, device, dataset_name, patience=patience)
            
            # 评估模型
            metrics = evaluate_model(model, val_loader, device)
            print(f'\nEvaluation metrics for {dataset_name} using MEA-UNet:')
            for metric, value in metrics.items():
                print(f'{metric}: {value:.4f}')
                
        except Exception as e:
            print(f"Training {dataset_name} occurred error: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main() 
