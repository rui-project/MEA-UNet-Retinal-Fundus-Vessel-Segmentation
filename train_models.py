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





# =============================================================
# 标准U-Net模型，没有任何注意力机制
# =============================================================
class StandardDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # 编码器
        self.inc = StandardDoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(512, 1024)
        )
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = StandardDoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = StandardDoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = StandardDoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = StandardDoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径
        x = self.up1(x5)
        if x4.shape != x.shape:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        if x3.shape != x.shape:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        if x2.shape != x.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        if x1.shape != x.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        x = self.outc(x)
        return x

# =============================================================
# Attention U-Net 实现
# =============================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # 下采样的gating信号
        g1 = self.W_g(g)
        # 上采样的l信号
        x1 = self.W_x(x)
        # 元素相加后通过ReLU
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 注意力权重乘以输入特征
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()
        
        # 编码器
        self.inc = StandardDoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            StandardDoubleConv(512, 1024)
        )
        
        # 注意力门
        self.attention1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attention2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attention4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = StandardDoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = StandardDoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = StandardDoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = StandardDoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径
        x = self.up1(x5)
        if x4.shape != x.shape:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        # 应用注意力门
        x4_att = self.attention1(g=x, x=x4)
        x = torch.cat([x4_att, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        if x3.shape != x.shape:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        # 应用注意力门
        x3_att = self.attention2(g=x, x=x3)
        x = torch.cat([x3_att, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        if x2.shape != x.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        # 应用注意力门
        x2_att = self.attention3(g=x, x=x2)
        x = torch.cat([x2_att, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        if x1.shape != x.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        # 应用注意力门
        x1_att = self.attention4(g=x, x=x1)
        x = torch.cat([x1_att, x], dim=1)
        x = self.up_conv4(x)
        
        x = self.outc(x)
        return x

# =============================================================
# CBAM-UNet实现 (使用先前定义的CBAM模块)
# =============================================================
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class CBAMDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAMDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x

class CBAMUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(CBAMUNet, self).__init__()
        
        # 编码器 - 使用CBAM注意力机制
        self.inc = CBAMDoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            CBAMDoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            CBAMDoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            CBAMDoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            CBAMDoubleConv(512, 1024)
        )
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = CBAMDoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = CBAMDoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = CBAMDoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = CBAMDoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径
        x = self.up1(x5)
        if x4.shape != x.shape:
            x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x4, x], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        if x3.shape != x.shape:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        if x2.shape != x.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        if x1.shape != x.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv4(x)
        
        x = self.outc(x)
        return x

# =============================================================
# UNet++ (嵌套U-Net)实现 - 替换DeepLabV3Plus
# =============================================================
class NestedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NestedDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# =============================================================
# 修正GaussNoise参数警告
# =============================================================
def get_transforms(target_size):
    train_transform = A.Compose([
        # 基础几何变换
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(  # 替换ShiftScaleRotate
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-45, 45),
            p=0.5
        ),
        
        # 医学图像特定增强
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
        
        # 图像质量和对比度增强
        A.OneOf([
            A.GaussNoise(
                var_limit=(10.0, 50.0),  # 修复参数警告
                p=1
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
        ], p=0.2),
        
        # 亮度对比度增强
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
        
        # 标准化和转换
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

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
        
        # 创建CLAHE对象
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
        
        # 确保mask为二值图像并添加通道维度
        mask = (mask > 127).float()
        
        # 添加通道维度到mask以匹配模型输出 [H,W] -> [1,H,W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

# 修改Dice损失以适应logits输出
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # 添加sigmoid激活
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

# 组合损失：BCE + Dice
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()  # 改用BCEWithLogitsLoss
        
    def forward(self, pred, target):
        # 确保pred和target形状匹配
        if pred.shape != target.shape:
            # 如果形状不匹配，调整target的形状
            if len(target.shape) == 3:  # [B,H,W]
                target = target.unsqueeze(1)  # [B,1,H,W]
        
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# 修改计算Dice系数的函数以适应logits输出
def calculate_dice(pred, target):
    pred = torch.sigmoid(torch.from_numpy(pred)).numpy()  # 添加sigmoid激活
    pred = pred > 0.5
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + 1.0) / (pred.sum() + target.sum() + 1.0)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, dataset_name, patience=15):
    best_val_dice = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    patience_counter = 0
    
    # 仅在CUDA上启用混合精度。CPU autocast默认使用bfloat16，转NumPy时会报错。
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # 创建学习率调度器
    # 1. 余弦退火学习率调度器，每10轮重启一次
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 第一次重启的轮数
        T_mult=2,  # 每次重启后周期长度的倍数
        eta_min=1e-6  # 最小学习率
    )
    
    # 2. 基于验证指标的学习率调度器
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控验证Dice分数
        factor=0.5,  # 学习率减半
        patience=5,  # 5轮无改善则降低学习率
        min_lr=1e-6
    )
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                # 使用梯度缩放器进行反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 计算训练Dice
                batch_dice = calculate_dice(outputs.detach().float().cpu().numpy(),
                                         masks.float().cpu().numpy())
                train_dice += batch_dice
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'dice': batch_dice})
        
        train_loss = train_loss / len(train_loader)
        train_dice = train_dice / len(train_loader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_dice += calculate_dice(outputs.float().cpu().numpy(),
                                            masks.float().cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_dice = val_dice / len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # 更新学习率
        cosine_scheduler.step()  # 更新余弦退火调度器
        plateau_scheduler.step(val_dice)  # 更新ReduceLROnPlateau调度器
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 保存最佳模型和早停
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': {
                    'cosine': cosine_scheduler.state_dict(),
                    'plateau': plateau_scheduler.state_dict()
                },
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
            
            # 处理UNet++模型在训练模式返回多输出的情况
            if isinstance(outputs, list):
                outputs = outputs[-1]  # 使用最后一个输出
                
            pred = torch.sigmoid(outputs.float()).cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)
            masks_np = masks.float().numpy()
            
            # 计算每个批次的Dice系数
            batch_dice = calculate_dice(outputs.float().cpu().numpy(), masks_np)
            total_dice += batch_dice
            
            # 计算IoU (Intersection over Union)
            intersection = np.logical_and(pred_binary, masks_np)
            union = np.logical_or(pred_binary, masks_np)
            batch_iou = np.sum(intersection) / (np.sum(union) + 1e-6)  # 添加平滑项避免除零
            total_iou += batch_iou
            
            # 收集预测结果用于计算其他指标
            predictions.extend(pred_binary.flatten())
            targets.extend(masks_np.flatten())
            num_batches += 1
    
    # 计算平均Dice和IoU
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
    # 计算其他指标
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice_score': avg_dice,
        'iou_score': avg_iou
    }

# 数据集合并类
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        # 确定idx属于哪个数据集
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) - 1:
            if idx < self.cumulative_lengths[dataset_idx + 1]:
                break
            dataset_idx += 1
        
        # 计算在原始数据集中的索引
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

# 在现有模型之后添加
# =============================================================
# SegFormer 实现
# =============================================================
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=7,
            stride=4,
            padding=3
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,H*W,C
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        return x

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)  # B,heads,H*W,C/heads
        
        # 空间降采样，降低计算复杂度
        x_sr = self.sr(x)
        _, _, H_sr, W_sr = x_sr.shape
        kv = self.kv(x_sr).reshape(B, 2, self.num_heads, C // self.num_heads, H_sr * W_sr).permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]  # B,heads,H*W,C/heads

        # 高效注意力机制
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class MixFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, mlp_ratio=mlp_ratio)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 应用注意力
        norm_x = x.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        norm_x = self.norm1(norm_x).permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        x = x + self.attn(norm_x)
        
        # 应用MLP
        norm_x = x.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        norm_x = self.norm2(norm_x).permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        x = x + self.mlp(norm_x)
        return x

class MiT(nn.Module):
    def __init__(self, in_channels=1, dims=[32, 64, 128, 256]):
        super().__init__()
        self.stages = nn.ModuleList()
        
        # 第一阶段
        self.stages.append(nn.Sequential(
            PatchEmbed(in_channels, dims[0]),
            TransformerBlock(dims[0], num_heads=1, sr_ratio=8)
        ))
        
        # 其余阶段
        for i in range(1, len(dims)):
            self.stages.append(nn.Sequential(
                nn.Conv2d(dims[i-1], dims[i], kernel_size=3, stride=2, padding=1),
                TransformerBlock(dims[i], num_heads=2**(i-1), sr_ratio=4//min(i, 2)),
                TransformerBlock(dims[i], num_heads=2**(i-1), sr_ratio=4//min(i, 2))
            ))

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class MLPDecoder(nn.Module):
    def __init__(self, dims, embed_dim=256):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        self.scales = []
        
        for i, dim in enumerate(dims):
            scale = 2 ** i  # 特征图尺度相对于原始图像
            self.scales.append(scale)
            self.linear_layers.append(nn.Sequential(
                nn.Conv2d(dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ))
        
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(dims), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, input_size):
        outs = []
        for i, feature in enumerate(features):
            x = self.linear_layers[i](feature)
            if x.shape[2:] != input_size:
                x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
            outs.append(x)
        
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        return out



def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 检查是否支持混合精度训练
    if device.type == 'cuda':
        print("Enable mixed-precision training.")
    else:
        print("Warning: Currently using CPU; mixed-precision training cannot be enabled.")
    
    # 训练参数
    batch_size = 4
    num_epochs = 10  # 减少训练轮数用于模型比较
    learning_rate = 0.001
    patience = 15  # 提前停止轮数
    target_size = (512, 512)
    
    # 获取数据增强转换
    train_transform, val_transform = get_transforms(target_size)
    
    # 创建无增强的转换（仅包含标准化和转换为张量）
    no_aug_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    # 数据集列表
    datasets = ['DRIVE', 'CHASEDB1', 'HRF']
    

    models_config = [
        {"name": "UNet", "class": UNet, "criterion": CombinedLoss(bce_weight=0.5)},
        {"name": "AttentionUNet", "class": AttentionUNet, "criterion": CombinedLoss(bce_weight=0.5)},
        {"name": "CBAMUNet", "class": CBAMUNet, "criterion": CombinedLoss(bce_weight=0.5)},

    ]
    
    # 创建结果收集字典
    results = {dataset: {model_config["name"]: {} for model_config in models_config} for dataset in datasets}
    
    # 对每个数据集进行训练
    for dataset_name in datasets:
        print(f'\n====== Training dataset：{dataset_name} ======')
        
        try:
            # 创建增强训练数据集
            train_dataset = RetinalVesselDataset(
                f'preprocessed_data/{dataset_name}/train/images',
                f'preprocessed_data/{dataset_name}/train/masks',
                transform=train_transform,
                target_size=target_size,
                is_training=True
            )
            
            val_dataset = RetinalVesselDataset(
                f'preprocessed_data/{dataset_name}/test/images',
                f'preprocessed_data/{dataset_name}/test/masks',
                transform=val_transform,
                target_size=target_size,
                is_training=False
            )
            
            # 显示数据集信息
            print(f"Training set size: {len(train_dataset)}")
            print(f"Validation set size: {len(val_dataset)}")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 对每个模型进行训练和评估
            for model_config in models_config:
                model_name = model_config["name"]
                model_class = model_config["class"]
                criterion = model_config["criterion"]
                
                print(f'\n------ Train the model：{model_name} ------')
                
                # 创建模型实例
                model = model_class().to(device)
                
                # 创建优化器
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                # 训练模型
                train_model(model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs, device, f"{dataset_name}_{model_name}", patience=patience)
                
                # 评估模型
                metrics = evaluate_model(model, val_loader, device)
                
                # 存储评估结果
                results[dataset_name][model_name] = metrics
                
                print(f'\nEvaluation metrics for {dataset_name} using {model_name}:')
                for metric, value in metrics.items():
                    print(f'{metric}: {value:.4f}')
            
            # 打印该数据集上的模型比较表
            print(f'\n====== {dataset_name} Comparison of Dataset Model Performance ======')
            metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'dice_score', 'iou_score']
            
            # 打印表头
            header = "Model".ljust(15)
            for metric in metrics_list:
                header += metric.ljust(12)
            print(header)
            
            # 打印每个模型的结果
            for model_name in [config["name"] for config in models_config]:
                row = model_name.ljust(15)
                for metric in metrics_list:
                    row += f"{results[dataset_name][model_name][metric]:.4f}".ljust(12)
                print(row)
                
        except Exception as e:
            print(f"Training {dataset_name} occurred error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印所有数据集和模型的性能比较汇总
    print("\n====== Summary of Model Performance Comparisons Across All Datasets ======")
    print("Use the average Dice score as the primary metric")
    
    # 打印表头
    header = "Model".ljust(15)
    for dataset in datasets:
        header += dataset.ljust(10)
    header += "Average".ljust(10)
    print(header)
    
    # 打印每个模型在所有数据集上的平均Dice分数
    for model_name in [config["name"] for config in models_config]:
        row = model_name.ljust(15)
        model_avg_dice = 0
        valid_datasets = 0
        
        for dataset in datasets:
            if dataset in results and model_name in results[dataset]:
                dice_score = results[dataset][model_name].get('dice_score', 0)
                row += f"{dice_score:.4f}".ljust(10)
                model_avg_dice += dice_score
                valid_datasets += 1
            else:
                row += "N/A".ljust(10)
        
        # 计算该模型在所有数据集上的平均Dice分数
        if valid_datasets > 0:
            avg_dice = model_avg_dice / valid_datasets
            row += f"{avg_dice:.4f}".ljust(10)
        else:
            row += "N/A".ljust(10)
            
        print(row)

if __name__ == '__main__':
    main() 
