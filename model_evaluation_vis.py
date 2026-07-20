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
import imageio.v2 as imageio
from matplotlib.colors import ListedColormap


from train import MEAUNet, RetinalVesselDataset, calculate_dice


from train_models import UNet, AttentionUNet, CBAMUNet


def read_image_rgb(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read the image: {image_path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask_gray(mask_path):
    if mask_path.lower().endswith('.gif'):
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read mask: {mask_path}")
    return mask


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
def evaluate_models(models_dict, test_loader, device, save_dir=None, dataset_name=None):

    results = {model_name: {
        'accuracy': 0, 'precision': 0, 'recall': 0, 
        'f1_score': 0, 'dice_score': 0, 'iou_score': 0,
        'predictions': [], 'targets': []
    } for model_name in models_dict.keys()}
    
    test_images_dir = os.path.join('preprocessed_data', dataset_name, 'test', 'images')
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.png')])
    
    for i, (images, masks) in enumerate(tqdm(test_loader, desc=f"Evaluate{dataset_name}Test set")):
        images = images.to(device)
        masks_np = masks.numpy()
        

        predictions = {}
        
        # 按顺序评估每个模型
        for model_name, model in models_dict.items():
            model.eval()
            
            with torch.no_grad():
                outputs = model(images)
                pred = torch.sigmoid(outputs).cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.float32)
                
                # 计算每个批次的Dice系数
                batch_dice = calculate_dice(outputs.cpu().numpy(), masks_np)
                results[model_name]['dice_score'] += batch_dice
                
                # 计算IoU
                intersection = np.logical_and(pred_binary, masks_np)
                union = np.logical_or(pred_binary, masks_np)
                batch_iou = np.sum(intersection) / (np.sum(union) + 1e-6)
                results[model_name]['iou_score'] += batch_iou
                
                # 收集预测结果
                results[model_name]['predictions'].extend(pred_binary.flatten())
                results[model_name]['targets'].extend(masks_np.flatten())
                
                # 保存预测以供可视化
                predictions[model_name] = pred_binary
        

        if save_dir is not None and dataset_name is not None:
            for j in range(images.size(0)):
                img_idx = i * test_loader.batch_size + j
                if img_idx < len(image_files):
                    img_name = image_files[img_idx]
                
                    orig_img_path, mask_path = get_original_image_mask_path(dataset_name, img_name)
                    
                    if orig_img_path and os.path.exists(orig_img_path):
                        orig_image = read_image_rgb(orig_img_path)
                            
                      
                        if mask_path and os.path.exists(mask_path):
                            true_mask = read_mask_gray(mask_path)
                        else:
                           
                            true_mask = masks_np[j][0] * 255
                            true_mask = true_mask.astype(np.uint8)
                    else:
                       
                        orig_image = images[j].cpu().numpy()[0]
                        orig_image = (orig_image * 0.5 + 0.5) * 255
                        orig_image = orig_image.astype(np.uint8)
                        
                    
                        true_mask = masks_np[j][0] * 255
                        true_mask = true_mask.astype(np.uint8)
                    
                 
                    n_cols = 6  
                    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
                    
                    # 显示原始图像
                    axes[0].imshow(orig_image)
                    axes[0].set_title('Original Image', fontsize=18)
                    axes[0].axis('off')
                    
                    # 显示真实掩码
                    axes[1].imshow(true_mask, cmap='gray')
                    axes[1].set_title('Ground Truth', fontsize=18)
                    axes[1].axis('off')
                    
                    # 所有模型名称列表，以固定顺序显示
                    all_model_names = ['MEAUNet', 'UNet', 'AttentionUNet', 'CBAMUNet']
                    
                    # 显示每个模型的预测结果，按固定顺序
                    for k, model_name in enumerate(all_model_names):
                        if model_name in predictions:
                            pred_mask = predictions[model_name][j][0] * 255
                            pred_mask = pred_mask.astype(np.uint8)
                            axes[k+2].imshow(pred_mask, cmap='gray')
                            axes[k+2].set_title(f'{model_name}', fontsize=18)
                        else:
                           
                            axes[k+2].imshow(np.zeros_like(true_mask), cmap='gray')
                            axes[k+2].set_title(f'{model_name}\n(Not Available)', fontsize=18)
                        axes[k+2].axis('off')
                    
                    plt.tight_layout(pad=1.5)
                    plt.savefig(os.path.join(save_dir, f'比较_{img_idx}.png'), 
                              bbox_inches='tight', dpi=150)
                    plt.close()
    
    # 计算平均指标
    num_batches = len(test_loader)
    for model_name in models_dict.keys():
        # 计算平均Dice和IoU
        results[model_name]['dice_score'] /= num_batches
        results[model_name]['iou_score'] /= num_batches
        
        # 计算其他指标
        results[model_name]['accuracy'] = accuracy_score(
            results[model_name]['targets'], 
            results[model_name]['predictions']
        )
        results[model_name]['precision'] = precision_score(
            results[model_name]['targets'], 
            results[model_name]['predictions'],
            zero_division=0
        )
        results[model_name]['recall'] = recall_score(
            results[model_name]['targets'], 
            results[model_name]['predictions'],
            zero_division=0
        )
        results[model_name]['f1_score'] = f1_score(
            results[model_name]['targets'], 
            results[model_name]['predictions'],
            zero_division=0
        )
        
     
        del results[model_name]['predictions']
        del results[model_name]['targets']
    
    return results

def main():

    try:
        from torch.serialization import add_safe_globals
        add_safe_globals(['numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar'])
        print("已添加numpy.scalar类型到PyTorch安全全局列表")
    except Exception as e:
        print(f"添加安全全局列表时出错: {e}")
    
    # 设置参数
    data_dir = 'preprocessed_data'  # 数据集根目录
    batch_size = 4  # 批次大小
    target_size = (512, 512)  # 图像目标尺寸
    output_dir = 'model_comparison_results'  # 输出结果保存目录
    
    # 要评估的数据集
    datasets = ['DRIVE', 'CHASEDB1', 'HRF']
    
    # 要比较的模型和对应的权重文件
    models_config = {
        'MEAUNet': {
            'DRIVE': 'best_model_DRIVE.pth',
            'CHASEDB1': 'best_model_CHASEDB1.pth',
            'HRF': 'best_model_HRF.pth'
        },
        'UNet': {
            'DRIVE': 'best_model_DRIVE_UNet.pth',
            'CHASEDB1': 'best_model_CHASEDB1_UNet.pth',
            'HRF': 'best_model_HRF_UNet.pth'
        },
        'AttentionUNet': {
            'DRIVE': 'best_model_DRIVE_AttentionUNet.pth',
            'CHASEDB1': 'best_model_CHASEDB1_AttentionUNet.pth',
            'HRF': 'best_model_HRF_AttentionUNet.pth'
        },
        'CBAMUNet': {
            'DRIVE': 'best_model_DRIVE_CBAMUNet.pth',
            'CHASEDB1': 'best_model_CHASEDB1_CBAMUNet.pth',
            'HRF': 'best_model_HRF_CBAMUNet.pth'
        }
    }
    
    # 模型类映射
    model_classes = {
        'MEAUNet': MEAUNet,
        'UNet': UNet,
        'AttentionUNet': AttentionUNet,
        'CBAMUNet': CBAMUNet
    }
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Equipment Used: {device}')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试转换
    test_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2()
    ])
    
    # 循环测试每个数据集
    all_results = {}  # 存储所有数据集的结果
    
    for dataset_name in datasets:
        print(f'\nStart evaluating the dataset: {dataset_name}')
        
        # 创建数据集保存目录
        dataset_save_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        # 创建测试数据加载器
        try:
            test_dataset = RetinalVesselDataset(
                os.path.join(data_dir, dataset_name, 'test', 'images'),
                os.path.join(data_dir, dataset_name, 'test', 'masks'),
                transform=test_transform,
                target_size=target_size,
                is_training=False
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0
            )
            
            print(f"Test set size: {len(test_dataset)}")
            
            # 加载所有可用的模型
            loaded_models = {}
            
            for model_name, model_class in model_classes.items():
                model_path = models_config[model_name].get(dataset_name)
                
                if model_path and os.path.exists(model_path):
                    print(f"Loading model: {model_name} from file: {model_path}")
                    
                    model = model_class().to(device)
                    
                    try:
                        # 加载模型权重 - 修复PyTorch 2.6的weights_only默认值变更问题
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        
                        loaded_models[model_name] = model
                        print(f"successfully loaded {model_name} model")
                    except Exception as e:
                        print(f"loading {model_name} Model failure: {e}")
                else:
                    print(f"Can NOT find {model_name} weight file: {model_path}")
            
            # 创建字典，将每个缺失的模型替换为空模型进行可视化（但不用于评估）
            for model_name in model_classes.keys():
                if model_name not in loaded_models:
                    print(f"Attention: {model_name} The model could not be loaded and will appear blank in the visualization.")
            
            # 如果没有成功加载任何模型，跳过当前数据集
            if not loaded_models:
                print(f"Warning: Failed to load any models，skip {dataset_name} evaluate dataset.")
                continue
            
            # 评估所有加载的模型
            results = evaluate_models(
                loaded_models, 
                test_loader, 
                device, 
                dataset_save_dir, 
                dataset_name
            )
            
            # 打印评估结果
            print(f"\n{dataset_name} Dataset evaluation results:")
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'dice_score', 'iou_score']
            
            # 打印表头
            header = "Model".ljust(15)
            for metric in metrics:
                header += metric.ljust(12)
            print(header)
            
            # 打印每个模型的结果
            for model_name, metrics_dict in results.items():
                row = model_name.ljust(15)
                for metric in metrics:
                    row += f"{metrics_dict[metric]:.4f}".ljust(12)
                print(row)
            
            # 保存评估结果到CSV文件
            with open(os.path.join(dataset_save_dir, 'results.csv'), 'w') as f:
                # 写入表头
                f.write('Model,' + ','.join(metrics) + '\n')
                
                # 写入每个模型的结果
                for model_name, metrics_dict in results.items():
                    f.write(f"{model_name}")
                    for metric in metrics:
                        f.write(f",{metrics_dict[metric]:.4f}")
                    f.write('\n')
            
            print(f"The evaluation results have been saved to {os.path.join(dataset_save_dir, 'results.csv')}")
            
            # 创建条形图来比较Dice分数
            plt.figure(figsize=(10, 6))
            model_names = list(results.keys())
            dice_scores = [results[model_name]['dice_score'] for model_name in model_names]
            iou_scores = [results[model_name]['iou_score'] for model_name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            plt.bar(x - width/2, dice_scores, width, label='Dice Score')
            plt.bar(x + width/2, iou_scores, width, label='IoU Score')
            
            plt.xlabel('Models', fontsize=16)
            plt.ylabel('Scores', fontsize=16)
            plt.title(f'{dataset_name} Dataset Performance Comparison', fontsize=18)
            plt.xticks(x, model_names, fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylim(0, 1.0)
            plt.legend(fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_save_dir, 'performance_comparison.png'), dpi=300)
            plt.close()
            
            # 保存这个数据集的结果到全局结果字典
            all_results[dataset_name] = results
            
        except Exception as e:
            print(f"Evaluating {dataset_name} occurred error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建所有数据集和模型的汇总表
    print("\n=========================================")
    print("Create a summary table of evaluation metrics for all datasets and models")
    print("=========================================")
    
    # 找出所有评估过的模型
    all_model_names = set()
    for dataset_results in all_results.values():
        all_model_names.update(dataset_results.keys())
    
    # 按照指定顺序排列模型
    model_order = ['MEAUNet', 'UNet', 'AttentionUNet', 'CBAMUNet']
    all_model_names = [model for model in model_order if model in all_model_names]
    
    # 找出所有的指标
    metrics = ['dice_score', 'iou_score', 'accuracy', 'precision', 'recall', 'f1_score']
    metrics_display = ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1']
    
    # 准备汇总数据
    summary_data = {}
    
    # 为每个指标创建一个汇总表并打印
    for metric_idx, metric in enumerate(metrics):
        metric_display = metrics_display[metric_idx]
        print(f"\n{metric_display} Score Summary:")
        
        # 打印表头
        header = "Model".ljust(15)
        for dataset_name in datasets:
            if dataset_name in all_results:
                header += dataset_name.ljust(12)
        header += "Average".ljust(12)
        print(header)
        
        # 创建该指标的CSV数据
        csv_rows = [['Model'] + [d for d in datasets if d in all_results] + ['Average']]
        
        # 打印每个模型的结果
        for model_name in all_model_names:
            row = model_name.ljust(15)
            model_values = []
            
            for dataset_name in datasets:
                if dataset_name in all_results and model_name in all_results[dataset_name]:
                    value = all_results[dataset_name][model_name].get(metric, 0)
                    row += f"{value:.4f}".ljust(12)
                    model_values.append(value)
                else:
                    row += "N/A".ljust(12)
            
            # 计算该模型在所有数据集上的平均指标
            if model_values:
                avg_value = sum(model_values) / len(model_values)
                row += f"{avg_value:.4f}".ljust(12)
                
                # 添加到CSV数据
                csv_row = [model_name]
                for dataset_name in datasets:
                    if dataset_name in all_results and model_name in all_results[dataset_name]:
                        csv_row.append(f"{all_results[dataset_name][model_name].get(metric, 0):.4f}")
                    else:
                        csv_row.append("N/A")
                csv_row.append(f"{avg_value:.4f}")
                csv_rows.append(csv_row)
            else:
                row += "N/A".ljust(12)
                
            print(row)
        
        # 保存该指标的CSV文件
        summary_data[metric] = csv_rows
    
    # 保存所有指标的汇总CSV
    os.makedirs('summary', exist_ok=True)
    
    # 为每个指标保存一个CSV文件
    for metric, csv_data in summary_data.items():
        with open(os.path.join('summary', f'{metric}_summary.csv'), 'w') as f:
            for row in csv_data:
                f.write(','.join([str(cell) for cell in row]) + '\n')
    
    print("\nAll summarized data has been saved to the 'summary' folder.")
    
    
    plt.figure(figsize=(20, 15))
    

    dataset_colors = {
        'DRIVE': '#3498db',       # 蓝色
        'CHASEDB1': '#2ecc71',    # 绿色
        'HRF': '#e74c3c',         # 红色
        'Average': '#9b59b6'      # 紫色
    }
    

    for subplot_idx, metric in enumerate(metrics):
        metric_display = metrics_display[subplot_idx]

        valid_datasets = [d for d in datasets if d in all_results]

        ax = plt.subplot(2, 3, subplot_idx + 1)

        x = np.arange(len(all_model_names))
        width = 0.8 / (len(valid_datasets) + 1)  

        for i, dataset_name in enumerate(valid_datasets):
            values = []
            for model_name in all_model_names:
                if model_name in all_results[dataset_name]:
                    values.append(all_results[dataset_name][model_name].get(metric, 0))
                else:
                    values.append(0)
            
            ax.bar(x - 0.4 + i * width, values, width, label=dataset_name if subplot_idx == 0 else "", 
                 color=dataset_colors[dataset_name])
        
        # 绘制平均值柱状图
        avg_values = []
        for model_name in all_model_names:
            model_values = []
            for dataset_name in valid_datasets:
                if model_name in all_results[dataset_name]:
                    model_values.append(all_results[dataset_name][model_name].get(metric, 0))
            
            if model_values:
                avg_values.append(sum(model_values) / len(model_values))
            else:
                avg_values.append(0)
        
        ax.bar(x - 0.4 + len(valid_datasets) * width, avg_values, width, 
             label="Average" if subplot_idx == 0 else "", color=dataset_colors['Average'])
        
        # 设置图表属性
        ax.set_xlabel('Models', fontsize=16)
        ax.set_ylabel(f'{metric_display} Score', fontsize=16)
        ax.set_title(f'{metric_display} Score Comparison', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(all_model_names, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if subplot_idx == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=14, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 为条形图添加数值标签 - 仅显示平均值
        for i, v in enumerate(avg_values):
            ax.text(x[i] - 0.4 + len(valid_datasets) * width, v + 0.02, f"{v:.3f}", 
                  ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join('summary', 'all_metrics_comparison.png'), dpi=300)
    plt.close()
    
    # 为每个单独指标创建一个更详细的图表
    for metric_idx, metric in enumerate(metrics):
        metric_display = metrics_display[metric_idx]
        
        plt.figure(figsize=(12, 8))
        ax = plt.axes()
        
        # 获取有效的数据集
        valid_datasets = [d for d in datasets if d in all_results]
        
        # 计算x轴位置
        x = np.arange(len(all_model_names))
        width = 0.8 / (len(valid_datasets) + 1)  # +1 为平均值腾出空间
        
        # 为每个数据集绘制柱状图
        for i, dataset_name in enumerate(valid_datasets):
            values = []
            for model_name in all_model_names:
                if model_name in all_results[dataset_name]:
                    values.append(all_results[dataset_name][model_name].get(metric, 0))
                else:
                    values.append(0)
            
            ax.bar(x - 0.4 + i * width, values, width, label=dataset_name, 
                 color=dataset_colors[dataset_name])
        
        # 绘制平均值柱状图
        avg_values = []
        for model_name in all_model_names:
            model_values = []
            for dataset_name in valid_datasets:
                if model_name in all_results[dataset_name]:
                    model_values.append(all_results[dataset_name][model_name].get(metric, 0))
            
            if model_values:
                avg_values.append(sum(model_values) / len(model_values))
            else:
                avg_values.append(0)
        
        ax.bar(x - 0.4 + len(valid_datasets) * width, avg_values, width, 
             label="Average", color=dataset_colors['Average'])
        
        # 为所有条形图添加数值标签
        for i, dataset_name in enumerate(valid_datasets):
            values = []
            for model_idx, model_name in enumerate(all_model_names):
                if model_name in all_results[dataset_name]:
                    value = all_results[dataset_name][model_name].get(metric, 0)
                    values.append(value)
                    ax.text(model_idx - 0.4 + i * width, value + 0.01, f"{value:.3f}", 
                          ha='center', fontsize=10, rotation=90)
                else:
                    values.append(0)
        
        # 为平均值条形图添加数值标签
        for i, v in enumerate(avg_values):
            ax.text(i - 0.4 + len(valid_datasets) * width, v + 0.01, f"{v:.3f}", 
                  ha='center', fontsize=10, rotation=90, fontweight='bold')
        
        # 设置图表属性
        ax.set_xlabel('Models', fontsize=16)
        ax.set_ylabel(f'{metric_display} Score', fontsize=16)
        ax.set_title(f'{metric_display} Score Comparison Across Datasets', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(all_model_names, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join('summary', f'{metric}_comparison.png'), dpi=300)
        plt.close()
    
    print("\nAll summary visualizations have been saved to the 'summary' folder.")
    print("\nEvaluation of all datasets complete!")

if __name__ == '__main__':
    main() 
