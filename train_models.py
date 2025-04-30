import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from datetime import datetime
from mnist_custom_model import FeatureExtractor
import cv2
from tqdm import tqdm

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义CBAM注意力机制模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio)
        self.spatialattention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

# 定义增强版CNN模型，接近AlexNet 1/4参数量
class AdvancedCNN(nn.Module):
    def __init__(self, in_channels=4):
        super(AdvancedCNN, self).__init__()
        
        # 第一层卷积块
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention1 = CBAMBlock(64)
        
        # 第二层卷积块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention2 = CBAMBlock(128)
        
        # 第三层卷积块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention3 = CBAMBlock(256)
        
        # 全连接分类器
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )
        
        # 残差连接 - 考虑池化操作的影响
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )
        
        # 池化加通道数调整
        self.shortcut2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 匹配池化操作
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )
        
        # 池化加通道数调整
        self.shortcut3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 匹配池化操作
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        
    def forward(self, x):
        # 第一个残差块
        residual = self.shortcut1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = x + residual  # 残差连接
        x = self.relu1_1(x)
        
        # 保存池化前的激活值用于残差连接
        pre_pool = x
        
        # 池化和注意力
        x = self.pool1(x)
        x = self.attention1(x)
        
        # 第二个残差块
        residual = self.shortcut2(pre_pool)  # 使用池化前的特征
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = x + residual  # 残差连接
        x = self.relu2_1(x)
        
        # 保存池化前的激活值用于残差连接
        pre_pool = x
        
        # 池化和注意力
        x = self.pool2(x)
        x = self.attention2(x)
        
        # 第三个残差块
        residual = self.shortcut3(pre_pool)  # 使用池化前的特征
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = x + residual  # 残差连接
        x = self.relu3_1(x)
        
        # 池化和注意力
        x = self.pool3(x)
        x = self.attention3(x)
        
        # 全局平均池化和分类
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# 定义轻量级CNN模型，作为备选
class LightCNN(nn.Module):
    def __init__(self, in_channels=4):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 自定义数据集类，使用FeatureExtractor进行特征提取
class EnhancedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_extractor, apply_augmentation=True):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.apply_augmentation = apply_augmentation
        # 困难样本权重字典，记录哪些样本需要额外关注
        self.difficult_samples = {}
        # 跟踪样本是否被正确分类
        self.sample_predictions = {}
        
        # 对特定数量样本进行预处理，以避免数据加载时的延迟
        # 仅预处理前1000个样本，其他样本按需处理
        self.cached_features = {}
        print(f"预处理前1000个样本 (apply_augmentation={apply_augmentation})...")
        
        # 使用tqdm创建进度条
        for idx in tqdm(range(min(1000, len(dataset)))):
            try:
                image, _ = dataset[idx]
                image_np = image.squeeze().numpy()
                # 仅提取特征，不做增强
                feature_channels = self.feature_extractor.create_feature_channels(image_np)
                self.cached_features[idx] = torch.FloatTensor(feature_channels).permute(2, 0, 1)
            except Exception as e:
                print(f"预处理样本 {idx} 时出错: {e}")
        
        print(f"预处理完成，缓存了 {len(self.cached_features)} 个样本的特征")
    
    def __len__(self):
        return len(self.dataset)
    
    def update_difficult_samples(self, indices, predictions, labels):
        """更新困难样本的权重，错误分类的样本会得到更高的权重"""
        for idx, pred, label in zip(indices, predictions, labels):
            idx = idx.item()
            is_correct = (pred == label)
            # 如果是数字8或被误分类为8，特别关注
            is_digit_8_related = (label == 8) or (pred == 8)
            
            if not is_correct:
                # 错误分类的样本增加权重
                self.difficult_samples[idx] = self.difficult_samples.get(idx, 1.0) + 0.5
                # 如果与数字8相关，额外增加权重
                if is_digit_8_related:
                    self.difficult_samples[idx] += 0.5
            else:
                # 正确分类的样本权重逐渐恢复
                if idx in self.difficult_samples:
                    self.difficult_samples[idx] = max(1.0, self.difficult_samples[idx] - 0.2)
            
            self.sample_predictions[idx] = (pred, label, is_correct)
    
    def __getitem__(self, idx):
        # 仅对前10个样本打印详细日志，避免日志过载
        verbose = idx < 10
        
        if verbose:
            print(f"Loading sample at index {idx}...")
        try:
            # 首先检查是否有缓存的特征
            if idx in self.cached_features:
                enhanced_image = self.cached_features[idx]
                _, label = self.dataset[idx]
                
                # 对于增强数据集，我们仍需要应用增强
                if self.apply_augmentation and np.random.random() < 0.7:
                    # 把缓存的特征转回numpy数组进行增强
                    image_np = enhanced_image.permute(1, 2, 0).numpy()
                    
                    # 决定是否需要额外增强
                    apply_extra_augmentation = False
                    if idx in self.difficult_samples:
                        prob_threshold = 1.0 / self.difficult_samples[idx]
                        apply_extra_augmentation = np.random.random() > prob_threshold
                    
                    # 应用增强
                    if apply_extra_augmentation:
                        if verbose:
                            print(f"  从缓存加载并应用强增强到困难样本 {idx}")
                        # 对特征图应用增强
                        for c in range(image_np.shape[2]):
                            channel = image_np[:,:,c]
                            image_np[:,:,c] = self.augment_image(channel, strong_augment=True)
                    elif np.random.random() < 0.5:
                        if verbose:
                            print(f"  从缓存加载并应用标准增强到样本 {idx}")
                        # 对特征图应用增强
                        for c in range(image_np.shape[2]):
                            channel = image_np[:,:,c]
                            image_np[:,:,c] = self.augment_image(channel)
                    
                    # 转回Tensor
                    enhanced_image = torch.FloatTensor(image_np).permute(2, 0, 1)
                
                if verbose:
                    print(f"  使用缓存的特征: {enhanced_image.shape}")
                
                return enhanced_image, label
            
            # 如果没有缓存，走正常流程
            image, label = self.dataset[idx]
            
            # 将图像转为numpy数组
            image_np = image.squeeze().numpy()
            if verbose:
                print(f"  Original image shape: {image_np.shape}, label: {label}")
            
            # 确定是否应用额外增强
            apply_extra_augmentation = False
            if idx in self.difficult_samples:
                # 困难样本的增强概率与其权重成正比
                prob_threshold = 1.0 / self.difficult_samples[idx]
                apply_extra_augmentation = np.random.random() > prob_threshold
                if verbose:
                    print(f"  Sample {idx} is a difficult sample with weight {self.difficult_samples[idx]}")
            
            # 如果启用数据增强，则应用增强
            if self.apply_augmentation:
                if apply_extra_augmentation:
                    # 对困难样本应用更强的增强
                    if verbose:
                        print(f"  Applying strong augmentation to difficult sample {idx}")
                    image_np = self.augment_image(image_np, strong_augment=True)
                elif np.random.random() < 0.5:
                    # 普通样本正常增强
                    if verbose:
                        print(f"  Applying standard augmentation to sample {idx}")
                    image_np = self.augment_image(image_np)
            
            # 特别处理数字8相关的样本
            if label == 8 and self.apply_augmentation and np.random.random() < 0.7:
                # 数字8样本有70%概率进行特定增强
                if verbose:
                    print(f"  Applying digit 8 specific augmentation to sample {idx}")
                image_np = self.specific_augment_digit_8(image_np)
            
            # 提取特征通道
            if verbose:
                print(f"  Extracting feature channels for sample {idx}")
            feature_channels = self.feature_extractor.create_feature_channels(image_np, verbose=verbose)
            if verbose:
                print(f"  Feature channels shape: {feature_channels.shape}")
            
            # 转回Tensor并调整通道顺序
            enhanced_image = torch.FloatTensor(feature_channels).permute(2, 0, 1)  # [C, H, W]
            if verbose:
                print(f"  Final enhanced image shape: {enhanced_image.shape}")
            
            return enhanced_image, label
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            raise
    
    def specific_augment_digit_8(self, image):
        """为数字8设计的特定增强"""
        augmented = image.copy()
        
        # 随机选择一种特定于数字8的增强方式
        aug_type = np.random.choice(['clarity', 'structure', 'noise_removal'])
        
        if aug_type == 'clarity':
            # 增强对比度，使8的形状更清晰
            contrast_factor = np.random.uniform(1.2, 1.8)
            mean = np.mean(augmented)
            augmented = (augmented - mean) * contrast_factor + mean
            augmented = np.clip(augmented, 0, 1)
            
        elif aug_type == 'structure':
            # 保持8的基本结构，略微侵蚀再膨胀
            binary = (augmented > 0.5).astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            augmented = dilated.astype(float)
            
        elif aug_type == 'noise_removal':
            # 去除噪点，平滑边缘
            augmented = cv2.GaussianBlur(augmented, (3, 3), 0.5)
            # 然后再次锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            augmented = cv2.filter2D(augmented, -1, kernel)
            augmented = np.clip(augmented, 0, 1)
        
        return augmented
    
    def augment_image(self, image, strong_augment=False):
        """应用数据增强，特别是背景明暗度变化"""
        augmented = image.copy()
        
        # 增强种类和概率
        aug_options = ['background_noise', 'brightness', 'contrast', 'rotation', 'scaling', 'color_inversion']
        
        # 对于强增强，可能应用多种增强
        num_augmentations = np.random.randint(2, 4) if strong_augment else 1
        
        for _ in range(num_augmentations):
            # 随机选择一种增强方式
            aug_type = np.random.choice(aug_options)
            
            if aug_type == 'background_noise':
                # 为背景添加噪声
                background_mask = augmented < 0.1  # 找到背景区域
                noise_level = np.random.uniform(0.05, 0.15) if strong_augment else np.random.uniform(0.03, 0.08)
                noise = np.random.normal(0, noise_level, augmented.shape)
                augmented[background_mask] += noise[background_mask]
                augmented = np.clip(augmented, 0, 1)  # 确保值在0-1范围内
                
            elif aug_type == 'brightness':
                # 调整亮度
                brightness_range = (0.6, 1.4) if strong_augment else (0.7, 1.3)
                brightness_factor = np.random.uniform(*brightness_range)
                augmented = augmented * brightness_factor
                augmented = np.clip(augmented, 0, 1)
                
            elif aug_type == 'contrast':
                # 调整对比度
                contrast_range = (0.7, 1.7) if strong_augment else (0.8, 1.5)
                contrast_factor = np.random.uniform(*contrast_range)
                mean_val = np.mean(augmented)
                augmented = (augmented - mean_val) * contrast_factor + mean_val
                augmented = np.clip(augmented, 0, 1)
                
            elif aug_type == 'rotation':
                # 随机旋转
                angle_range = (-20, 20) if strong_augment else (-15, 15)
                angle = np.random.uniform(*angle_range)
                rows, cols = augmented.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                augmented = cv2.warpAffine(augmented, M, (cols, rows))
                
            elif aug_type == 'scaling':
                # 随机缩放
                scale_range = (0.7, 1.3) if strong_augment else (0.8, 1.2)
                scale_factor = np.random.uniform(*scale_range)
                rows, cols = augmented.shape
                
                # 计算新尺寸
                new_rows, new_cols = int(rows * scale_factor), int(cols * scale_factor)
                
                # 进行缩放
                scaled = cv2.resize(augmented, (new_cols, new_rows))
                
                # 创建新画布并将缩放后的图像放入中心
                result = np.zeros_like(augmented)
                
                # 计算偏移量
                offset_y = max(0, (rows - new_rows) // 2)
                offset_x = max(0, (cols - new_cols) // 2)
                
                # 如果缩小了图像，放到中心
                if scale_factor < 1.0:
                    result[offset_y:offset_y+new_rows, offset_x:offset_x+new_cols] = scaled
                # 如果放大了图像，裁剪中心区域
                else:
                    crop_offset_y = max(0, (new_rows - rows) // 2)
                    crop_offset_x = max(0, (new_cols - cols) // 2)
                    result = scaled[crop_offset_y:crop_offset_y+rows, crop_offset_x:crop_offset_x+cols]
                    # 确保结果尺寸正确
                    if result.shape != augmented.shape:
                        result = cv2.resize(result, (cols, rows))
                
                augmented = result
                
            elif aug_type == 'color_inversion':
                # 黑白反转 - 将背景色和前景色互换
                augmented = 1 - augmented
        
        return augmented

# 准备数据加载器
def prepare_dataloaders(batch_size=128, val_split=0.1, max_samples=56000):
    """准备训练、验证和测试数据加载器，使用特征提取器处理图像"""
    # 添加进度信息显示
    print("Step 1: Defining transforms...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    print("Step 2: Creating feature extractor...")
    # 创建特征提取器
    feature_extractor = FeatureExtractor()
    
    print("Step 3: Loading MNIST dataset...")
    try:
        # 加载MNIST数据集
        base_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        print(f"MNIST datasets loaded successfully: {len(base_train_dataset)} training samples, {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        raise
    
    print("Step 4: Creating original and augmented datasets...")
    print("创建原始数据集 (无数据增强)...")
    original_dataset = EnhancedMNISTDataset(base_train_dataset, feature_extractor, apply_augmentation=False)
    
    print("创建增强数据集 (有数据增强)...")
    augmented_dataset = EnhancedMNISTDataset(base_train_dataset, feature_extractor, apply_augmentation=True)
    
    print("Step 5: Computing sample numbers for original/augmented ratio...")
    # 限制总样本数，但保持1:7的比例
    # 确保总样本数不超过max_samples和可用样本数中的较小值
    total_available = len(base_train_dataset)
    total_samples = min(max_samples, total_available)
    
    # 原始数据占总数的1/8，增强数据占总数的7/8
    original_samples = int(total_samples * 1/8)  # 约7000
    augmented_samples = total_samples - original_samples  # 约49000
    
    print(f"Original samples: {original_samples}, Augmented samples: {augmented_samples}, Total samples: {total_samples}")
    
    print("Step 6: Creating subsets from original and augmented datasets...")
    # 随机抽样创建原始数据集和增强数据集
    torch.manual_seed(42)  # 确保可重复性
    
    print("创建原始数据子集...")
    # 对于原始数据集，随机选择original_samples个样本
    original_indices = torch.randperm(total_available)[:original_samples]
    original_subset = torch.utils.data.Subset(original_dataset, original_indices)
    
    print("创建增强数据子集...")
    # 对于增强数据集，随机选择augmented_samples个样本
    # 为了避免重复处理相同的图像，重新生成随机索引
    torch.manual_seed(24)  # 使用不同的种子
    augmented_indices = torch.randperm(total_available)[:augmented_samples]
    augmented_subset = torch.utils.data.Subset(augmented_dataset, augmented_indices)
    
    print("Step 7: Concatenating datasets...")
    # 将两个数据集合并
    train_dataset = torch.utils.data.ConcatDataset([original_subset, augmented_subset])
    
    print("Step 8: Splitting into training and validation sets...")
    # 划分训练集和验证集
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])
    
    print("Step 9: Creating enhanced test dataset...")
    # 为测试集创建增强数据集
    print("处理测试集数据...")
    enhanced_test_dataset = EnhancedMNISTDataset(test_dataset, feature_extractor, apply_augmentation=False)
    
    print("Step 10: Creating data loaders...")
    # 创建数据加载器
    print("创建训练数据加载器...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    print("创建验证数据加载器...")
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print("创建测试数据加载器...")
    test_loader = DataLoader(enhanced_test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Data loaders created successfully.")
    return train_loader, val_loader, test_loader

# 训练一个epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    batch_indices = []
    batch_predictions = []
    batch_labels = []
    
    # 添加进度条
    print(f"训练批次: 0/{len(dataloader)}", end='\r')
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 检测困难样本 - 获取预测结果
        _, predicted = outputs.max(1)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        total_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 收集批次索引、预测和标签，用于更新困难样本集
        indices = torch.arange(0, inputs.size(0))
        batch_indices.append(indices)
        batch_predictions.append(predicted.cpu())
        batch_labels.append(targets.cpu())
        
        # 更新进度条
        print(f"训练批次: {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f} - Acc: {predicted.eq(targets).sum().item()/targets.size(0):.4f}", end='\r')
    
    print() # 换行
    
    # 更新困难样本集
    if isinstance(dataloader.dataset, torch.utils.data.Subset):
        # 如果是Subset，要转换索引到原始数据集
        if hasattr(dataloader.dataset.dataset, 'update_difficult_samples'):
            for indices, preds, labels in zip(batch_indices, batch_predictions, batch_labels):
                # 调整索引以匹配原始数据集
                original_indices = torch.tensor([dataloader.dataset.indices[idx] for idx in indices])
                dataloader.dataset.dataset.update_difficult_samples(original_indices, preds, labels)
    elif hasattr(dataloader.dataset, 'update_difficult_samples'):
        # 直接更新原始数据集
        for indices, preds, labels in zip(batch_indices, batch_predictions, batch_labels):
            dataloader.dataset.update_difficult_samples(indices, preds, labels)
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

# 验证模型
def validate(model, dataloader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 添加进度条
    print(f"验证批次: 0/{len(dataloader)}", end='\r')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计损失和准确率
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            print(f"验证批次: {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f} - Acc: {predicted.eq(targets).sum().item()/targets.size(0):.4f}", end='\r')
    
    print() # 换行
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

# 训练模型
def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001, 
                weight_decay=1e-5, model_name="model"):
    """训练模型的完整流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # 设置类别权重，增加数字8的权重以改善混淆问题
    # 权重分布: 默认为1.0，给数字8设置更高权重，同时给容易与8混淆的数字(0,1,3,5,6,9)设置略高权重
    class_weights = torch.ones(10)
    class_weights[8] = 2.0  # 数字8的权重增加
    class_weights[0] = 1.2  # 容易与8混淆的数字
    class_weights[1] = 1.2
    class_weights[3] = 1.2
    class_weights[5] = 1.2
    class_weights[6] = 1.2
    class_weights[9] = 1.2
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # 确保模型保存目录存在
    os.makedirs('models', exist_ok=True)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'digit_8_acc': []  # 特别跟踪数字8的准确率
    }
    
    best_val_acc = 0.0
    best_digit_8_acc = 0.0
    
    print("\n训练进度:")
    print("="*50)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*50)
        
        # 训练阶段
        start_time = time.time()
        print("训练阶段:")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time
        
        # 验证阶段
        start_time = time.time()
        print("验证阶段:")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_time = time.time() - start_time
        
        # 特别评估数字8的准确率
        print("评估数字8准确率...")
        digit_8_acc = evaluate_digit_accuracy(model, val_loader, device, target_digit=8)
        
        # 学习率调整 - 基于数字8的准确率和整体验证损失的组合
        scheduler.step(val_loss)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['digit_8_acc'].append(digit_8_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} (用时: {train_time:.2f}s), "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} (用时: {val_time:.2f}s), "
              f"Digit 8 Acc: {digit_8_acc:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型 - 结合整体准确率和数字8的准确率
        combined_score = val_acc * 0.7 + digit_8_acc * 0.3  # 70%整体准确率，30%数字8准确率的权重
        if combined_score > (best_val_acc * 0.7 + best_digit_8_acc * 0.3):
            best_val_acc = val_acc
            best_digit_8_acc = digit_8_acc
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"保存最佳模型 (val_acc={val_acc:.4f}, digit_8_acc={digit_8_acc:.4f})")
        
        print("="*50)
    
    # 保存最终模型
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    print(f"模型已保存: models/{model_name}_final.pth")
    
    return model, history

# 评估特定数字的准确率
def evaluate_digit_accuracy(model, dataloader, device, target_digit=8):
    """评估模型对特定数字的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 只关注目标数字
            digit_mask = (targets == target_digit)
            if not torch.any(digit_mask):
                continue
                
            # 前向传播
            outputs = model(inputs[digit_mask])
            _, predicted = outputs.max(1)
            
            # 计算目标数字的准确率
            total += digit_mask.sum().item()
            correct += (predicted == target_digit).sum().item()
            
            # 显示进度 - 每10个批次更新一次
            if batch_idx % 10 == 0:
                print(f"评估数字 {target_digit} 进度: {batch_idx}/{len(dataloader)}", end='\r')
    
    print() # 换行
    
    return correct / total if total > 0 else 0.0

# 分析数字8混淆情况
def analyze_digit_8_confusion(model, dataloader, device):
    """分析模型对数字8的混淆情况，返回详细统计信息"""
    model.eval()
    digit_8_stats = {
        'total': 0,       # 数字8的总样本数 
        'correct': 0,     # 正确识别的数字8样本数
        'confused_as': {},  # 数字8被误认为其他数字的统计
        'other_as_8': {}    # 其他数字被误认为8的统计
    }
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # 将预测结果和目标转为numpy数组，方便处理
            preds_np = predicted.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # 统计数字8的样本
            digit_8_mask = (targets_np == 8)
            digit_8_stats['total'] += np.sum(digit_8_mask)
            digit_8_stats['correct'] += np.sum(np.logical_and(digit_8_mask, preds_np == 8))
            
            # 统计数字8被误认为其他数字的情况
            for i in range(10):
                if i == 8:
                    continue
                count = np.sum(np.logical_and(digit_8_mask, preds_np == i))
                if count > 0:
                    digit_8_stats['confused_as'][i] = digit_8_stats['confused_as'].get(i, 0) + count
            
            # 统计其他数字被误认为8的情况
            for i in range(10):
                if i == 8:
                    continue
                count = np.sum(np.logical_and(targets_np == i, preds_np == 8))
                if count > 0:
                    digit_8_stats['other_as_8'][i] = digit_8_stats['other_as_8'].get(i, 0) + count
            
            all_preds.extend(preds_np)
            all_targets.extend(targets_np)
    
    # 计算8的精确率和召回率
    if np.sum(np.array(all_preds) == 8) > 0:
        precision = digit_8_stats['correct'] / np.sum(np.array(all_preds) == 8)
    else:
        precision = 0.0
        
    if digit_8_stats['total'] > 0:
        recall = digit_8_stats['correct'] / digit_8_stats['total']
    else:
        recall = 0.0
    
    # 计算F1分数
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    digit_8_stats['precision'] = precision
    digit_8_stats['recall'] = recall
    digit_8_stats['f1'] = f1
    
    return digit_8_stats

# 评估模型
def evaluate_model(model, test_loader, device):
    """在测试集上评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    print(f"模型评估中: 0/{len(test_loader)}", end='\r')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            print(f"模型评估中: {batch_idx+1}/{len(test_loader)} - Acc: {predicted.eq(targets).sum().item()/targets.size(0):.4f}", end='\r')
    
    print() # 换行
    
    accuracy = correct / total
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # 分析数字8的混淆情况
    print("分析数字8的识别情况...")
    digit_8_stats = analyze_digit_8_confusion(model, test_loader, device)
    
    # 输出数字8的详细分析
    print(f"\n数字8分析结果:")
    print(f"  准确率: {digit_8_stats['correct']/digit_8_stats['total']:.4f} ({digit_8_stats['correct']}/{digit_8_stats['total']})")
    print(f"  精确率: {digit_8_stats['precision']:.4f}")
    print(f"  召回率: {digit_8_stats['recall']:.4f}")
    print(f"  F1分数: {digit_8_stats['f1']:.4f}")
    
    print("\n  数字8被错误识别为:")
    for digit, count in sorted(digit_8_stats['confused_as'].items()):
        print(f"    {digit}: {count} ({count/digit_8_stats['total']:.2%})")
    
    print("\n  其他数字被错误识别为8:")
    for digit, count in sorted(digit_8_stats['other_as_8'].items()):
        print(f"    {digit}: {count}")
    
    return accuracy, conf_matrix, digit_8_stats

# 集成多个模型的预测
def ensemble_predictions(models, test_loader, device):
    """集成多个模型的预测结果"""
    for model in models:
        model.eval()
    
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    # 记录每个样本的详细预测情况
    sample_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取每个模型的输出概率
            model_probs = []
            for model in models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                model_probs.append(probs)
            
            # 计算集成输出
            ensemble_probs = torch.zeros_like(model_probs[0])
            for probs in model_probs:
                ensemble_probs += probs
            ensemble_probs /= len(models)
            
            # 获取每个模型的独立预测
            model_predictions = []
            for probs in model_probs:
                _, model_pred = probs.max(1)
                model_predictions.append(model_pred)
            
            # 计算集成预测
            _, predicted = ensemble_probs.max(1)
            
            # 收集模型的共识度和详细预测情况
            for i in range(targets.size(0)):
                sample_detail = {
                    'target': targets[i].item(),
                    'ensemble_pred': predicted[i].item(),
                    'model_preds': [p[i].item() for p in model_predictions],
                    'ensemble_probs': ensemble_probs[i].cpu().numpy(),
                    'is_correct': predicted[i] == targets[i]
                }
                # 计算模型共识度 - 有多少模型达成一致
                consensus = sum(1 for p in sample_detail['model_preds'] 
                               if p == sample_detail['ensemble_pred']) / len(models)
                sample_detail['consensus'] = consensus
                sample_predictions.append(sample_detail)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct / total
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # 分析数字8的统计情况
    digit_8_stats = {
        'total': 0,
        'correct': 0,
        'confused_as': {},
        'other_as_8': {},
        'consensus_correct': 0,
        'consensus_total': 0
    }
    
    # 分析每个样本的预测详情
    for sample in sample_predictions:
        if sample['target'] == 8:
            digit_8_stats['total'] += 1
            if sample['ensemble_pred'] == 8:
                digit_8_stats['correct'] += 1
                digit_8_stats['consensus_correct'] += sample['consensus']
                digit_8_stats['consensus_total'] += 1
            else:
                # 记录数字8被误认为其他数字的情况
                digit_8_stats['confused_as'][sample['ensemble_pred']] = \
                    digit_8_stats['confused_as'].get(sample['ensemble_pred'], 0) + 1
        elif sample['ensemble_pred'] == 8:
            # 记录其他数字被误认为8的情况
            digit_8_stats['other_as_8'][sample['target']] = \
                digit_8_stats['other_as_8'].get(sample['target'], 0) + 1
    
    # 计算精确率和召回率
    if all_preds.count(8) > 0:
        precision = digit_8_stats['correct'] / all_preds.count(8)
    else:
        precision = 0.0
        
    if digit_8_stats['total'] > 0:
        recall = digit_8_stats['correct'] / digit_8_stats['total']
        # 计算数字8正确预测时的平均共识度
        avg_consensus = digit_8_stats['consensus_correct'] / digit_8_stats['consensus_total'] \
                        if digit_8_stats['consensus_total'] > 0 else 0
    else:
        recall = 0.0
        avg_consensus = 0.0
    
    # 计算F1分数
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    digit_8_stats['precision'] = precision
    digit_8_stats['recall'] = recall
    digit_8_stats['f1'] = f1
    digit_8_stats['avg_consensus'] = avg_consensus
    
    # 输出数字8的详细分析
    print(f"\nEnsemble Digit 8 Analysis:")
    print(f"  Accuracy: {digit_8_stats['correct']/digit_8_stats['total']:.4f} ({digit_8_stats['correct']}/{digit_8_stats['total']})")
    print(f"  Precision: {digit_8_stats['precision']:.4f}")
    print(f"  Recall: {digit_8_stats['recall']:.4f}")
    print(f"  F1 Score: {digit_8_stats['f1']:.4f}")
    print(f"  Average Consensus (correct predictions): {digit_8_stats['avg_consensus']:.4f}")
    
    print("\n  Digit 8 confused as:")
    for digit, count in sorted(digit_8_stats['confused_as'].items()):
        print(f"    {digit}: {count} ({count/digit_8_stats['total']:.2%})")
    
    print("\n  Other digits confused as 8:")
    for digit, count in sorted(digit_8_stats['other_as_8'].items()):
        print(f"    {digit}: {count}")
    
    # 查找一些难以分类的数字8样本
    difficult_8_samples = [s for s in sample_predictions 
                          if s['target'] == 8 and not s['is_correct'] and s['consensus'] < 0.5]
    print(f"\n  Found {len(difficult_8_samples)} difficult digit 8 samples with low consensus.")
    
    return accuracy, conf_matrix, digit_8_stats, sample_predictions

# 绘制训练历史
def plot_training_history(history, title="Model Training History"):
    """绘制训练历史曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # 保存图像
    plt.savefig(f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(conf_matrix, title="Confusion Matrix"):
    """绘制混淆矩阵热图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    # 保存图像
    plt.savefig(f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

# 主函数
def main(num_models=3, epochs=15, batch_size=64, max_samples=56000):
    """主函数，运行整个训练和评估流程"""
    # 设置详细的调试日志
    import sys
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Command line arguments: {sys.argv}")
    print(f"Training {num_models} models for {epochs} epochs each")
    print(f"Batch size: {batch_size}, Max samples: {max_samples}")
    
    # 准备数据
    print("="*50)
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(batch_size=batch_size, max_samples=max_samples)
    print(f"Data prepared - Dataset sizes:")
    print(f"  Training: {len(train_loader.dataset)} samples in {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader.dataset)} samples in {len(val_loader)} batches")
    print(f"  Testing: {len(test_loader.dataset)} samples in {len(test_loader)} batches")
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # 训练多个模型
    models = []
    histories = []
    
    for i in range(num_models):
        print("="*50)
        print(f"MODEL {i+1}/{num_models} TRAINING")
        print("="*50)
        
        # 尝试使用高级模型，如果失败则回退到简单模型
        try:
            print("Initializing advanced model...")
            model = AdvancedCNN()
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Advanced model initialized with {num_params:,} parameters")
            
            # 验证模型能否处理输入数据
            print("Validating model architecture with sample data...")
            sample_input = next(iter(train_loader))[0][:1].to(device)
            model = model.to(device)
            with torch.no_grad():
                output = model(sample_input)
                print(f"Model output shape: {output.shape}")
            print("Model architecture validated successfully.")
            
        except Exception as e:
            print(f"Error with advanced model: {e}")
            print("Falling back to simpler model...")
            model = LightCNN()
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Light model initialized with {num_params:,} parameters")
        
        # 训练模型
        print("-"*40)
        print(f"Starting training for model {i+1}...")
        try:
            model, history = train_model(
                model, 
                train_loader, 
                val_loader, 
                num_epochs=epochs, 
                model_name=f"model_{i+1}"
            )
            models.append(model)
            histories.append(history)
            print(f"Model {i+1} training completed successfully.")
        except Exception as e:
            print(f"Error training model {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue  # 继续训练下一个模型而不是退出
    
    # 如果没有成功训练的模型，则退出
    if len(models) == 0:
        print("No models were trained successfully. Exiting.")
        return
    
    # 评估每个单独模型
    print("="*50)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*50)
    for i, model in enumerate(models):
        print(f"Evaluating model {i+1}/{len(models)}...")
        try:
            accuracy, conf_matrix, digit_8_stats = evaluate_model(model, test_loader, device)
            print(f"Model {i+1} test accuracy: {accuracy:.4f}")
            print(f"Saving model {i+1} confusion matrix and training history...")
            plot_confusion_matrix(conf_matrix, title=f"Model {i+1} Confusion Matrix")
            plot_training_history(histories[i], title=f"Model {i+1} Training History")
        except Exception as e:
            print(f"Error evaluating model {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # 评估集成模型
    if len(models) > 1:
        print("="*50)
        print("ENSEMBLE MODEL EVALUATION")
        print("="*50)
        try:
            print("Evaluating ensemble model...")
            ensemble_acc, ensemble_conf_matrix, ensemble_digit_8_stats, sample_predictions = ensemble_predictions(models, test_loader, device)
            print(f"Ensemble model test accuracy: {ensemble_acc:.4f}")
            
            print("Generating ensemble model visualizations...")
            # 绘制增强版混淆矩阵
            plt.figure(figsize=(12, 10))
            sns.heatmap(ensemble_conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=range(10), yticklabels=range(10))
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Ensemble Model Confusion Matrix')
            matrix_filename = f"ensemble_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(matrix_filename)
            plt.close()
            print(f"Saved confusion matrix to {matrix_filename}")
            
            # 绘制模型一致性与准确率的关系
            consensus_values = [s['consensus'] for s in sample_predictions]
            is_correct = [1 if s['is_correct'] else 0 for s in sample_predictions]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(consensus_values, is_correct, alpha=0.3)
            plt.xlabel('Model Consensus')
            plt.ylabel('Prediction Correct')
            plt.title('Relationship Between Model Consensus and Accuracy')
            plt.grid(True, alpha=0.3)
            consensus_filename = f"model_consensus_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(consensus_filename)
            plt.close()
            print(f"Saved consensus plot to {consensus_filename}")
        except Exception as e:
            print(f"Error evaluating ensemble model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nOnly one model trained successfully. Skipping ensemble evaluation.")
    
    print("="*50)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*50)
    print(f"Model files saved in 'models/' directory")
    print(f"Result charts saved in current directory")

if __name__ == "__main__":
    print("Script started...")
    print("PyTorch version:", torch.__version__)
    print("NumPy version:", np.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    # 使用原始参数运行训练
    try:
        print("Starting main function with original parameters...")
        main(num_models=3, epochs=15)
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc() 