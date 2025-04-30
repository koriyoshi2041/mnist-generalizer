import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from mnist_custom_model import FeatureExtractor as BaseFeatureExtractor
from skimage.feature import hog
import cv2
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
from datetime import datetime
from tqdm import tqdm

# 高级特征提取器 - 保持4通道输出，兼容现有模型
class AdvancedFeatureExtractor(BaseFeatureExtractor):
    """高级特征提取器，添加HOG特征和高级纹理特征，保持4通道输出以兼容现有模型"""
    def __init__(self, weights=None):
        # 默认特征权重 [原图, 骨架特征, 密度特征, 梯度特征]
        # 增强梯度特征的权重，可能会使性能更好
        self.default_weights = [0.2, 0.2, 0.2, 0.6]
        weights = weights if weights is not None else self.default_weights
        super(AdvancedFeatureExtractor, self).__init__(weights=weights)
    
    def extract_hog_features(self, image):
        """提取HOG特征"""
        # 确保图像已预处理
        image = self.preprocess(image)
        
        # 将图像转换为uint8类型，HOG需要0-255的输入
        image_uint8 = (image * 255).astype(np.uint8)
        
        # 提取HOG特征
        # 参数调整：较小的cell尺寸可以捕获更精细的细节
        fd, hog_image = hog(
            image_uint8, 
            orientations=9, 
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), 
            visualize=True,
            block_norm='L2-Hys'
        )
        
        # 归一化HOG图像
        hog_image = hog_image / np.max(hog_image) if np.max(hog_image) > 0 else hog_image
        
        return hog_image
    
    def extract_advanced_texture(self, image):
        """提取高级纹理特征"""
        # 确保图像已预处理
        image = self.preprocess(image)
        image_uint8 = (image * 255).astype(np.uint8)
        
        # 多尺度LBP - 捕获不同尺度的纹理信息
        lbp_small = local_binary_pattern(image, P=8, R=1, method='uniform')
        lbp_medium = local_binary_pattern(image, P=16, R=2, method='uniform')
        
        # 归一化
        lbp_small = lbp_small / np.max(lbp_small) if np.max(lbp_small) > 0 else lbp_small
        lbp_medium = lbp_medium / np.max(lbp_medium) if np.max(lbp_medium) > 0 else lbp_medium
        
        # 组合多尺度LBP特征
        combined_texture = (lbp_small * 0.6 + lbp_medium * 0.4)
        
        # 边缘检测 - 使用Canny
        edges = cv2.Canny(image_uint8, 50, 150) / 255.0
        
        # 组合边缘和纹理特征
        advanced_texture = combined_texture * 0.7 + edges * 0.3
        
        return advanced_texture
    
    def create_feature_channels(self, image, verbose=False):
        """创建特征通道，保持4通道输出但极其轻微地增强现有特征"""
        try:
            # 基础预处理
            processed = self.preprocess(image)
            
            # 提取基本特征
            morph_features = self.extract_morphological_features(processed)
            regional_features = self.extract_regional_features(processed)
            texture_features = self.extract_texture_features(processed)
            
            # 提取额外特征
            hog_features = self.extract_hog_features(processed)
            advanced_texture = self.extract_advanced_texture(processed)
            
            # 创建4通道图像并应用权重，但极其轻微地增强特征
            # 使用更高级的梯度特征极其轻微地增强基本梯度特征 (97% 原始 + 3% 高级)
            enhanced_gradient = texture_features['gradient_magnitude'] * 0.97 + hog_features * 0.03
            
            # 极其轻微地增强骨架特征 (98% 原始 + 2% 高级)
            enhanced_skeleton = morph_features['skeleton'] * 0.98 + advanced_texture * 0.02
            
            # 最终4通道 - 使用调整后的权重
            channels = [
                processed * self.weights[0],                  # 原始预处理图像
                enhanced_skeleton * self.weights[1],          # 极其轻微增强的骨架图
                regional_features * self.weights[2],          # 区域密度图 (保持不变)
                enhanced_gradient * self.weights[3]           # 极其轻微增强的梯度特征
            ]
            
            # 堆叠通道 - 保持4通道输出
            multi_channel = np.stack(channels, axis=-1)
            
            return multi_channel
            
        except Exception as e:
            print(f"Error in advanced feature extraction: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果高级特征提取失败，回退到基本特征提取
            print("Falling back to basic feature extraction...")
            return super().create_feature_channels(image, verbose)

# 注意力机制 - 缩放版本的ECA (Efficient Channel Attention)
class ECABlock(nn.Module):
    """高效通道注意力块"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((np.log2(channels) / gamma) + b / gamma))
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)
        
        # 重塑为序列
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 应用1D卷积
        y = self.conv(y)
        
        # 重塑并应用sigmoid激活
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x * y

# 空间注意力模块
class SpatialAttention(nn.Module):
    """改进的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接
        x = torch.cat([avg_out, max_out], dim=1)
        
        # 应用卷积和sigmoid激活
        x = self.conv(x)
        return self.sigmoid(x)

# 组合注意力块
class CombinedAttention(nn.Module):
    """组合ECA通道注意力和空间注意力"""
    def __init__(self, channels):
        super(CombinedAttention, self).__init__()
        self.channel_attention = ECABlock(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # 应用通道注意力
        x = self.channel_attention(x)
        
        # 应用空间注意力
        spatial_map = self.spatial_attention(x)
        x = x * spatial_map
        
        return x

# 残差块，支持SE注意力
class ResidualBlock(nn.Module):
    """带有注意力机制的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_attention=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        # 条件添加注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = CombinedAttention(out_channels)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用注意力
        if self.use_attention:
            out = self.attention(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

# 困难数字专注模块
class DifficultDigitModule(nn.Module):
    """专注于困难数字（0，7，8，9）识别的模块"""
    def __init__(self, in_channels):
        super(DifficultDigitModule, self).__init__()
        
        # 为每个困难数字创建单独的特征提取器
        self.digit_0_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.digit_7_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.digit_8_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.digit_9_conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        
        # 整合困难数字特征
        self.integration = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 各数字特征提取
        feat_0 = F.relu(self.digit_0_conv(x))
        feat_7 = F.relu(self.digit_7_conv(x))
        feat_8 = F.relu(self.digit_8_conv(x))
        feat_9 = F.relu(self.digit_9_conv(x))
        
        # 特征合并
        concat_feats = torch.cat([feat_0, feat_7, feat_8, feat_9], dim=1)
        attention_map = self.integration(concat_feats)
        
        # 应用注意力，增强原始特征
        enhanced = x * attention_map
        
        # 残差连接
        return x + enhanced

# 增强版模型 - 结合残差结构、注意力机制和困难数字模块
class EnhancedCNN(nn.Module):
    """增强版CNN模型，专为手写数字识别优化"""
    def __init__(self, in_channels=6):  # 默认使用6通道输入
        super(EnhancedCNN, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 第一阶段 - 两个残差块
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.attention1 = CombinedAttention(64)
        
        # 下采样
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二阶段 - 两个残差块
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=1)
        self.attention2 = CombinedAttention(128)
        
        # 下采样
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三阶段 - 三个残差块
        self.layer3 = self._make_layer(128, 256, blocks=3, stride=1)
        self.attention3 = CombinedAttention(256)
        
        # 下采样
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 困难数字专注模块
        self.difficult_digit_module = DifficultDigitModule(256)
        
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
        
        # 辅助分类器 - 专注于识别数字8
        self.aux_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # 二分类：是8 vs 不是8
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第一阶段
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.pool1(x)
        
        # 第二阶段
        x = self.layer2(x)
        x = self.attention2(x)
        x = self.pool2(x)
        
        # 第三阶段
        x = self.layer3(x)
        x = self.attention3(x)
        
        # 应用困难数字专注模块
        x = self.difficult_digit_module(x)
        
        x = self.pool3(x)
        
        # 全局池化
        x = self.avg_pool(x)
        features = x.view(x.size(0), -1)
        
        # 主分类器
        main_logits = self.classifier(features)
        
        # 辅助分类器
        aux_logits = self.aux_classifier(features)
        
        return main_logits, aux_logits

# 混合损失函数 - 组合交叉熵、Focal Loss和辅助损失
class MixedLoss(nn.Module):
    """混合损失函数，结合交叉熵、Focal Loss和针对数字8的辅助损失"""
    def __init__(self, alpha=0.25, gamma=2.0, ce_weight=0.6, focal_weight=0.3, aux_weight=0.1):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss()
        self.aux_ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # 解包输出
        main_logits, aux_logits = outputs
        
        # 交叉熵损失
        ce_loss = self.ce(main_logits, targets)
        
        # Focal Loss
        # 计算概率
        pt = torch.exp(-ce_loss)
        # 应用公式
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # 创建辅助分类器的目标 - 数字8为1，其他为0
        aux_targets = (targets == 8).long()
        aux_loss = self.aux_ce(aux_logits, aux_targets)
        
        # 混合损失
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss + self.aux_weight * aux_loss

# 带有预热和余弦退火的学习率调度器
class WarmupCosineScheduler:
    """实现学习率预热和余弦退火调度"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性预热阶段
            lr_scale = epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * lr_scale
        else:
            # 余弦退火阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * cosine_decay

# 自适应模型训练函数
def train_enhanced_model(model, train_loader, val_loader, device, 
                        num_epochs=25, lr=0.001, weight_decay=1e-5, 
                        model_name="enhanced_model_90"):
    """训练增强版模型"""
    print(f"Training {model_name}...", flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Number of epochs: {num_epochs}", flush=True)
    print(f"Learning rate: {lr}", flush=True)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("优化器初始化完成", flush=True)
    
    # 使用自定义混合损失函数
    criterion = MixedLoss(alpha=0.25, gamma=2.0, ce_weight=0.6, focal_weight=0.3, aux_weight=0.1)
    print("损失函数初始化完成", flush=True)
    
    # 学习率调度器
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, total_epochs=num_epochs)
    print("学习率调度器初始化完成", flush=True)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'digit_8_acc': []
    }
    
    # 最佳验证精度和对应的模型状态
    best_val_acc = 0.0
    best_model_state = None
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    print("创建模型保存目录完成", flush=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\n开始第 {epoch+1}/{num_epochs} 个epoch", flush=True)
        
        # 更新学习率
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr}", flush=True)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 创建进度条
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        print(f"开始训练阶段，共有 {len(train_loader)} 个批次", flush=True)
        batch_count = 0
        
        for i, (images, labels) in pbar:
            try:
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"处理批次 {batch_count}/{len(train_loader)}", flush=True)
                
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                
                # 更新统计
                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item(), "lr": current_lr})
                
            except Exception as e:
                print(f"训练过程中出错: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        print(f"Epoch {epoch+1} 训练损失: {train_loss:.4f}", flush=True)
        
        # 验证阶段
        print("开始验证阶段", flush=True)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        digit_8_correct = 0
        digit_8_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 更新统计
                val_loss += loss.item()
                
                # 计算主分类器精度
                main_outputs = outputs[0]
                _, predicted = torch.max(main_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 计算数字8的精度
                mask_8 = (labels == 8)
                digit_8_total += mask_8.sum().item()
                digit_8_correct += ((predicted == 8) & mask_8).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        digit_8_acc = 100 * digit_8_correct / max(1, digit_8_total)  # 避免除以零
        
        # 更新历史
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['digit_8_acc'].append(digit_8_acc)
        
        # 打印结果
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Digit 8 Acc: {digit_8_acc:.2f}%", flush=True)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"models/{model_name}_best.pth")
            print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%", flush=True)
        
        # 每个epoch都保存检查点
        print(f"保存epoch {epoch+1} 的检查点", flush=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_val_acc': best_val_acc
        }, f"models/checkpoints/{model_name}_epoch{epoch+1}.pth")
    
    # 保存最终模型
    print("保存最终模型", flush=True)
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    
    # 恢复最佳模型的权重
    print("恢复最佳模型权重", flush=True)
    model.load_state_dict(best_model_state)
    
    return model, history

# 高级图像增强 - 模拟自定义数据集的特点
class AdvancedAugmentation:
    """高级图像增强，专为手写数字设计"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        """应用高级增强"""
        # 将PyTorch张量转换为NumPy数组
        if isinstance(img, torch.Tensor):
            img = img.numpy().transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img[:, :, 0]
            is_tensor = True
        else:
            is_tensor = False
        
        # 确保图像是float32，范围0-1
        if img.max() > 1.0:
            img = img / 255.0
        
        # 以一定概率应用增强
        if np.random.random() < self.p:
            # 选择一种增强方法
            aug_type = np.random.choice([
                'elastic', 'perspective', 'noise', 'blur', 'thickness', 'shear', 'identity'
            ], p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1])
            
            if aug_type == 'elastic':
                # 弹性变形 - 模拟手写时的笔触不均
                img = self._elastic_transform(img)
            elif aug_type == 'perspective':
                # 透视变换 - 模拟不同角度的观察
                img = self._perspective_transform(img)
            elif aug_type == 'noise':
                # 添加噪声 - 模拟扫描或照片质量不佳
                img = self._add_noise(img)
            elif aug_type == 'blur':
                # 模拟模糊 - 低分辨率或模糊的手写
                img = self._add_blur(img)
            elif aug_type == 'thickness':
                # 改变线条粗细 - 模拟不同书写工具
                img = self._change_thickness(img)
            elif aug_type == 'shear':
                # 剪切变换 - 模拟倾斜的手写
                img = self._shear_transform(img)
        
        # 如果输入是张量，转换回张量
        if is_tensor:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img.copy())
        
        return img
    
    def _elastic_transform(self, image, alpha=40, sigma=5):
        """弹性变形"""
        shape = image.shape
        dx = np.random.uniform(-1, 1, shape) * alpha
        dy = np.random.uniform(-1, 1, shape) * alpha
        
        # 高斯滤波使变形平滑
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    def _perspective_transform(self, image):
        """透视变换"""
        h, w = image.shape[:2]
        
        # 生成随机的透视变换
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # 随机移动四个角点，但保持一定限制以避免过度变形
        max_shift = min(w, h) * 0.1
        dst_pts = src_pts + np.random.uniform(-max_shift, max_shift, src_pts.shape)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 应用变换
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    
    def _add_noise(self, image):
        """添加噪声"""
        noise_type = np.random.choice(['gaussian', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            # 高斯噪声
            mean = 0
            var = np.random.uniform(0.001, 0.01)
            sigma = var ** 0.5
            noise = np.random.normal(mean, sigma, image.shape)
            noisy_image = image + noise
            return np.clip(noisy_image, 0, 1)
        else:
            # 椒盐噪声
            s_vs_p = 0.5
            amount = np.random.uniform(0.001, 0.01)
            noisy_image = np.copy(image)
            
            # 盐噪声（白点）
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[tuple(coords)] = 1
            
            # 椒噪声（黑点）
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[tuple(coords)] = 0
            
            return noisy_image
    
    def _add_blur(self, image):
        """添加模糊效果"""
        blur_type = np.random.choice(['gaussian', 'motion'])
        
        if blur_type == 'gaussian':
            # 高斯模糊
            kernel_size = np.random.choice([3, 5])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            # 运动模糊
            kernel_size = np.random.choice([3, 5, 7])
            kernel = np.zeros((kernel_size, kernel_size))
            
            # 随机选择一个方向
            if np.random.random() < 0.5:
                # 水平方向
                kernel[kernel_size // 2, :] = 1.0
            else:
                # 垂直方向
                kernel[:, kernel_size // 2] = 1.0
            
            kernel = kernel / kernel_size
            return cv2.filter2D(image, -1, kernel)
    
    def _change_thickness(self, image):
        """改变线条粗细"""
        operation = np.random.choice(['thin', 'thick'])
        kernel_size = np.random.choice([2, 3])
        
        if operation == 'thin':
            # 使线条变细
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.erode(image, kernel, iterations=1)
        else:
            # 使线条变粗
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.dilate(image, kernel, iterations=1)
    
    def _shear_transform(self, image):
        """剪切变换"""
        h, w = image.shape[:2]
        
        # 随机剪切角度
        shear_factor = np.random.uniform(-0.2, 0.2)
        
        # 剪切矩阵
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        
        # 应用变换
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

# 自定义数据集 - 用于领域自适应
class DomainAdaptationDataset(Dataset):
    """领域自适应数据集，结合MNIST和自定义数据"""
    def __init__(self, mnist_dataset, custom_dataset=None, feature_extractor=None, 
                 domain_weight=0.5, transform=None, custom_transform=None):
        """
        初始化领域自适应数据集
        
        参数:
            mnist_dataset: MNIST数据集
            custom_dataset: 自定义数据集（可以为None）
            feature_extractor: 特征提取器
            domain_weight: 自定义数据的权重 (0-1)
            transform: 应用于所有数据的变换
            custom_transform: 仅应用于自定义数据的变换
        """
        self.mnist_dataset = mnist_dataset
        self.custom_dataset = custom_dataset
        self.feature_extractor = feature_extractor or AdvancedFeatureExtractor()
        self.domain_weight = domain_weight  # 控制采样概率
        self.transform = transform
        self.custom_transform = custom_transform
        
        # 如果没有自定义数据集，则仅使用MNIST
        self.has_custom = custom_dataset is not None
        
        # 记录难例
        self.difficult_samples = []
        self.difficult_threshold = 0.2  # 难例判定阈值
    
    def __len__(self):
        if self.has_custom:
            return len(self.mnist_dataset) + len(self.custom_dataset)
        else:
            return len(self.mnist_dataset)
    
    def update_difficult_samples(self, indices, predictions, labels):
        """更新困难样本列表"""
        for idx, pred, label in zip(indices, predictions, labels):
            if pred != label:
                # 错误预测，添加到困难样本
                if idx not in self.difficult_samples:
                    self.difficult_samples.append(idx)
        
        # 限制困难样本数量
        if len(self.difficult_samples) > int(len(self) * self.difficult_threshold):
            # 随机删除一些样本以保持大小
            np.random.shuffle(self.difficult_samples)
            self.difficult_samples = self.difficult_samples[:int(len(self) * self.difficult_threshold)]
    
    def __getitem__(self, idx):
        """获取数据项，使用难例挖掘和领域自适应策略"""
        # 确定是否使用困难样本
        use_difficult = len(self.difficult_samples) > 0 and np.random.random() < 0.3
        
        if use_difficult:
            # 从困难样本中随机选择
            idx = np.random.choice(self.difficult_samples)
        
        # 确定从哪个数据集采样
        use_custom = self.has_custom and np.random.random() < self.domain_weight
        
        if use_custom:
            # 从自定义数据集采样
            custom_idx = idx % len(self.custom_dataset)
            image, label = self.custom_dataset[custom_idx]
            
            # 应用自定义数据的特殊变换
            if self.custom_transform:
                image = self.custom_transform(image)
        else:
            # 从MNIST采样
            mnist_idx = idx % len(self.mnist_dataset)
            image, label = self.mnist_dataset[mnist_idx]
        
        # 提取特征
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
            if image_np.shape[0] == 1:  # 如果是单通道图像
                image_np = image_np.squeeze(0)
        else:
            image_np = np.array(image)
        
        # 使用特征提取器创建多通道特征
        feature_channels = self.feature_extractor.create_feature_channels(image_np)
        
        # 转换为torch张量
        feature_tensor = torch.from_numpy(feature_channels.astype(np.float32)).permute(2, 0, 1)
        
        # 应用通用变换
        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        
        return feature_tensor, label

# 训练数据加载器创建函数
def create_domain_adaptation_dataloaders(batch_size=64, val_split=0.1, custom_data_weight=0.7,
                                        feature_extractor=None, custom_dataset_path=None):
    """创建领域自适应数据加载器"""
    # 设置特征提取器
    feature_extractor = feature_extractor or AdvancedFeatureExtractor()
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_mnist = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    
    test_mnist = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    # 自定义变换
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        AdvancedAugmentation(p=0.8),  # 应用高级增强
    ])
    
    # 加载自定义数据集（如果提供）
    custom_dataset = None
    if custom_dataset_path:
        try:
            from test_on_mymnist import CustomTestDataset
            custom_dataset = CustomTestDataset(custom_dataset_path, transform=transform)
            print(f"Loaded custom dataset from {custom_dataset_path} with {len(custom_dataset)} samples")
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            print("Proceeding with only MNIST dataset")
    
    # 分割训练集和验证集
    train_size = int(len(train_mnist) * (1 - val_split))
    val_size = len(train_mnist) - train_size
    train_mnist, val_mnist = random_split(train_mnist, [train_size, val_size])
    
    # 创建领域自适应数据集
    train_dataset = DomainAdaptationDataset(
        train_mnist, custom_dataset, feature_extractor,
        domain_weight=custom_data_weight, transform=None, custom_transform=custom_transform
    )
    
    val_dataset = DomainAdaptationDataset(
        val_mnist, None, feature_extractor,
        domain_weight=0.0, transform=None, custom_transform=None
    )
    
    test_dataset = DomainAdaptationDataset(
        test_mnist, None, feature_extractor,
        domain_weight=0.0, transform=None, custom_transform=None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, feature_extractor

# 主函数
def main():
    """主函数 - 训练和评估增强模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建高级特征提取器
    feature_extractor = AdvancedFeatureExtractor()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, _ = create_domain_adaptation_dataloaders(
        batch_size=64, val_split=0.1, custom_data_weight=0.7,
        feature_extractor=feature_extractor, custom_dataset_path='mymnist/mymnist/all'
    )
    
    # 创建模型目录
    os.makedirs('models/checkpoints', exist_ok=True)
    
    # 创建增强版模型
    model = EnhancedCNN(in_channels=6).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # 训练模型
    model, history = train_enhanced_model(
        model, train_loader, val_loader, device,
        num_epochs=25, lr=0.001, weight_decay=1e-5,
        model_name="enhanced_model_90"
    )
    
    # 保存训练历史
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f"training_history_{timestamp}.npy", history)
    
    print("Training complete!")
    return model, history

if __name__ == "__main__":
    main() 