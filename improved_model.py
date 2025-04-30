import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from mnist_custom_model import FeatureExtractor as BaseFeatureExtractor

class EnhancedFeatureExtractor(BaseFeatureExtractor):
    """增强版特征提取器，调整权重比例"""
    def __init__(self):
        # 调整特征权重 [原图, 笔划特征, 密度特征, 梯度特征]
        weights = [1.0, 0.3, 0.1, 0.6]
        super(EnhancedFeatureExtractor, self).__init__(weights=weights)

# 增强版通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用更大的中间层
        mid_channels = max(8, in_channels // reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 增强版空间注意力
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

# CBAM注意力块
class CBAMBlock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio)
        self.spatialattention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

# 深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 瓶颈残差块
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(BottleneckBlock, self).__init__()
        self.bottleneck_channels = out_channels // expansion
        
        self.conv1 = nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.bottleneck_channels)
        
        self.conv2 = DepthwiseSeparableConv(
            self.bottleneck_channels, self.bottleneck_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        
        self.conv3 = nn.Conv2d(self.bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 跳跃连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

# 困难数字专注模块 - 用于中等大小模型
class MediumDigitModule(nn.Module):
    def __init__(self, in_channels):
        super(MediumDigitModule, self).__init__()
        # 针对困难数字的特征提取 - 中等复杂度
        mid_channels = 96  # 中等通道数
        self.digit_0_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.digit_7_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.digit_8_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.digit_9_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        # 特征集成 - 中等复杂度
        integration_in = mid_channels * 4
        integration_mid = integration_in // 2
        self.integration = nn.Sequential(
            nn.Conv2d(integration_in, integration_mid, kernel_size=1),
            nn.BatchNorm2d(integration_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(integration_mid, in_channels, kernel_size=1),
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
        
        # 应用注意力
        enhanced = x * attention_map
        return x + enhanced  # 残差连接

# 调整模型参数量，降低复杂度以适应内存限制
class OptimizedAdvancedMNISTModel(nn.Module):
    def __init__(self, in_channels=4):
        super(OptimizedAdvancedMNISTModel, self).__init__()
        
        # 减少初始通道数从192到96
        # 第一阶段 - 特征提取
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.stage1_attention = CBAMBlock(96)
        self.stage1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二阶段 - 特征编码
        self.stage2 = nn.Sequential(
            BottleneckBlock(96, 192),
            BottleneckBlock(192, 192),
        )
        self.stage2_attention = CBAMBlock(192)
        self.stage2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三阶段 - 深度特征
        self.stage3 = nn.Sequential(
            BottleneckBlock(192, 384),
            BottleneckBlock(384, 384),
            BottleneckBlock(384, 384),
        )
        self.stage3_attention = CBAMBlock(384)
        self.stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四阶段 - 高级特征
        self.stage4 = nn.Sequential(
            BottleneckBlock(384, 768),
            BottleneckBlock(768, 768),
        )
        self.stage4_attention = CBAMBlock(768)
        
        # 针对困难数字的专用模块 - 降低复杂度
        self.difficult_digit_module = MediumDigitModule(768)
        
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 主分类器 - 减少全连接层节点数
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )
        
        # 数字8专用辅助分类器
        self.digit8_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 二分类：是否为数字8
        )
    
    def forward(self, x):
        # 第一阶段
        x = self.stage1(x)
        x = self.stage1_attention(x)
        x = self.stage1_pool(x)
        
        # 第二阶段
        x = self.stage2(x)
        x = self.stage2_attention(x)
        x = self.stage2_pool(x)
        
        # 第三阶段
        x = self.stage3(x)
        x = self.stage3_attention(x)
        x = self.stage3_pool(x)
        
        # 第四阶段
        x = self.stage4(x)
        x = self.stage4_attention(x)
        
        # 困难数字专用处理
        x = self.difficult_digit_module(x)
        
        # 全局池化
        features = self.avg_pool(x)
        
        # 主分类器
        main_output = self.classifier(features)
        
        # 辅助分类器
        digit8_output = self.digit8_classifier(features)
        
        # 无论是否处于训练模式，都返回两个输出
        return main_output, digit8_output

# 自定义损失函数
class EnhancedCELoss(nn.Module):
    def __init__(self):
        super(EnhancedCELoss, self).__init__()
        # 基础交叉熵损失
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
        # 针对难识别数字设置权重
        self.digit_weights = torch.ones(10)
        self.digit_weights[0] = 1.5  # 数字0
        self.digit_weights[7] = 1.3  # 数字7
        self.digit_weights[8] = 2.0  # 数字8 - 特别加强
        self.digit_weights[9] = 1.5  # 数字9
        
    def forward(self, outputs, targets):
        loss = self.ce(outputs, targets)
        
        # 应用数字特定权重
        weights = self.digit_weights[targets].to(loss.device)
        weighted_loss = loss * weights
        
        return weighted_loss.mean()

# Focal Loss实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets):
        # 标准交叉熵
        ce_loss = self.ce(outputs, targets)
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # Focal Loss公式
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# 混合损失函数
class MixedLoss(nn.Module):
    def __init__(self, ce_weight=0.7, focal_weight=0.3):
        super(MixedLoss, self).__init__()
        self.enhanced_ce = EnhancedCELoss()
        self.focal = FocalLoss()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
    def forward(self, outputs, targets):
        ce_loss = self.enhanced_ce(outputs, targets)
        focal_loss = self.focal(outputs, targets)
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss

# 训练函数
def train_advanced_model(model, train_loader, val_loader, device, num_epochs=20, 
                         lr=0.001, weight_decay=1e-5, model_name="advanced_model"):
    """训练增强版模型"""
    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=lr * 0.01
    )
    
    # 损失函数
    main_criterion = MixedLoss()
    aux_criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'digit_8_acc': []
    }
    
    # 最佳模型跟踪
    best_val_acc = 0.0
    best_digit8_acc = 0.0
    
    # 使用tqdm进度条包装epoch循环
    from tqdm import tqdm
    
    # 训练循环
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 使用tqdm包装训练循环
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        # 训练阶段
        for batch_idx, (inputs, targets) in enumerate(batch_progress):
            # 优化内存使用: 如果是CPU训练，避免不必要的.to(device)
            if str(device) != 'cpu' or inputs.device != device:
                inputs, targets = inputs.to(device), targets.to(device)
            
            # 创建数字8的辅助标签（1表示是8，0表示不是8）
            is_digit8 = (targets == 8).long()
            
            # 前向传播
            optimizer.zero_grad()
            main_outputs, digit8_outputs = model(inputs)
            
            # 计算损失
            main_loss = main_criterion(main_outputs, targets)
            digit8_loss = aux_criterion(digit8_outputs, is_digit8)
            
            # 组合损失
            total_loss = main_loss + 0.3 * digit8_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item() * inputs.size(0)
            _, predicted = main_outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            acc = 100 * predicted.eq(targets).sum().item() / targets.size(0)
            batch_progress.set_postfix({
                'loss': total_loss.item(), 
                'acc': f"{acc:.2f}%"
            })
            
            # 减少日志输出频率，减轻CPU负担
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch: {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {total_loss.item():.4f} - "
                      f"Acc: {100 * predicted.eq(targets).sum().item() / targets.size(0):.2f}%")
            
            # CPU训练时，主动清理内存
            if str(device) == 'cpu' and (batch_idx + 1) % 100 == 0:
                import gc
                gc.collect()
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        digit8_correct = 0
        digit8_total = 0
        
        # 使用tqdm包装验证循环
        val_progress = tqdm(val_loader, desc="验证", leave=False)
        
        with torch.no_grad():
            for inputs, targets in val_progress:
                # 优化内存使用
                if str(device) != 'cpu' or inputs.device != device:
                    inputs, targets = inputs.to(device), targets.to(device)
                is_digit8 = (targets == 8).long()
                
                # 前向传播
                main_outputs, digit8_outputs = model(inputs)  # 现在模型总是返回两个输出
                
                # 计算损失
                main_loss = main_criterion(main_outputs, targets)
                digit8_loss = aux_criterion(digit8_outputs, is_digit8)
                total_loss = main_loss + 0.3 * digit8_loss
                
                # 统计
                val_loss += total_loss.item() * inputs.size(0)
                _, predicted = main_outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # 数字8的表现统计
                digit8_mask = (targets == 8)
                if digit8_mask.sum() > 0:
                    digit8_total += digit8_mask.sum().item()
                    digit8_correct += predicted[digit8_mask].eq(targets[digit8_mask]).sum().item()
                
                # 更新进度条
                val_progress.set_postfix({
                    'loss': total_loss.item(),
                    'acc': f"{100 * predicted.eq(targets).sum().item() / targets.size(0):.2f}%"
                })
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        digit8_acc = digit8_correct / digit8_total if digit8_total > 0 else 0.0
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['digit_8_acc'].append(digit8_acc)
        
        # 输出进度
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
              f"Digit 8 Acc: {digit8_acc:.4f}")
        
        # 保存最佳模型
        combined_score = val_acc * 0.7 + digit8_acc * 0.3
        if combined_score > (best_val_acc * 0.7 + best_digit8_acc * 0.3):
            best_val_acc = val_acc
            best_digit8_acc = digit8_acc
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"Saved best model with val_acc={val_acc:.4f}, digit_8_acc={digit8_acc:.4f}")
        
        # CPU训练时每个epoch后主动清理内存
        if str(device) == 'cpu':
            import gc
            gc.collect()
        
        print("=" * 50)
    
    # 保存最终模型
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")
    print(f"Model saved: models/{model_name}_final.pth")
    
    return model, history

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 中等复杂度MNIST模型 (目标10-12M参数)
class MediumMNISTModel(nn.Module):
    def __init__(self, in_channels=4):
        super(MediumMNISTModel, self).__init__()
        
        # 第一阶段 - 特征提取
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage1_attention = CBAMBlock(128)
        self.stage1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二阶段 - 特征编码
        self.stage2 = nn.Sequential(
            BottleneckBlock(128, 256),
            BottleneckBlock(256, 256),
            BottleneckBlock(256, 256),
        )
        self.stage2_attention = CBAMBlock(256)
        self.stage2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三阶段 - 深度特征
        self.stage3 = nn.Sequential(
            BottleneckBlock(256, 512),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 512),
            BottleneckBlock(512, 512),
        )
        self.stage3_attention = CBAMBlock(512)
        self.stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四阶段 - 高级特征
        self.stage4 = nn.Sequential(
            BottleneckBlock(512, 1024),
            BottleneckBlock(1024, 1024),
            BottleneckBlock(1024, 1024),
        )
        self.stage4_attention = CBAMBlock(1024)
        
        # 针对困难数字的专用模块
        self.difficult_digit_module = MediumDigitModule(1024)
        
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 主分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1536, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, 10)
        )
        
        # 数字8专用辅助分类器
        self.digit8_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(384, 2)  # 二分类：是否为数字8
        )
    
    def forward(self, x):
        # 第一阶段
        x = self.stage1(x)
        x = self.stage1_attention(x)
        x = self.stage1_pool(x)
        
        # 第二阶段
        x = self.stage2(x)
        x = self.stage2_attention(x)
        x = self.stage2_pool(x)
        
        # 第三阶段
        x = self.stage3(x)
        x = self.stage3_attention(x)
        x = self.stage3_pool(x)
        
        # 第四阶段
        x = self.stage4(x)
        x = self.stage4_attention(x)
        
        # 困难数字专用处理
        x = self.difficult_digit_module(x)
        
        # 全局池化
        features = self.avg_pool(x)
        
        # 主分类器
        main_output = self.classifier(features)
        
        # 辅助分类器
        digit8_output = self.digit8_classifier(features)
        
        # 无论是否处于训练模式，都返回两个输出
        return main_output, digit8_output

if __name__ == "__main__":
    # 创建模型实例进行参数统计
    model = MediumMNISTModel()
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} parameters") 