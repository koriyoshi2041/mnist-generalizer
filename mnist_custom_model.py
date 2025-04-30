import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 固定随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# 1. 特征工程函数
class FeatureExtractor:
    def __init__(self, weights=None):
        """
        初始化特征提取器，支持为不同特征通道设置权重
        
        参数:
        - weights: 四个特征通道的权重列表 [原始图像权重, 骨架图权重, 区域密度图权重, 梯度图权重]
                   如果为None，则使用默认权重 [0.2, 0.2, 0.2, 0.5]
        """
        # 为不同特征通道设置权重，增加梯度特征的权重
        self.weights = weights if weights is not None else [0.2, 0.2, 0.2, 0.5]
    
    def preprocess(self, image):
        """基础图像预处理"""
        if isinstance(image, torch.Tensor):
            image = image.numpy().squeeze()
        
        # 确保图像是2D的
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 标准化到0-1范围
        if image.max() > 1.0:
            image = image / 255.0
            
        # 如果需要反色（MNIST数字是白底黑字）
        if np.mean(image) > 0.5:
            image = 1 - image
        
        return image
    
    def extract_morphological_features(self, image):
        """提取形态学特征"""
        # 确保图像已预处理
        image = self.preprocess(image)
        
        # 二值化
        binary = (image > 0.5).astype(np.uint8)
        
        # 骨架提取
        skeleton = skeletonize(binary).astype(np.uint8)
        
        # 计算连通区域
        num_labels, labels = cv2.connectedComponents(binary)
        
        # 计算轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建特征图像
        edge_image = cv2.Canny(binary, 50, 150)
        
        return {
            'skeleton': skeleton,
            'edge': edge_image,
            'num_components': num_labels - 1  # 减去背景
        }
    
    def extract_regional_features(self, image):
        """提取区域特征"""
        # 确保图像已预处理
        image = self.preprocess(image)
        h, w = image.shape
        
        # 将图像分为3×3区域
        cell_h, cell_w = h // 3, w // 3
        density_map = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                region = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                density_map[i, j] = np.mean(region)
        
        # 创建密度图像
        density_image = np.repeat(density_map, cell_h, axis=0)
        density_image = np.repeat(density_image, cell_w, axis=1)
        
        # 确保正确的尺寸
        density_image = cv2.resize(density_image, (w, h))
        
        return density_image
    
    def extract_texture_features(self, image):
        """提取纹理特征"""
        # 确保图像已预处理
        image = self.preprocess(image)
        
        # 确保图像是正确的类型和范围
        # 转换为8位无符号整数类型，然后再转回float32用于计算
        image_uint8 = (image * 255).astype(np.uint8)
        
        # 计算局部二值模式(LBP)
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        # 归一化到0-1
        lbp = (lbp / lbp.max())
        
        # 梯度特征 - 使用8位图像进行Sobel处理
        gx = cv2.Sobel(image_uint8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image_uint8, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = magnitude / (magnitude.max() + 1e-8)
        
        return {
            'lbp': lbp,
            'gradient_magnitude': magnitude
        }
    
    def create_feature_channels(self, image, verbose=False):
        """创建多通道特征图像，并应用特征权重"""
        try:
            if verbose:
                print("Creating feature channels...")
            
            # 基础预处理
            if verbose:
                print("  Preprocessing image...")
            processed = self.preprocess(image)
            if verbose:
                print(f"  Preprocessed image shape: {processed.shape}")
            
            # 提取各类特征
            if verbose:
                print("  Extracting morphological features...")
            morph_features = self.extract_morphological_features(processed)
            if verbose:
                print(f"  Morphological features extracted: {list(morph_features.keys())}")
            
            if verbose:
                print("  Extracting regional features...")
            regional_features = self.extract_regional_features(processed)
            if verbose:
                print(f"  Regional features shape: {regional_features.shape}")
            
            if verbose:
                print("  Extracting texture features...")
            texture_features = self.extract_texture_features(processed)
            if verbose:
                print(f"  Texture features extracted: {list(texture_features.keys())}")
            
            # 创建4通道图像并应用权重
            channels = [
                processed * self.weights[0],  # 原始预处理图像
                morph_features['skeleton'] * self.weights[1],  # 骨架图
                regional_features * self.weights[2],  # 区域密度图
                texture_features['gradient_magnitude'] * self.weights[3]  # 梯度幅值图
            ]
            
            # 堆叠通道
            if verbose:
                print("  Stacking channels...")
            multi_channel = np.stack(channels, axis=-1)
            if verbose:
                print(f"  Final multi-channel feature shape: {multi_channel.shape}")
            
            return multi_channel
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            import traceback
            traceback.print_exc()
            raise

# 自定义Dataset类
class EnhancedMNISTDataset(Dataset):
    def __init__(self, dataset, feature_extractor=None, transform=None, is_train=True):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 应用特征提取
        if self.feature_extractor is not None:
            # 将图像转为numpy数组
            image_np = np.array(image)
            # 提取特征通道
            feature_channels = self.feature_extractor.create_feature_channels(image_np)
            # 转回Tensor
            image = torch.FloatTensor(feature_channels).permute(2, 0, 1)  # [C, H, W]
        
        # 应用数据增强
        if self.transform is not None and self.is_train:
            image = self.transform(image)
        
        return image, label

# 自定义测试数据集
class CustomTestDataset(Dataset):
    def __init__(self, directory, feature_extractor=None, transform=None):
        self.directory = directory
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory)
                      if img.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        # 从文件名获取标签 (格式: 数字_子集_序号.png)
        label = int(os.path.basename(image_path).split('_')[0])
        
        # 加载图像
        image = Image.open(image_path).convert('L')  # 转为灰度图
        
        # 应用特征提取
        if self.feature_extractor is not None:
            image_np = np.array(image)
            feature_channels = self.feature_extractor.create_feature_channels(image_np)
            image = torch.FloatTensor(feature_channels).permute(2, 0, 1)  # [C, H, W]
        else:
            # 转为Tensor
            image = transforms.ToTensor()(image)
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

# 2. 模型定义
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积模块，用于减少模型参数量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SqueezeExcitation(nn.Module):
    """通道注意力模块，增强有用特征，抑制不重要特征"""
    def __init__(self, channel, reduction=8):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LightweightMNISTModel(nn.Module):
    """轻量级MNIST识别模型，使用深度可分离卷积和注意力机制"""
    def __init__(self, in_channels=4, dropout_rate=0.2, use_se=True):
        super(LightweightMNISTModel, self).__init__()
        
        self.use_se = use_se
        
        # 第一层卷积 - 标准卷积
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 深度可分离卷积层
        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 注意力模块
        if self.use_se:
            self.se2 = SqueezeExcitation(32)
            self.se3 = SqueezeExcitation(64)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        # 第一层卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 第二层卷积块 + SE
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_se:
            x = self.se2(x)
        
        # 第三层卷积块 + SE
        x = F.relu(self.bn3(self.conv3(x)))
        if self.use_se:
            x = self.se3(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def extract_edge_features(image):
    """提取边缘增强特征"""
    # 转为灰度图并标准化
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 使用Sobel算子提取梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 归一化到0-1
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    return magnitude

def extract_texture_features(image, radius=2, n_points=16):
    """提取纹理特征 - 局部二值模式(LBP)"""
    # 转为灰度图并标准化
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 计算LBP特征
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # 归一化到0-1
    lbp = lbp / lbp.max()
    
    return lbp

def preprocess_image(image, size=(28, 28)):
    """将输入图像预处理为模型输入的特征通道"""
    # 确保图像是灰度的
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # 调整大小
    if gray.shape[0] != size[0] or gray.shape[1] != size[1]:
        gray = cv2.resize(gray, size)
    
    # 归一化原始灰度图像
    normalized = gray / 255.0
    
    # 提取边缘特征
    edge_features = extract_edge_features(gray)
    
    # 提取纹理特征
    texture_features = extract_texture_features(gray)
    
    # 形态学处理后的特征
    kernel = np.ones((2, 2), np.uint8)
    morph_features = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) / 255.0
    
    # 合并特征通道
    feature_channels = np.stack([normalized, edge_features, texture_features, morph_features], axis=0)
    
    return feature_channels

def create_data_transforms():
    """创建训练和测试的数据变换"""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return train_transform, test_transform

def tensor_to_features(tensor):
    """将PyTorch张量转换为特征通道"""
    # 假设输入是[1, 28, 28]的MNIST图像
    image = tensor.squeeze().numpy() * 255.0
    image = image.astype(np.uint8)
    
    # 提取特征通道
    feature_channels = preprocess_image(image)
    
    # 转回PyTorch张量 [4, 28, 28]
    return torch.from_numpy(feature_channels).float()

def create_feature_model(model_variant=0, dropout_rate=0.2):
    """创建不同变体的模型"""
    if model_variant == 0:
        # 基础模型
        return LightweightMNISTModel(in_channels=4, dropout_rate=dropout_rate, use_se=True)
    elif model_variant == 1:
        # 没有SE模块的变体
        return LightweightMNISTModel(in_channels=4, dropout_rate=dropout_rate, use_se=False)
    elif model_variant == 2:
        # 更高的dropout率
        return LightweightMNISTModel(in_channels=4, dropout_rate=0.3, use_se=True)
    elif model_variant == 3:
        # 仅使用3个输入通道（不使用形态学特征）
        return LightweightMNISTModel(in_channels=3, dropout_rate=dropout_rate, use_se=True)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

def count_parameters(model):
    """计算模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 3. 数据增强函数
def get_train_transforms():
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

# 4. 训练函数
def train_model(model, train_loader, val_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best model with validation accuracy: {best_val_acc:.2f}%')
        
        # 学习率调度
        scheduler.step(val_loss)
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    return model

# 5. 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 6. 集成模型函数
def ensemble_predict(models, test_loader, device):
    all_predictions = []
    true_labels = []
    
    # 获取每个模型的预测
    for model in models:
        model.eval()
        model_predictions = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                model_predictions.append(probs.cpu().numpy())
                
                # 只需要在第一个模型时收集真实标签
                if len(all_predictions) == 0:
                    true_labels.append(labels.cpu().numpy())
        
        # 将当前模型的所有批次预测合并
        model_predictions = np.vstack([p for p in model_predictions])
        all_predictions.append(model_predictions)
    
    # 合并真实标签
    true_labels = np.concatenate(true_labels)
    
    # 平均所有模型的预测概率
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # 计算准确率
    accuracy = np.mean(ensemble_preds == true_labels) * 100
    print(f'Ensemble Test Accuracy: {accuracy:.2f}%')
    return accuracy

def trim_image(image_np, padding=2):
    """
    裁剪图像周围的空白区域，使数字居中并放大
    
    参数:
    - image_np: 输入图像的numpy数组
    - padding: 在裁剪区域周围保留的额外像素数
    
    返回:
    - trimmed_image: 裁剪和缩放后的图像
    """
    # 确保图像是灰度的
    if len(image_np.shape) > 2:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    
    # 如果需要，反转图像（确保数字是黑底白字）
    if np.mean(gray) < 0.5:  # 如果平均值小于0.5，图像是暗色背景
        gray = 1 - gray
    
    # 二值化找到非零区域
    _, thresh = cv2.threshold(np.uint8(gray * 255), 20, 255, cv2.THRESH_BINARY)
    
    # 找到非零区域的边界框
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # 如果没有非零区域，返回原始图像
        return cv2.resize(image_np, (28, 28))
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # 添加padding，但确保不超出图像边界
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image_np.shape[1] - x, w + 2 * padding)
    h = min(image_np.shape[0] - y, h + 2 * padding)
    
    # 裁剪图像
    cropped = image_np[y:y+h, x:x+w]
    
    # 创建一个正方形画布，保持数字的宽高比
    max_dim = max(w, h)
    square = np.zeros((max_dim, max_dim), dtype=image_np.dtype)
    
    # 计算居中位置
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    
    # 将裁剪后的图像放入正方形画布
    square[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
    
    # 缩放到28x28
    return cv2.resize(square, (28, 28))

# 7. 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor()
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])
    
    # 创建增强数据集
    train_transforms = get_train_transforms()
    
    enhanced_train = EnhancedMNISTDataset(mnist_train, feature_extractor, train_transforms, is_train=True)
    enhanced_val = EnhancedMNISTDataset(mnist_val, feature_extractor, None, is_train=False)
    enhanced_test = EnhancedMNISTDataset(mnist_test, feature_extractor, None, is_train=False)
    
    # 创建数据加载器
    train_loader = DataLoader(enhanced_train, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(enhanced_val, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(enhanced_test, batch_size=128, shuffle=False, num_workers=4)
    
    # 创建自定义测试集
    custom_test = CustomTestDataset('./mymnist/mymnist/all', feature_extractor)
    custom_test_loader = DataLoader(custom_test, batch_size=128, shuffle=False, num_workers=4)
    
    # 训练多个模型
    num_models = 4
    models = []
    
    for i in range(num_models):
        print(f"\n======= 训练模型 {i+1}/{num_models} =======")
        model = LightweightMNISTModel(in_channels=4).to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 使用不同的随机种子
        set_seed(42 + i*10)
        
        # 训练模型
        trained_model = train_model(model, train_loader, val_loader, device, num_epochs=15)
        models.append(trained_model)
        
        # 评估单个模型
        print(f"\n模型 {i+1} MNIST测试集评估:")
        mnist_acc = evaluate_model(trained_model, test_loader, device)
        
        print(f"\n模型 {i+1} 自定义测试集评估:")
        custom_acc = evaluate_model(trained_model, custom_test_loader, device)
        
        # 保存模型
        torch.save(trained_model.state_dict(), f'model_{i+1}.pth')
    
    # 模型集成
    print("\n======= 集成模型评估 =======")
    print("MNIST测试集:")
    ensemble_mnist_acc = ensemble_predict(models, test_loader, device)
    
    print("\n自定义测试集:")
    ensemble_custom_acc = ensemble_predict(models, custom_test_loader, device)
    
    # 打印最终结果
    print("\n======= 最终评估结果 =======")
    print(f"MNIST测试集准确率: {ensemble_mnist_acc:.2f}%")
    print(f"自定义测试集准确率: {ensemble_custom_acc:.2f}%")

if __name__ == "__main__":
    main() 