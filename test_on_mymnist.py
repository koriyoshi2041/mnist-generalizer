import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import argparse
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn

# 导入特征提取器和模型类
from improved_model import MediumMNISTModel, EnhancedFeatureExtractor
from mnist_custom_model import FeatureExtractor

# 添加一个简单模型用于加载model_2和model_3
class SimpleModel(nn.Module):
    def __init__(self, in_channels=4):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def preprocess_image(image_path, extractor):
    """预处理图像以适应模型输入"""
    # 加载图像
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    
    # 提取特征通道
    feature_channels = extractor.create_feature_channels(image_np)
    
    # 转换为Tensor并调整通道顺序
    tensor_image = torch.FloatTensor(feature_channels).permute(2, 0, 1).unsqueeze(0)
    
    return tensor_image, image_np

def load_model(model_path, model_type="medium", device=None):
    """加载训练好的模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查模型文件名来判断使用哪种模型结构
    model_filename = os.path.basename(model_path)
    
    if model_type.lower() == "medium":
        model = MediumMNISTModel().to(device)
    elif model_type.lower() == "advanced" or "model_1" in model_filename:
        # 使用train_models.py中定义的AdvancedCNN
        from train_models import AdvancedCNN
        model = AdvancedCNN().to(device)
    elif "model_2" in model_filename or "model_3" in model_filename:
        # 使用简单模型加载model_2和model_3
        model = SimpleModel().to(device)
    elif model_type.lower() == "light":
        # 使用train_models.py中定义的LightCNN
        from train_models import LightCNN
        model = LightCNN().to(device)
    else:
        # 默认使用轻量级模型
        from train_models import LightCNN
        model = LightCNN().to(device)
    
    # 加载状态字典并设置为评估模式
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, image_tensor, device):
    """使用模型进行预测"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 处理不同模型类型的输出
        outputs = model(image_tensor)
        if isinstance(outputs, tuple):
            # 如果模型返回多个输出（例如MediumMNISTModel返回主输出和辅助输出）
            outputs = outputs[0]  # 使用主输出
        
        probs = torch.softmax(outputs, dim=1)
        
    # 获取最高概率的预测
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0, prediction].item()
    
    return prediction, confidence, probs.cpu().numpy()[0]

def test_on_dataset(folder_path, model, device, extractor=None, output_dir='mymnist_results'):
    """在指定数据集上测试模型"""
    # 创建特征提取器
    if extractor is None:
        extractor = FeatureExtractor()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PNG图像
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # 用于存储结果的列表
    results = []
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 使用tqdm添加进度条
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(folder_path, image_file)
        
        # 从文件名获取真实标签
        try:
            # 假设文件名格式为: [数字]_子集_序号.png
            true_label = int(image_file.split('_')[0])
        except:
            # 如果无法解析标签，设为None
            true_label = None
            print(f"警告: 无法从 {image_file} 解析标签")
            continue
        
        # 预处理图像
        image_tensor, image_np = preprocess_image(image_path, extractor)
        
        # 预测
        prediction, confidence, probs = predict(model, image_tensor, device)
        
        # 确定是否正确预测
        is_correct = (true_label == prediction)
        
        # 收集结果
        result = {
            'file': image_file,
            'true_label': true_label,
            'prediction': prediction,
            'confidence': confidence,
            'correct': is_correct
        }
        
        results.append(result)
        all_predictions.append(prediction)
        all_true_labels.append(true_label)
        all_confidences.append(confidence)
    
    # 分析结果
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        
        accuracy = correct_count / total_count * 100
        print(f"\n总体准确率: {accuracy:.2f}% ({correct_count}/{total_count})")
        
        # 计算各个数字的准确率
        digit_counts = {}
        digit_correct = {}
        
        for r in results:
            digit = r['true_label']
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
            if r['correct']:
                digit_correct[digit] = digit_correct.get(digit, 0) + 1
        
        print("\n各个数字的准确率:")
        for digit in sorted(digit_counts.keys()):
            correct = digit_correct.get(digit, 0)
            total = digit_counts[digit]
            print(f"  数字 {digit}: {correct/total*100:.2f}% ({correct}/{total})")
        
        # 生成分类报告
        print("\n详细分类报告:")
        report = classification_report(all_true_labels, all_predictions, digits=4)
        print(report)
        
        # 保存结果图表
        visualize_results(all_predictions, all_true_labels, all_confidences, output_dir)
    
    return results

def visualize_results(predictions, true_labels, confidences, output_dir):
    """对测试结果进行可视化"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_dir}/confusion_matrix_{timestamp}.png")
    plt.close()
    
    # 2. 各数字的准确率
    digit_accuracies = np.zeros(10)
    for digit in range(10):
        digit_indices = np.where(np.array(true_labels) == digit)[0]
        if len(digit_indices) > 0:
            digit_preds = np.array(predictions)[digit_indices]
            digit_correct = np.sum(digit_preds == digit)
            digit_accuracies[digit] = digit_correct / len(digit_indices) * 100
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), digit_accuracies)
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.title('Recognition Accuracy for Each Digit')
    plt.xticks(range(10))
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # 在图表中显示具体数值
    for i, acc in enumerate(digit_accuracies):
        plt.text(i, acc + 2, f"{acc:.1f}%", ha='center')
    
    plt.savefig(f"{output_dir}/digit_accuracies_{timestamp}.png")
    plt.close()
    
    # 将数字准确率保存到文本文件
    with open(f"{output_dir}/digit_accuracies.txt", "w") as f:
        f.write("Digit Accuracies:\n")
        for digit, acc in enumerate(digit_accuracies):
            f.write(f"Digit {digit}: {acc:.2f}%\n")
        f.write(f"\nOverall Accuracy: {np.mean(digit_accuracies):.2f}%\n")
    
    # 3. 聚焦数字8
    digit_8_indices = np.where(np.array(true_labels) == 8)[0]
    if len(digit_8_indices) > 0:
        digit_8_preds = np.array(predictions)[digit_8_indices]
        digit_8_conf = np.array(confidences)[digit_8_indices]
        
        # 创建数字8的混淆矩阵
        digit_8_cm = np.zeros(10)
        for pred in digit_8_preds:
            digit_8_cm[pred] += 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), digit_8_cm)
        plt.xlabel('Predicted as Digit')
        plt.ylabel('Count')
        plt.title('Distribution of Digit 8 Recognition')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/digit_8_distribution_{timestamp}.png")
        plt.close()
        
        # 数字8的正确和错误预测的置信度分布
        digit_8_correct = digit_8_preds == 8
        
        plt.figure(figsize=(10, 6))
        if np.any(digit_8_correct):
            plt.hist(digit_8_conf[digit_8_correct], bins=20, alpha=0.5, label='Correct Predictions', color='green')
        if np.any(~digit_8_correct):
            plt.hist(digit_8_conf[~digit_8_correct], bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution for Digit 8 Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/digit_8_confidence_{timestamp}.png")
        plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="在自定义MNIST数据集上测试模型")
    parser.add_argument('--model', type=str, default='models/medium_model_1_best.pth', help='模型文件路径')
    parser.add_argument('--model_type', type=str, default='medium', choices=['medium', 'advanced', 'light'], help='模型类型')
    parser.add_argument('--dataset', type=str, default='mymnist/mymnist/all', help='测试数据集路径')
    parser.add_argument('--output', type=str, default='mymnist_results', help='结果输出目录')
    parser.add_argument('--feature_type', type=str, default='enhanced', choices=['enhanced', 'basic'], help='特征提取器类型')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建特征提取器
    if args.feature_type == 'enhanced':
        print("使用增强版特征提取器...")
        extractor = EnhancedFeatureExtractor()
    else:
        print("使用基础特征提取器...")
        extractor = FeatureExtractor()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_model(args.model, args.model_type, device)
    
    # 测试模型
    print(f"在数据集 {args.dataset} 上测试模型...")
    results = test_on_dataset(args.dataset, model, device, extractor, args.output)
    print(f"测试完成，结果保存在 {args.output} 目录")

if __name__ == "__main__":
    main() 