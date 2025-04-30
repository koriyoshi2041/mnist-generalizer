import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import cv2

from enhanced_model_90 import AdvancedFeatureExtractor
from test_on_mymnist import load_model, predict

# 图像处理函数
def process_image(image_path):
    """处理图像文件，返回预处理后的图像"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 确保图像大小为28x28
    if image.shape != (28, 28):
        image = cv2.resize(image, (28, 28))
    
    # 归一化到0-1范围
    image = image.astype(np.float32) / 255.0
    
    # 反色（如果需要）- 确保数字是白色，背景是黑色
    if np.mean(image) > 0.5:
        image = 1.0 - image
    
    return image

def test_with_advanced_features(model_path, dataset_path, results_dir, model_type="advanced"):
    """使用高级特征提取器测试模型在自定义数据集上的表现"""
    print(f"测试模型: {model_path}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(model_path, model_type, device)
    print("模型加载成功")
    
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 创建高级特征提取器
    print("使用高级特征提取器...")
    extractor = AdvancedFeatureExtractor()
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取所有测试图像
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 初始化存储结果的变量
    predictions = []
    true_labels = []
    confidences = []
    digit_8_confidences = []
    digit_8_predictions = []
    
    # 处理每个图像
    progress_bar = tqdm(image_files, desc="处理图像")
    for image_file in progress_bar:
        # 从文件名中获取真实标签
        true_label = int(image_file.split('_')[0])
        
        # 读取并处理图像
        image_path = os.path.join(dataset_path, image_file)
        try:
            # 处理图像并提取特征
            processed_image = process_image(image_path)
            feature_image = extractor.create_feature_channels(processed_image)
            feature_tensor = torch.from_numpy(feature_image.astype(np.float32)).permute(2, 0, 1)
            
            # 使用模型预测
            prediction, confidence, probs = predict(model, feature_tensor.unsqueeze(0), device)
            
            # 存储结果
            predictions.append(prediction)
            true_labels.append(true_label)
            confidences.append(confidence)
            
            # 存储数字8的相关信息
            if true_label == 8:
                digit_8_confidences.append(confidence)
                digit_8_predictions.append(prediction)
                
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {e}")
            continue
    
    # 计算总体准确率
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    accuracy = correct / len(true_labels) if len(true_labels) > 0 else 0
    print(f"\n总体准确率: {accuracy*100:.2f}% ({correct}/{len(true_labels)})")
    
    # 计算各个数字的准确率
    digit_accuracies = {}
    for digit in range(10):
        digit_indices = [i for i, label in enumerate(true_labels) if label == digit]
        if len(digit_indices) > 0:
            digit_correct = sum(1 for i in digit_indices if predictions[i] == true_labels[i])
            digit_accuracies[digit] = digit_correct / len(digit_indices)
            print(f"  数字 {digit}: {digit_accuracies[digit]*100:.2f}% ({digit_correct}/{len(digit_indices)})")
    
    # 保存各个数字的准确率
    with open(f"{results_dir}/digit_accuracies.txt", "w") as f:
        f.write("Digit Accuracies:\n")
        for digit in range(10):
            if digit in digit_accuracies:
                f.write(f"Digit {digit}: {digit_accuracies[digit]*100:.2f}%\n")
        f.write(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")
    
    # 打印详细分类报告
    print("\n详细分类报告:")
    report = classification_report(true_labels, predictions)
    print(report)
    
    # 保存分类报告
    with open(f"{results_dir}/classification_report.txt", "w") as f:
        f.write(report)
    
    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 保存混淆矩阵
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f"{results_dir}/confusion_matrix_{timestamp}.png")
    plt.close()
    
    # 绘制各数字准确率
    plt.figure(figsize=(12, 6))
    digits = list(digit_accuracies.keys())
    accs = [digit_accuracies[d] * 100 for d in digits]
    plt.bar(digits, accs)
    plt.axhline(y=accuracy*100, color='r', linestyle='-', label=f'Overall: {accuracy*100:.2f}%')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Digit')
    plt.xticks(range(10))
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(f"{results_dir}/digit_accuracies_{timestamp}.png")
    plt.close()
    
    # 分析数字8的预测
    if 8 in digit_accuracies:
        # 绘制数字8的预测分布
        plt.figure(figsize=(10, 6))
        digit_8_pred_counts = {}
        for pred in digit_8_predictions:
            digit_8_pred_counts[pred] = digit_8_pred_counts.get(pred, 0) + 1
        
        pred_digits = list(range(10))
        counts = [digit_8_pred_counts.get(d, 0) for d in pred_digits]
        plt.bar(pred_digits, counts)
        plt.xlabel('Predicted as')
        plt.ylabel('Count')
        plt.title('Distribution of Digit 8 Predictions')
        plt.xticks(range(10))
        plt.savefig(f"{results_dir}/digit_8_distribution_{timestamp}.png")
        plt.close()
        
        # 绘制数字8的置信度分布
        plt.figure(figsize=(10, 6))
        plt.hist(digit_8_confidences, bins=20, alpha=0.7)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution for Digit 8')
        plt.savefig(f"{results_dir}/digit_8_confidence_{timestamp}.png")
        plt.close()
    
    print(f"测试完成，结果保存在 {results_dir} 目录")
    
    return accuracy, digit_accuracies

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用高级特征提取器测试MNIST模型")
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--model_type', type=str, default='advanced', choices=['advanced', 'medium', 'light'], help='模型类型')
    parser.add_argument('--dataset', type=str, default='mymnist/mymnist/all', help='测试数据集路径')
    parser.add_argument('--results_dir', type=str, default='mymnist_results/advanced_features_test', help='结果输出目录')
    
    args = parser.parse_args()
    
    # 测试模型
    test_with_advanced_features(args.model, args.dataset, args.results_dir, args.model_type)

if __name__ == "__main__":
    main() 