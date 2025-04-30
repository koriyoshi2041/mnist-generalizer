# MNIST Handwritten Digit Recognition Project

This is a PyTorch-based MNIST handwritten digit recognition project exploring generalization issues between standard datasets and custom datasets.

## Project Background

MNIST is a classic dataset in machine learning, with most models achieving over 98% accuracy on the MNIST test set. However, we found that these models perform significantly worse when facing custom handwritten digits (such as the mymnist dataset).

Our project focuses on:
- Model generalization between MNIST and non-MNIST datasets
- The impact of feature engineering on model performance
- Adjusting feature channel weights for specific dataset adaptation

## Actual Performance Data

- On standard MNIST test set: **~99%** accuracy
- On custom mymnist dataset:
  - Basic weights: **57.5%** accuracy
  - After weight adjustment: **58.8%** accuracy
- Model parameters: **~2M**
- Feature extractor before adjustment: [0.2, 0.2, 0.2, 0.5]
- Feature extractor after adjustment: [0.2, 0.2, 0.2, 0.6]

## Core Technologies

- **Multi-channel Feature Extraction**: Using 4 channels to capture different features
- **Feature Weight Adjustment**: Optimizing feature weights for custom datasets
- **Subtle Feature Enhancement**: Applying very slight enhancements to skeleton and gradient features

## Project Components

- `mnist_custom_model.py`: Feature extraction and model implementation
- `test_on_mymnist.py`: Custom dataset testing script
- `train_models.py`: Training script
- `test_enhanced_features.py`: Feature adjustment testing script

## Feature Channel Description

Our model uses 4 feature channels:

1. **Original Image** (weight: 0.2)
   - Preprocessed original grayscale image

2. **Skeleton Features** (weight: 0.2)
   - Extracting digit skeletons through morphological operations

3. **Regional Density Features** (weight: 0.2)
   - Calculating pixel density in image regions

4. **Gradient Features** (weight: 0.6, previously 0.5)
   - Extracting edges and texture changes
   - Weight adjustment from 0.5 to 0.6 improved accuracy by 1.3%

## Feature Weight Adjustment Process

To improve model performance on custom datasets, we made the following adjustments:

1. **Increased Gradient Feature Weight**: from 0.5 to 0.6
2. **Very Slight Feature Enhancement**:
   - Skeleton features: 98% original + 2% advanced texture
   - Gradient features: 97% original + 3% HOG features

These fine-tuning improved accuracy from 57.5% to 58.8%, a small but noticeable improvement

## Recognition Rate Changes by Digit

Before → After weight adjustment:
- Digit 0: 36% → 37% (+1%)
- Digit 1: 81% → 79% (-2%)
- Digit 2: 57% → 59% (+2%)
- Digit 3: 66% → 67% (+1%)
- Digit 4: 80% → 83% (+3%)
- Digit 5: 56% → 57% (+1%)
- Digit 6: 66% → 65% (-1%)
- Digit 7: 42% → 46% (+4%)
- Digit 8: 52% → 55% (+3%)
- Digit 9: 39% → 40% (+1%)

## Usage Instructions

### 1. Environment Setup

```bash
pip install torch torchvision numpy matplotlib opencv-python scikit-learn scikit-image tqdm seaborn
```

### 2. Testing Model Performance

```bash
python test_enhanced_features.py --model models/model_1_best.pth --model_type advanced --dataset mymnist/mymnist/all --results_dir results/test_results
```

### 3. Training the Model

```bash
python train_models.py --model_type advanced
```

## Observations and Limitations

Despite our feature weight adjustments yielding slight improvements, the accuracy on custom datasets remains far below the performance on MNIST. Possible reasons include:

1. Significant distribution differences between datasets (stroke thickness, background contrast)
2. Limitations of feature engineering
3. Insufficient domain adaptation strategies

We also tested more complex model architectures (5M+ parameters) but did not achieve better results