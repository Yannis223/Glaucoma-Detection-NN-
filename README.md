# Glaucoma Detection Using Deep Learning

A comprehensive deep learning project for automated glaucoma detection from retinal fundus images using CNN, VGG16, and MobileNetV2 architectures.

##  Overview

This project implements and compares three deep learning models for binary classification of retinal images to detect glaucoma. The models achieve high accuracy through transfer learning, fine-tuning, and K-Fold cross-validation.

**Dataset**: 598 retinal fundus images (299 glaucoma, 299 normal)

**Best Performance**: Custom CNN - 98% accuracy

##  Features

- **Three Model Architectures**:
  - Custom CNN (built from scratch)
  - VGG16 (transfer learning)
  - MobileNetV2 (transfer learning)

- **Advanced Training Techniques**:
  - Transfer learning with ImageNet weights
  - Fine-tuning with unfrozen layers
  - K-Fold cross-validation (5 folds)
  - Early stopping to prevent overfitting
  - Data normalization and augmentation

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - ROC curves and AUC scores
  - Statistical comparison (Kruskal-Wallis test)

##  Quick Start

### Prerequisites

```bash
Python 3.7+
Google Colab (recommended) or local GPU environment
```

### Required Libraries

```bash
tensorflow >= 2.0
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
graphviz
scipy
```

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd glaucoma-detection
```

2. **Install dependencies**:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python graphviz scipy
```

3. **Prepare your dataset**:
   - Organize images in two folders: `glaucoma/` and `normal/`
   - Compress into a ZIP file
   - Upload to Google Colab or local directory

## 📁 Project Structure

```
glaucoma-detection/
│
├── NeuralNetworksGlaucoma.ipynb    # Main Jupyter notebook
├── README.md                        # This file
│
├── data/
│   ├── raw_images/                 # Original images (extracted)
│   │   ├── glaucoma/
│   │   └── normal/
│   └── resized_images/             # Preprocessed images (224x224)
│       ├── glaucoma/
│       └── normal/
│
└── outputs/
    ├── models/                     # Trained model weights
    ├── plots/                      # Visualization outputs
    └── architectures/              # Model architecture diagrams
```

##  Methodology

### 1. Data Preprocessing
- **Image Resizing**: All images resized to 224×224 pixels
- **Normalization**: Pixel values scaled to [0, 1]
- **Data Split**: 
  - Training: 80%
  - Validation: 10%
  - Test: 10%

### 2. Model Architectures

#### Custom CNN
```
Input (224×224×3)
→ Conv2D(32) + BatchNorm + MaxPooling
→ Conv2D(64) + BatchNorm + MaxPooling
→ Conv2D(128) + BatchNorm + MaxPooling
→ Conv2D(256) + BatchNorm + MaxPooling
→ Flatten
→ Dense(256) + Dropout(0.5)
→ Output (Sigmoid)
```

#### VGG16 (Transfer Learning)
```
Input (224×224×3)
→ VGG16 Base (pretrained, frozen)
→ GlobalAveragePooling2D
→ Dense(128) + Dropout(0.5)
→ Output (Sigmoid)
```

#### MobileNetV2 (Transfer Learning)
```
Input (224×224×3)
→ MobileNetV2 Base (pretrained, frozen)
→ GlobalAveragePooling2D
→ Dense(128) + Dropout(0.4)
→ Output (Sigmoid)
```

### 3. Training Process

**Initial Training**:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Batch Size: 16
- Early Stopping: Patience 3-5 epochs

**Fine-Tuning**:
- Unfroze last layers of pretrained models
- Lower learning rates (1e-5 to 1e-6)
- Batch Size: 32
- Extended training epochs

**K-Fold Validation**:
- 5-fold cross-validation
- Stratified splits
- Performance averaging across folds

##  Results

### Model Comparison

| Model | Accuracy | F1-Score | Avg Recall | AUC |
|-------|----------|----------|------------|-----|
| **Custom CNN** | **98.0%** | **0.98** | **0.985** | **0.99** |
| VGG16 | 95.0% | 0.95 | 0.950 | 0.97 |
| MobileNetV2 | 92.0% | 0.92 | 0.915 | 0.95 |

### K-Fold Cross-Validation Results

- **CNN**: 98.0% ± 0.015
- **VGG16**: 95.0% ± 0.020
- **MobileNetV2**: 92.0% ± 0.025

### Statistical Analysis

**Kruskal-Wallis H-Test**:
- Statistic: 6.8571
- p-value: 0.0324 (< 0.05)
- **Conclusion**: Significant difference between models; CNN performs significantly better

## 📈 Visualizations

The project generates the following visualizations:

1. **Sample Images**: Visual examples from each class
2. **Pixel Intensity Histograms**: Distribution analysis
3. **Training Curves**: Accuracy and loss over epochs
4. **Confusion Matrices**: Classification performance
5. **ROC Curves**: True positive vs false positive rates
6. **Model Architectures**: Visual flowcharts of network structures
7. **Comparative Bar Charts**: Side-by-side performance metrics

##  Usage

### Running the Complete Pipeline

```python
# 1. Upload and extract dataset
from google.colab import files
uploaded = files.upload()

# 2. Run preprocessing
# (Code automatically resizes and organizes images)

# 3. Train models
history_cnn = model_cnn.fit(X_train, y_train, ...)
history_vgg = model_vgg.fit(X_train, y_train, ...)
history_mobilenet = model_mobilenet.fit(X_train, y_train, ...)

# 4. Evaluate models
test_loss, test_acc = model_cnn.evaluate(X_test, y_test)

# 5. Fine-tune models
# (Unfreeze layers and retrain with lower learning rate)

# 6. K-Fold validation
# (Run cross-validation for robust performance estimation)
```

### Making Predictions

```python
# Load a single image
import cv2
img = cv2.imread('path/to/image.jpg')
img = cv2.resize(img, (224, 224)) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model_cnn.predict(img)
result = "Glaucoma" if prediction[0][0] < 0.5 else "Normal"
print(f"Prediction: {result} (confidence: {prediction[0][0]:.2%})")
```

##  Key Findings

1. **Custom CNN outperformed transfer learning models** for this specific dataset
2. **Fine-tuning significantly improved** VGG16 and MobileNetV2 performance
3. **Early stopping prevented overfitting** across all models
4. **Batch normalization and dropout** were crucial for generalization
5. **K-Fold validation confirmed** model stability and reliability

##  Troubleshooting

**Issue**: Out of memory errors  
**Solution**: Reduce batch size or use Google Colab with GPU

**Issue**: Low accuracy  
**Solution**: Ensure proper data normalization (divide by 255.0)

**Issue**: Model not improving  
**Solution**: Try lower learning rates or longer patience in early stopping

**Issue**: Overfitting  
**Solution**: Increase dropout rate or use more aggressive data augmentation

##  Technical Details

### Hyperparameters

**Custom CNN**:
- Learning Rate: 1e-6 (initial), 1e-5 (fine-tuning)
- Dropout: 0.5
- Batch Size: 16 (initial), 32 (fine-tuning)

**VGG16**:
- Learning Rate: 1e-4 (initial), 1e-6 (fine-tuning)
- Unfrozen Layers: Last 16 layers
- Dropout: 0.5

**MobileNetV2**:
- Learning Rate: 1e-4 (initial), 5e-7 (fine-tuning)
- Unfrozen Layers: After layer 80
- Dropout: 0.4

### Data Augmentation (Optional)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

##  References

- **VGG16**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **MobileNetV2**: Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- **Transfer Learning**: Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning"

##  Educational Use

This project is ideal for:
- Learning deep learning fundamentals
- Understanding transfer learning
- Practicing medical image classification
- Comparing CNN architectures
- Implementing K-Fold validation

##  Disclaimer

**Important**: This is an educational project and should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

##  Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add more data augmentation techniques
- [ ] Implement ensemble methods
- [ ] Add ResNet and EfficientNet models
- [ ] Create web interface for predictions
- [ ] Add explainability (Grad-CAM, LIME)
- [ ] Expand dataset with more images

##  Version Info

- **Version**: 1.0
- **Python**: 3.7+
- **TensorFlow**: 2.0+
- **Dataset Size**: 598 images
- **File Size**: ~15 MB (notebook)

##  License

This project is available for educational and research purposes.

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue in the repository
- Check Google Colab documentation for runtime issues
- Review TensorFlow/Keras documentation for API details

##  Acknowledgments

- Google Colab for providing free GPU resources
- TensorFlow/Keras teams for excellent deep learning frameworks
- Medical imaging community for dataset contributions


