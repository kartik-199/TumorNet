# 🧠 TumorNet: Brain Tumor Classification using CNN and PyTorch

This project implements TumorNet, a convolutional neural network (CNN) to classify brain MRI images into four categories: **glioma**, **meningioma**, **no tumor**, and **pituitary tumor**. The model is built with PyTorch and includes data preprocessing, model training, evaluation, and saliency-based visualization techniques.

---

## 🚀 Features

- **Data Preprocessing**: Image resizing, normalization, and augmentation.
- **Model Architecture**: TumorNet CNN with convolutional layers, batch normalization, and dropout.
- **Training**: Optimized using Adam optimizer with cross-entropy loss.
- **Evaluation**: Model evaluation with confusion matrix, classification report, and saliency map visualization.

---

## 🛠️ Architecture 

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 224, 224]           2,432
       BatchNorm2d-2         [-1, 32, 224, 224]              64
         MaxPool2d-3         [-1, 32, 112, 112]               0
            Conv2d-4         [-1, 64, 112, 112]          32,832
       BatchNorm2d-5         [-1, 64, 112, 112]             128
         MaxPool2d-6           [-1, 64, 56, 56]               0
            Conv2d-7          [-1, 128, 56, 56]          73,856
       BatchNorm2d-8          [-1, 128, 56, 56]             256
         MaxPool2d-9          [-1, 128, 28, 28]               0
           Conv2d-10          [-1, 256, 28, 28]         295,168
      BatchNorm2d-11          [-1, 256, 28, 28]             512
        MaxPool2d-12          [-1, 256, 14, 14]               0
           Linear-13                  [-1, 512]      25,690,624
          Dropout-14                  [-1, 512]               0
           Linear-15                    [-1, 4]           2,052
================================================================
Total params: 26,097,924
Trainable params: 26,097,924
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 51.69
Params size (MB): 99.56
Estimated Total Size (MB): 151.82
----------------------------------------------------------------
```

## 🎯 Model Accuracy
```text
              precision    recall  f1-score   support

      glioma       1.00      0.95      0.98       150
  meningioma       0.95      0.97      0.96       153
     notumor       0.98      1.00      0.99       203
   pituitary       1.00      0.99      1.00       150

    accuracy                           0.98       656
   macro avg       0.98      0.98      0.98       656
weighted avg       0.98      0.98      0.98       656
```

## 🖼️ Saliency Mapping
This model uses saliency maps which are visualizations that highlight the areas of the input image that most influenced the model's prediction.
