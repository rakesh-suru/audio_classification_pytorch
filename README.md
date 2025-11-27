# 📘 Quran Recitation Audio Classification using CNN

This project focuses on classifying Quran recitations into different categories using **Mel-Spectrograms** and a **Convolutional Neural Network (CNN)** built with PyTorch.  
The dataset is sourced from Kaggle and contains audio files of Quran recitations stored with corresponding labels.

## 📂 Dataset

- **Source:** Kaggle – Quran Recitations for Audio Classification
- **Processing:**
  - Loaded using `opendatasets`
  - Mel-spectrograms generated using **Librosa**
  - Resized to **128 × 256**
  - Labels encoded using `LabelEncoder`
- **Split:**
  - 70% Train  
  - 15% Validation  
  - 15% Test

## 📊 Class Distribution

The dataset classes are visualized using a pie chart.  
Each class represents a category of Quran recitation.

## 🛠️ Project Workflow

### 1️⃣ Data Preparation
- Load dataset CSV
- Correct file paths
- Encode labels
- Generate Mel-Spectrograms using:
  ```
  librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128)
  ```
- Resize using `skimage.resize`

### 2️⃣ Custom Dataset Class
A `CustomAudioDataset` is implemented to:
- Load audio → convert to spectrogram
- Convert spectrogram into tensors
- Return (audio_tensor, label)

### 3️⃣ Model Architecture
A 3-layer **Convolutional Neural Network (CNN)**:

- Conv2D → ReLU → MaxPool2D → Dropout  
- Flatten  
- Fully connected layers  
- Output layer  

Optimizer: **Adam**  
Loss: **CrossEntropyLoss**

### 4️⃣ Training
Tracks:
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

### 5️⃣ Evaluation
- Test accuracy printed at the end  
- Loss & accuracy curves plotted  

## ▶️ How to Run

### Install dependencies
```
pip install opendatasets librosa torch scikit-learn matplotlib numpy scikit-image torchsummary
```

### Download dataset
```
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification")
```

### Run the notebook  
Use GPU (Colab recommended).

## 🚀 Future Improvements
- Add data augmentation  
- Use ResNet / MobileNet backbones  
- Add CRNN model  
- SpecAugment  
- Streamlit UI  

## ✨ Author
**Rakesh Suru**
