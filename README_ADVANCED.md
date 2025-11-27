# 📘 Quran Recitation Audio Classification using CNN & Mel-Spectrograms

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Active-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?style=for-the-badge)

A complete deep-learning pipeline for **Quran Recitation Classification** using **Mel-Spectrograms** and a **Convolutional Neural Network (CNN)** built with PyTorch.

This project includes:
- Dataset downloading  
- Audio preprocessing  
- Spectrogram generation  
- Model architecture  
- Training + Validation loops  
- Testing & Graph Visualization  

---

# 📂 Dataset

**Source:** Kaggle – *Quran Recitations for Audio Classification*  
The dataset consists of labeled Quran recitation audio files.

### **Processing Steps**
- Load audio using **Librosa**
- Convert to **Mel-Spectrogram (128 × 256)**
- Normalize & resize
- Encode labels (`LabelEncoder`)
- Split into:
  - **70% Train**
  - **15% Validation**
  - **15% Test**

---

# 🧠 Model Architecture

### **Custom CNN**
```
Conv2D → ReLU → MaxPool2D → Dropout
Conv2D → ReLU → MaxPool2D → Dropout
Conv2D → ReLU → MaxPool2D → Dropout
Flatten → Fully Connected Layers → Output
```

### Hyperparameters
| Parameter | Value |
|----------|--------|
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Epochs | 25 |
| LR | 1e-4 |
| Batch Size | 16 |

---

# 🛠️ Project Workflow

## 1️⃣ Data Preparation
- Load CSV  
- Fix file paths  
- Encode labels  
- Generate Mel-Spectrogram  
- Resize to consistent dimensions  

## 2️⃣ Dataset Loader
- Custom PyTorch Dataset  
- Preloads spectrogram tensors  
- Returns `(audio_tensor, label)`  

## 3️⃣ Training Loop
Tracks:
- Loss (Train + Validation)
- Accuracy (Train + Validation)
- GPU acceleration if available

## 4️⃣ Evaluation
- Test accuracy  
- Loss & Accuracy plots  

---

# 📈 Visualizations

The notebook generates:

### **Training vs Validation Loss**
### **Training vs Validation Accuracy**

Both graphs help detect overfitting and overall model performance.

---

# 📁 Recommended Project Structure

```
📦 Quran-Audio-Classification
├── README.md
├── audio_classification.ipynb
├── Dataset/
│   ├── files_paths.csv
│   ├── audio_files...
├── models/
│   └── model.pth  (optional)
└── plots/
    ├── accuracy.png
    └── loss.png
```

---

# ▶️ How to Run

### Install dependencies
```
pip install opendatasets librosa torch scikit-learn matplotlib numpy scikit-image torchsummary
```

### Download dataset
```
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mohammedalrajeh/quran-recitations-for-audio-classification")
```

### Run notebook
Use **GPU (Colab recommended)** for faster training.

---

# 🚀 Future Improvements

- ✔ Add **SpecAugment**
- ✔ Use **CRNN (CNN + GRU/LSTM)**
- ✔ Replace CNN with **ResNet18 / MobileNetV2**
- ✔ Add **training-progress callbacks**
- ✔ Deploy as **Streamlit App**
- ✔ Export as **ONNX / TorchScript**

---

# 🤝 Contributions

Pull requests are welcome!  
Feel free to:
- Improve model accuracy  
- Optimize preprocessing  
- Add new architectures  
- Enhance documentation  

---

# ✨ Author

**Rakesh Suru**  
Deep Learning • Audio Processing • PyTorch  
