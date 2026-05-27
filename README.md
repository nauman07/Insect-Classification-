# 🐛 Insect Classification Using CNN and Transfer Learning

A deep learning project for **automated insect classification** using Convolutional Neural Networks (CNN) and Transfer Learning, with an accompanying published research paper.

---

## 📖 About

Insect classification is critical in fields like agriculture, ecology, and pest management. This project automates species identification from insect images using deep learning — comparing a custom CNN approach with Transfer Learning techniques to achieve high classification accuracy.

This work is based on published research. Please cite the paper if you use this code.

---

## 📂 Repository Structure

```
Insect-Classification/
├── train_cnn.py              # Custom CNN training script
├── anotherCNN.py             # Alternative CNN architecture
├── Transfer.py               # Transfer Learning model
├── predict.py                # Prediction (255 input)
├── predict256.py             # Prediction (256 input)
├── predict_transfer.py       # Prediction using transfer model
├── confusion.py              # Confusion matrix generation
├── npy.py                    # NumPy array data utilities
├── model (1).py              # Additional model variant
├── cnn.txt                   # CNN architecture summary
├── Transfer.txt              # Transfer learning model summary
├── 47__Insect_NC.pdf         # Published research paper
├── Accuracy_curve_CNN_255.jpg
├── Accuracy_curve_CNN_256.jpg
├── Loss_curve_CNN_255.jpg
├── Loss_curve_CNN_256.jpg
├── Transfer_Learning1.jpg
├── Transfer_Learning2.jpg
└── README.md
```

---

## ✨ Features

- Custom CNN architecture for insect classification
- Transfer Learning using pre-trained models for improved accuracy
- Multiple input resolution support (255×255, 256×256)
- Confusion matrix for detailed performance analysis
- Accuracy and loss curve visualizations

---

## 📦 Dataset

The dataset is not publicly available in this repository. Contact the author to request it.

📧 **nabjad258@gmail.com**

---

## 🛠️ Tech Stack

| Library | Purpose |
|--------|---------|
| `TensorFlow` / `Keras` | CNN and Transfer Learning models |
| `NumPy` | Data array handling |
| `Matplotlib` | Accuracy/loss visualization |
| `Scikit-learn` | Confusion matrix and evaluation |
| `OpenCV` | Image preprocessing |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

### Train Custom CNN

```bash
python train_cnn.py
```

### Train with Transfer Learning

```bash
python Transfer.py
```

### Run Predictions

```bash
python predict.py         # For 255x255 input
python predict256.py      # For 256x256 input
python predict_transfer.py  # Using transfer learning model
```

### Generate Confusion Matrix

```bash
python confusion.py
```

---

## 📄 Publication

If you use this code in your research, please cite:

> **Insect Classification Using CNN**  
> Published in IJIRSET, November 2020  
> 🔗 [Read the Paper](http://www.ijirset.com/upload/2020/november/47__Insect_NC.pdf)

---

## 📬 Contact

For dataset requests or questions: **nabjad258@gmail.com**

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before submitting major changes.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
