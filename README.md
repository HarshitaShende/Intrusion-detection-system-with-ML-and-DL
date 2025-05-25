# Intrusion-detection-system-with-ML-and-DL

# 🚨 Intrusion Detection System with Machine Learning & Deep Learning

This project implements a hybrid Intrusion Detection System (IDS) using Machine Learning and Deep Learning techniques to classify network traffic as normal or malicious. The system is built on publicly available datasets and aims to improve network security by identifying potential threats in real-time.

## 📌 Project Highlights

- ✅ Binary and multiclass classification of network traffic
- 🧠 Algorithms: Random Forest, XGBoost, CNN, LSTM
- 📊 Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- 📂 Dataset: NSL-KDD Dataset
- 🛠️ Tools: Python, Scikit-learn, TensorFlow/Keras, Pandas, Matplotlib

---

## 📁 Dataset

We used the **NSL-KDD** dataset which is an improved version of the KDD’99 dataset. It contains labeled records of both normal and attack types of traffic.

- Features: 41
- Classes: Normal + 4 categories of attacks (DoS, R2L, U2R, Probe)

You can download the dataset [here](https://www.unb.ca/cic/datasets/nsl.html).

---

## 🧪 Models Used

### 1. Machine Learning Models
- **Random Forest**  
- **XGBoost**

### 2. Deep Learning Models
- **Convolutional Neural Network (CNN)**
- **Long Short-Term Memory (LSTM)**

---

## 🧮 Evaluation Metrics

| Metric      | Description                                     |
|-------------|-------------------------------------------------|
| Accuracy    | Overall performance of the model                |
| Precision   | Correct positive predictions vs. total predicted |
| Recall      | Correct positive predictions vs. actual positives |
| F1-Score    | Harmonic mean of precision and recall           |

All models are benchmarked using confusion matrices and classification reports.

---

## 📈 Results

- Random Forest achieved high accuracy and good generalization.
- CNN and LSTM models improved performance on complex patterns.
- Deep learning models were especially effective on imbalanced attack classes.

---

## 🛠️ Installation

```bash
git clone https://github.com/HarshitaShende/intrusion-detection-ml-dl.git
cd intrusion-detection-ml-dl
pip install -r requirements.txt
