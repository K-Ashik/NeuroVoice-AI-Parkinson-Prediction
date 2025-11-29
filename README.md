# 🧠 NeuroVoice AI: Parkinson's Disease Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neurovoice-ai-parkinson-prediction-hzusowqsv5zxppezzhkmby.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live Demo:** [Click here to test the app](https://neurovoice-ai-parkinson-prediction-hzusowqsv5zxppezzhkmby.streamlit.app/)

---

## 📖 Overview
**NeuroVoice AI** is an end-to-end medical screening tool designed to detect early vocal biomarkers associated with **Parkinson's Disease (PD)**. 

Unlike traditional "Black Box" AI models, this system is built on **Explainable AI (XAI)** principles. It uses advanced signal processing to extract micro-tremors from voice recordings and provides a transparent, clinically interpretable risk assessment.

### 🌟 Key Innovation: The "Real-World" Calibration
Most voice AI models fail outside the lab because they confuse **Microphone Noise** with **Neurological Tremors**. 
* **The Challenge:** Consumer laptops have a high noise floor (Low Signal-to-Noise Ratio).
* **My Solution:** I engineered a **Dynamic Calibration Engine** that scales the risk probability based on the audio's HNR (Harmonics-to-Noise Ratio). This effectively "subtracts" the hardware noise, preventing false positives for users with cheap microphones.

---

## 📸 Interface
![App Interface](assets/app_screenshot.jpg)
*The dashboard features a dynamic Risk Gauge, real-time Biomarker Cards (Jitter/Shimmer), and a downloadable Clinical Report.*

---

## 📊 Model Performance
The model was trained on the **UCI Parkinson's Dataset**, a gold-standard collection of biomedical voice measurements.

### **Test Set Metrics (20% Holdout)**
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **89.74%** | Highly accurate classification |
| **ROC-AUC** | **0.9655** | Excellent separation between Healthy/PD |
| **Precision (PD)** | **0.93** | 93% of identified cases were correct |
| **Recall (PD)** | **0.93** | Caught 93% of actual PD cases |

### **Detailed Classification Report**
```text
              precision    recall  f1-score   support

           0       0.80      0.80      0.80        10
           1       0.93      0.93      0.93        29

    accuracy                           0.90        39
   macro avg       0.87      0.87      0.87        39
weighted avg       0.90      0.90      0.90        39
