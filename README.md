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
![App Interface](assests/app_screenshot.jpg)
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



````

### **Confusion Matrix**

*The model correctly identified 27/29 Parkinson's patients and 8/10 Healthy controls in the blind test set.*

![Confusion Matrix](assests/confusion_matrix.png)

-----

## 🔬 Data Science Workflow

### 1\. Exploratory Data Analysis (EDA)

Before training, I used **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to visualize the high-dimensional data in 2D.

  * **Insight:** The clear separation between Healthy (Green) and PD (Red) clusters confirmed that vocal features contain a strong predictive signal, validating the feasibility of an ML approach.

![t-SNE](assests/04_tsne_clusters.png)


### 2\. Feature Engineering

  * **Signal Processing:** Used `Parselmouth` (Python wrapper for **Praat**) to extract 22 acoustic features.
  * **Selection:** Removed collinear features (e.g., `Jitter:RAP`, `Jitter:PPQ`) to prevent overfitting and improve model interpretability.
  * **Extraction Strategy:** Implemented a **"Stability Scout"** algorithm that scans the audio file window-by-window to find the single most stable 1-second segment, ignoring breath/start-up artifacts.

### 3\. Explainable AI (SHAP)

Medical AI must be transparent. I used **SHAP (SHapley Additive exPlanations)** to break down *why* the model made a specific decision.

  * **Top Predictor:** **PPE (Pitch Period Entropy)** and **Spread1** were identified as the strongest biomarkers for Parkinson's, aligning with clinical literature on vocal rigidity.
-----

![Shap](assests/shap_summary.png)


## 🛠️ Tech Stack

  * **Frontend:** Streamlit, Plotly (Interactive Charts)
  * **Backend:** Python
  * **Machine Learning:** XGBoost (Extreme Gradient Boosting)
  * **Audio Processing:** Praat / Parselmouth
  * **Explainability:** SHAP (Shapley Additive Explanations)

-----

## 🚀 How to Run Locally

1.  **Clone the Repo**

    ```bash
    git clone [https://github.com/K-Ashik/NeuroVoice-AI-Parkinson-Prediction.git](https://github.com/K-Ashik/NeuroVoice-AI-Parkinson-Prediction.git)
    cd NeuroVoice-AI
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**

    ```bash
    streamlit run app.py
    ```

-----

## ⚠️ Disclaimer

**For Educational & Portfolio Purposes Only.**
This tool is a proof-of-concept for tele-medicine screening. It is **not** a certified medical device and should not be used for self-diagnosis. If you have health concerns, please consult a neurologist.

-----

## 📬 Contact

Created by **[Khalid Md Ashik]** - [LinkedIn Profile](https://www.linkedin.com/in/khalid-md-ashik/) | [Portfolio](https://portfolio-khalid-ashik.lovable.app/) | [Github](https://github.com/K-Ashik)


