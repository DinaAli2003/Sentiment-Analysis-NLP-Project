# 🎭 Sentiment Analysis NLP Project

An end-to-end **Natural Language Processing (NLP)** project for sentiment analysis that compares traditional Machine Learning, Deep Learning, and Transformer-based models. The project includes data preprocessing, model training, explainable AI, a REST API, and an interactive Streamlit application.

> **Developed as part of the Natural Language Processing (NLP) course in the Digilians Initiative under the supervision of the Ministry of Communications and Information Technology (MCIT), Egypt, and the Egyptian Military Academy.**

---

## 🚀 Features

- End-to-end NLP pipeline
- Text preprocessing and feature engineering
- Traditional Machine Learning models
- Deep Learning models (LSTM/GRU/RNN/HMM/BILSTM)
- Transformer-based models (e.g., BERT/DistilBERT)
- Ensemble learning(Stacking)
- Model evaluation and comparison
- Explainability using LIME & SHAPE
- REST API for inference
- Interactive Streamlit application

---

## 📂 Project Structure

```text
Sentiment Analysis Project/
├── api/            # REST API
├── configs/        # Configuration files
├── data/           # Dataset
├── logs/           # Training logs
├── models/         # Trained models
├── notebooks/      # Experiments and analysis
├── src/            # Source code
├── streamlit/      # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- Hugging Face Transformers
- NLTK
- LIME
- Streamlit
- FastAPI / Flask

---

## 📊 Workflow

```text
Dataset
   ↓
Text Preprocessing
   ↓
Feature Extraction
   ↓
Model Training
   ↓
Evaluation
   ↓
LIME Explainability
   ↓
Deployment (API & Streamlit)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/DinaAli2003/Sentiment-Analysis-NLP-Project.git
cd sentiment-analysis-project

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Streamlit

```bash
streamlit run streamlit/app.py
```

### API

```bash
uvicorn api.app: app --reload
```

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---
⭐ If you find this project helpful, consider starring the repository!
