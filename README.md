# 🎭 Sentiment Analysis on IMDB Movie Reviews

An end-to-end **Natural Language Processing (NLP)** project that compares **12 sentiment classification models** spanning classical machine learning, probabilistic sequence modeling, deep learning, transformer-based architectures, and ensemble learning on the IMDB Movie Reviews dataset.

The project provides a complete experimental pipeline including data preprocessing, exploratory data analysis, feature engineering, hyperparameter optimization, model evaluation, explainability, error analysis, and deployment-ready artifacts.

> **Developed as part of the Natural Language Processing (NLP) course in the Digilians Initiative under the supervision of the Ministry of Communications and Information Technology (MCIT), Egypt, and the Egyptian Military Academy.**

---

## 🚀 Project Highlights

* End-to-end sentiment analysis pipeline
* Exploratory Data Analysis (EDA)
* Configurable text preprocessing
* TF-IDF and Word2Vec feature representations
* Comparison of **11 sentiment classification models**
* Hyperparameter optimization for every model family
* Explainable AI using **LIME**, **SHAP**, and Transformer Attention Visualization
* Comprehensive error analysis
* Streamlit-ready saved models and artifacts

---

## 🧠 Models Evaluated

### Classical Machine Learning

* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (SVM)
* Random Forest
* XGBoost

### Probabilistic Model

* Hidden Markov Model (HMM)

### Deep Learning

* Simple RNN
* LSTM
* GRU
* Bidirectional LSTM (BiLSTM)

### Transformer

* Fine-tuned DistilBERT

### Ensemble learning

* stacking
---

## ⚙️ Methodology

The experimental pipeline consists of:

1. Dataset loading and analysis
2. Exploratory Data Analysis
3. Text preprocessing and normalization
4. Feature extraction (TF-IDF & Word2Vec)
5. Stratified train/validation/test split
6. Hyperparameter optimization
7. Model training and evaluation
8. Explainability and error analysis
9. Saving trained models and evaluation artifacts

---

## 📊 Evaluation Metrics

All models are evaluated using the same protocol with:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Additional analyses include:

* ROC Curve comparison
* Learning curves
* Misclassified review analysis
* Feature importance visualization
* Confidence distribution

---

## 🔍 Explainable AI

To improve model interpretability, the project incorporates:

* LIME
* SHAP
* Transformer Attention Visualization

These techniques provide insights into how different models make sentiment predictions.

---

## 📈 Key Findings

* **DistilBERT** achieved the best overall performance with an **F1-score of 0.922** and **ROC-AUC of 0.969**.
* Three TF-IDF-based classical machine learning models outperformed all recurrent neural network architectures.
* The Hidden Markov Model surpassed every recurrent neural network despite its architectural limitations for document-level sentiment classification.
* Hyperparameter optimization consistently improved model performance across all model families.

---

## 📂 Project Structure

```text
Sentiment Analysis Project/
├── api/
├── attachments
├── configs/
├── data/
├── logs/
├── models/
├── notebooks/
├── src/
├── streamlit/
├── requirements.txt
├── setup.ps1
├── setup.sh
└── README.md
```

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* NLTK
* Gensim
* Optuna
* LIME
* SHAP
* Matplotlib
* Seaborn
* Plotly
* Streamlit

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/DinaAli2003/sentiment-analysis-project.git
cd sentiment-analysis-project
```

Create the environment using the provided setup script:

**Windows**

```powershell
.\setup.ps1
```

**Linux / macOS**

```bash
bash setup.sh
```

The project is tested with **Python 3.11**, which provides stable binary wheels for all required dependencies.

---

⭐ If you find this project useful, consider giving it a star on GitHub.

