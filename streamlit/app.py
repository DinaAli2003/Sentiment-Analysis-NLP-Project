"""
Sentiment Analysis — Premium AI SaaS Dashboard
Production-ready UI/UX with modern cards, sidebar, and reusable components.
All ML/NLP logic preserved.
"""

import os
import re
import json
import pickle
import subprocess
import sys
import importlib
import time
import types

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Sentiment Analysis — AI Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS — Premium Light Theme
# ------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .stApp {
        background: #F7F9FC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111827;
    }

    .main > div {
        padding: 0 2rem 2rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* --- Scrollbar --- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; }
    ::-webkit-scrollbar-thumb { background: #2563EB; border-radius: 8px; }

    /* --- Typography --- */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #111827;
    }

    /* --- Cards --- */
    .card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
        border: 1px solid #E5E7EB;
        transition: box-shadow 0.25s ease, transform 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        transform: translateY(-2px);
    }
    .card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
    }

    /* --- Hero --- */
    .hero {
        background: linear-gradient(135deg, #2563EB 0%, #14B8A6 100%);
        border-radius: 24px;
        padding: 2.5rem 2.5rem;
        margin-bottom: 2rem;
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: -30%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }
    .hero p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        max-width: 600px;
        position: relative;
        z-index: 1;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        background: rgba(255,255,255,0.15);
        color: #fff;
        padding: 0.25rem 1rem;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-badge-highlight {
        background: #FFFFFF;
        color: #2563EB;
        border-color: transparent;
        font-weight: 600;
    }

    /* --- Buttons --- */
    .btn-primary {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        background: linear-gradient(135deg, #2563EB, #14B8A6);
        color: #fff;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25);
        width: 100%;
    }
    .btn-primary:hover {
        box-shadow: 0 8px 28px rgba(37, 99, 235, 0.35);
        transform: translateY(-2px);
    }
    .btn-primary:active {
        transform: scale(0.98);
    }

    /* --- Result Card --- */
    .result-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .result-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-3px);
    }
    .result-card .emoji {
        font-size: 3rem;
        line-height: 1.2;
    }
    .result-card .sentiment {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    .result-card .confidence {
        font-size: 1rem;
        color: #6B7280;
    }
    .result-card .bar {
        width: 100%;
        height: 6px;
        background: #E5E7EB;
        border-radius: 6px;
        margin-top: 0.75rem;
        overflow: hidden;
    }
    .result-card .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .result-positive .sentiment { color: #10B981; }
    .result-positive .bar-fill { background: linear-gradient(90deg, #10B981, #34D399); }
    .result-negative .sentiment { color: #EF4444; }
    .result-negative .bar-fill { background: linear-gradient(90deg, #EF4444, #F87171); }
    .result-neutral .sentiment { color: #F59E0B; }
    .result-neutral .bar-fill { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .result-card .badge {
        position: absolute;
        top: 12px;
        right: 12px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        background: #F1F5F9;
        color: #6B7280;
        padding: 0.2rem 0.8rem;
        border-radius: 100px;
    }

    /* --- Metrics --- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    .metric-item .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .metric-item .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6B7280;
        margin-top: 0.2rem;
    }

    /* --- Sidebar --- */
    .css-1d391kg { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
    .sidebar-content {
        padding: 1rem 0.5rem;
    }
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6B7280;
        margin: 1rem 0 0.5rem 0;
    }
    .sidebar-divider {
        border-top: 1px solid #E5E7EB;
        margin: 1rem 0;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 2px solid #E5E7EB;
        padding: 0 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 0.9rem;
        color: #6B7280;
        border-radius: 10px 10px 0 0;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #111827;
        background: #F1F5F9;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB;
        background: transparent;
        border-bottom: 2px solid #2563EB;
    }

    /* --- Alerts --- */
    .alert-info, .alert-error, .alert-warning {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .alert-info { background: #EFF6FF; border-left-color: #3B82F6; color: #1E3A8A; }
    .alert-error { background: #FEF2F2; border-left-color: #EF4444; color: #991B1B; }
    .alert-warning { background: #FFFBEB; border-left-color: #F59E0B; color: #92400E; }

    /* --- Responsive --- */
    @media (max-width: 768px) {
        .main > div { padding: 0 1rem 1rem 1rem; }
        .hero { padding: 1.5rem; }
        .hero h1 { font-size: 2rem; }
        .hero p { font-size: 0.95rem; }
        .metric-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 480px) {
        .hero h1 { font-size: 1.6rem; }
        .metric-grid { grid-template-columns: 1fr; }
    }

    /* --- Utility --- */
    .flex { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
    .gap-1 { gap: 0.5rem; }
    .gap-2 { gap: 1rem; }
    .mt-1 { margin-top: 0.5rem; }
    .mt-2 { margin-top: 1rem; }
    .mb-1 { margin-bottom: 0.5rem; }
    .mb-2 { margin-bottom: 1rem; }
    .text-center { text-align: center; }
    .text-muted { color: #6B7280; }
    .text-sm { font-size: 0.85rem; }
    .w-full { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Constants & Dependency Check
# ------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.sidebar.warning("TensorFlow not installed. Recurrent models and ensemble unavailable.")

TRADITIONAL_ML_MODELS = ["Logistic Regression", "Naive Bayes", "SVM (Linear)", "Random Forest", "XGBoost"]
RECURRENT_MODELS = ["RNN", "LSTM", "GRU", "BiLSTM"] if TF_AVAILABLE else []

# ------------------------------
# Loaders (cached)
# ------------------------------
@st.cache_resource
def load_config():
    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        return json.load(f)

@st.cache_resource
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_resource
def load_keras_tokenizer():
    if 'keras.src.preprocessing.text' not in sys.modules:
        dummy = types.ModuleType('keras.src.preprocessing.text')
        sys.modules['keras.src.preprocessing.text'] = dummy
        from tensorflow.keras.preprocessing.text import Tokenizer
        dummy.Tokenizer = Tokenizer
    with open(os.path.join(MODELS_DIR, "keras_tokenizer.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_tfidf_vectorizer():
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_traditional_ml_model(name):
    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + "_model.pkl"
    with open(os.path.join(MODELS_DIR, filename), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_recurrent_model(name):
    if not TF_AVAILABLE:
        st.error(f"TensorFlow required for {name}.")
        st.stop()
    path = os.path.join(MODELS_DIR, f"{name.lower()}_model.keras")
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_hmm_models():
    with open(os.path.join(MODELS_DIR, "hmm_pos.pkl"), "rb") as f:
        hmm_pos = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "hmm_neg.pkl"), "rb") as f:
        hmm_neg = pickle.load(f)
    return hmm_pos, hmm_neg

@st.cache_resource
def load_transformer():
    try:
        from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
    except ImportError:
        st.error("Transformers library not installed.")
        st.stop()
    path = os.path.join(MODELS_DIR, "distilbert_sentiment")
    tokenizer = DistilBertTokenizerFast.from_pretrained(path)
    model = TFDistilBertForSequenceClassification.from_pretrained(path)
    return tokenizer, model

@st.cache_resource
def load_ensemble():
    if not TF_AVAILABLE:
        st.error("TensorFlow required for ensemble.")
        st.stop()
    with open(os.path.join(MODELS_DIR, "ensemble_meta_model.pkl"), "rb") as f:
        meta_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "ensemble_model_names.pkl"), "rb") as f:
        model_names = pickle.load(f)
    return meta_model, model_names

# ------------------------------
# Text Cleaner (auto-install)
# ------------------------------
def ensure_package(pkg_name):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        with st.spinner(f"Installing {pkg_name}..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

@st.cache_resource
def get_cleaner():
    for pkg in ["contractions", "bs4", "emoji"]:
        ensure_package(pkg)
    import contractions
    from bs4 import BeautifulSoup
    import emoji as emoji_lib
    from nltk.stem import WordNetLemmatizer
    import nltk
    for pkg in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    lemmatizer = WordNetLemmatizer()
    def clean(text: str) -> str:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = contractions.fix(text)
        text = emoji_lib.demojize(text, delimiters=(" ", " "))
        text = text.lower()
        text = re.sub(r"\d+", " <num> ", text)
        text = re.sub(r"[^a-zA-Z<>\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [lemmatizer.lemmatize(t) for t in text.split()]
        return " ".join(tokens)
    return clean

# ------------------------------
# Prediction wrappers (preserved)
# ------------------------------
def encode_and_pad(clean_text, tokenizer, max_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([clean_text])
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post"), seq[0]

def predict_traditional_ml(name, clean_text):
    model = load_traditional_ml_model(name)
    vectorizer = load_tfidf_vectorizer()
    features = vectorizer.transform([clean_text])
    prob = float(model.predict_proba(features)[0][1])
    label = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence

def predict_recurrent(name, padded_seq):
    model = load_recurrent_model(name)
    prob = float(model.predict(padded_seq, verbose=0)[0][0])
    label = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence

def predict_hmm(raw_seq, hmm_vocab):
    hmm_pos, hmm_neg = load_hmm_models()
    if len(raw_seq) == 0:
        return "Negative", 0.5
    clipped = np.clip(raw_seq, 0, hmm_vocab - 1).reshape(-1, 1)
    score_pos = hmm_pos.score(clipped)
    score_neg = hmm_neg.score(clipped)
    label = "Positive" if score_pos > score_neg else "Negative"
    m = max(score_pos, score_neg)
    p_pos = np.exp(score_pos - m) / (np.exp(score_pos - m) + np.exp(score_neg - m))
    confidence = p_pos if label == "Positive" else 1 - p_pos
    return label, float(confidence)

def predict_transformer(text, max_len):
    import tensorflow as tf
    tokenizer, model = load_transformer()
    enc = tokenizer([text], truncation=True, padding=True, max_length=max_len, return_tensors="tf")
    logits = model(enc).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    label = "Positive" if probs[1] > probs[0] else "Negative"
    confidence = float(max(probs))
    return label, confidence

def predict_ensemble(clean_text, raw_seq, padded_seq, text):
    meta_model, model_names = load_ensemble()
    config = load_config()
    probs = []
    for name in model_names:
        if name in TRADITIONAL_ML_MODELS:
            model = load_traditional_ml_model(name)
            vectorizer = load_tfidf_vectorizer()
            features = vectorizer.transform([clean_text])
            prob = model.predict_proba(features)[0][1]
        elif name in RECURRENT_MODELS:
            model = load_recurrent_model(name)
            prob = float(model.predict(padded_seq, verbose=0)[0][0])
        elif name == "HMM":
            hmm_pos, hmm_neg = load_hmm_models()
            if len(raw_seq) == 0:
                prob = 0.5
            else:
                clipped = np.clip(raw_seq, 0, config["hmm_vocab"] - 1).reshape(-1, 1)
                score_pos = hmm_pos.score(clipped)
                score_neg = hmm_neg.score(clipped)
                m = max(score_pos, score_neg)
                prob = np.exp(score_pos - m) / (np.exp(score_pos - m) + np.exp(score_neg - m))
        elif name == "Transformer (DistilBERT)":
            tokenizer, model = load_transformer()
            enc = tokenizer([text], truncation=True, padding=True,
                            max_length=config["transformer_max_len"], return_tensors="tf")
            logits = model(enc).logits
            probs_soft = tf.nn.softmax(logits, axis=1).numpy()[0]
            prob = probs_soft[1]
        else:
            prob = 0.5
        probs.append(prob)
    X_ensemble = np.array(probs).reshape(1, -1)
    prob_ensemble = meta_model.predict_proba(X_ensemble)[0][1]
    label = "Positive" if prob_ensemble > 0.5 else "Negative"
    confidence = prob_ensemble if label == "Positive" else 1 - prob_ensemble
    return label, confidence

# ------------------------------
# LIME Explainability (preserved)
# ------------------------------
@st.cache_resource
def get_lime_explainer():
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=["negative", "positive"])

def explain_with_lime(text, model_fn, num_features=10):
    explainer = get_lime_explainer()
    explanation = explainer.explain_instance(text, model_fn, num_features=num_features)
    return explanation

# ------------------------------
# UI Components
# ------------------------------

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🎬 Sentiment Analysis</h1>
        <p>Compare 10+ models — from classical ML to state-of-the-art Transformers — on movie reviews.</p>
        <div class="hero-badges">
            <span class="hero-badge">🧠 HMM</span>
            <span class="hero-badge">📈 RNN</span>
            <span class="hero-badge">🔁 LSTM</span>
            <span class="hero-badge">⚡ GRU</span>
            <span class="hero-badge">🔄 BiLSTM</span>
            <span class="hero-badge">🤗 DistilBERT</span>
            <span class="hero-badge hero-badge-highlight">🧩 Ensemble (Stacking)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0;">
            <h3 style="font-weight: 700; color: #2563EB; margin: 0;">🎬 Sentiment AI</h3>
            <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">Model Comparison Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown('<div class="sidebar-title">Model Family</div>', unsafe_allow_html=True)
        model_family = st.selectbox(
            "Family",
            ["All Models", "Traditional ML", "Deep Learning", "Transformer", "Ensemble"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="sidebar-title">Model Selection</div>', unsafe_allow_html=True)
        # Build model list based on family
        if model_family == "All Models":
            model_options = ["All models (compare side-by-side)"] + TRADITIONAL_ML_MODELS + RECURRENT_MODELS + ["HMM", "Transformer (DistilBERT)"]
            if TF_AVAILABLE:
                model_options.append("Ensemble (Stacking)")
        elif model_family == "Traditional ML":
            model_options = TRADITIONAL_ML_MODELS
        elif model_family == "Deep Learning":
            model_options = RECURRENT_MODELS + ["HMM"]
        elif model_family == "Transformer":
            model_options = ["Transformer (DistilBERT)"]
        elif model_family == "Ensemble":
            model_options = ["Ensemble (Stacking)"] if TF_AVAILABLE else ["Ensemble (unavailable)"]
        else:
            model_options = ["All models (compare side-by-side)"]

        if "model_choice" not in st.session_state:
            st.session_state.model_choice = "All models (compare side-by-side)"

        # Use radio with custom styling via columns? We'll use selectbox for clean UI
        model_choice = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0,
            label_visibility="collapsed"
        )
        st.session_state.model_choice = model_choice

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)

        explain_toggle = st.toggle("Explainability (LIME)", value=False)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.5, 0.05)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.8rem; color: #6B7280;">
            <p><strong>Technologies</strong></p>
            <p>TensorFlow · Transformers · XGBoost · Streamlit</p>
            <p style="margin-top: 0.5rem;">Version 2.0 · Explainable AI</p>
        </div>
        """, unsafe_allow_html=True)

        return model_choice, explain_toggle, confidence_threshold

def render_input_section(text, set_text):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📝 Input Text</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        input_text = st.text_area(
            "Review",
            value=text,
            height=150,
            label_visibility="collapsed",
            key="input_text_area"
        )
    with col2:
        st.markdown("""
        <div style="padding: 0.5rem 0;">
            <p style="font-size: 0.8rem; color: #6B7280;">Examples</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📌 Positive", key="ex_pos", use_container_width=True):
            input_text = "This film was an absolute masterpiece, the acting and story blew me away."
        if st.button("📌 Negative", key="ex_neg", use_container_width=True):
            input_text = "Terrible movie, worst acting I've ever seen, complete waste of time."
        if st.button("📌 Neutral", key="ex_neu", use_container_width=True):
            input_text = "The movie was okay, not great but not terrible either."

    # Character and word counter
    if input_text:
        word_count = len(input_text.split())
        char_count = len(input_text)
        st.markdown(f"<div style='font-size:0.8rem; color:#6B7280;'>{char_count} characters · {word_count} words</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    return input_text

def render_model_info(model_choice, config):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Model Information</div>', unsafe_allow_html=True)

    if model_choice == "All models (compare side-by-side)":
        info = "Comparing all available models. Results will be shown side-by-side."
    else:
        info = f"**{model_choice}** – {get_model_description(model_choice)}"
    st.markdown(info)
    st.markdown('</div>', unsafe_allow_html=True)

def get_model_description(name):
    descriptions = {
        "Logistic Regression": "Linear classifier with L2 regularization.",
        "Naive Bayes": "Probabilistic classifier based on Bayes' theorem.",
        "SVM (Linear)": "Support Vector Machine with linear kernel.",
        "Random Forest": "Ensemble of decision trees with bagging.",
        "XGBoost": "Gradient boosting optimized for performance.",
        "RNN": "Recurrent Neural Network with SimpleRNN layer.",
        "LSTM": "Long Short-Term Memory network.",
        "GRU": "Gated Recurrent Unit network.",
        "BiLSTM": "Bidirectional LSTM for context from both directions.",
        "HMM": "Hidden Markov Model for sequence classification.",
        "Transformer (DistilBERT)": "Fine-tuned DistilBERT transformer model.",
        "Ensemble (Stacking)": "Logistic regression meta-model combining all base models."
    }
    return descriptions.get(name, "Machine learning model for sentiment analysis.")

def render_results(rows, confidence_threshold):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Prediction Results</div>', unsafe_allow_html=True)

    if not rows:
        st.info("No predictions yet. Enter text and click 'Analyze Sentiment'.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Apply confidence threshold: adjust label if confidence < threshold
    adjusted_rows = []
    for name, label, conf, runtime in rows:
        if conf < confidence_threshold:
            adjusted_label = "Neutral"  # or keep original but mark low confidence
        else:
            adjusted_label = label
        adjusted_rows.append((name, adjusted_label, conf, runtime))

    # Display in grid
    cols = st.columns(min(4, len(adjusted_rows)))
    for idx, (name, label, conf, runtime) in enumerate(adjusted_rows):
        col = cols[idx % len(cols)]
        emoji = "😊" if label == "Positive" else "😞" if label == "Negative" else "😐"
        color_class = "positive" if label == "Positive" else "negative" if label == "Negative" else "neutral"
        bar_width = conf * 100

        with col:
            st.markdown(f"""
            <div class="result-card result-{color_class}">
                <div class="badge">{name}</div>
                <div class="emoji">{emoji}</div>
                <div class="sentiment">{label}</div>
                <div class="confidence">{conf*100:.1f}% confidence</div>
                <div class="bar"><div class="bar-fill" style="width: {bar_width}%;"></div></div>
                <div style="font-size:0.7rem; color:#6B7280; margin-top:0.3rem;">⏱ {runtime:.3f}s</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_metrics(rows):
    if not rows:
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Confidence Metrics</div>', unsafe_allow_html=True)

    # Show metrics for each model or aggregate
    # For simplicity, show best and worst confidence
    best = max(rows, key=lambda x: x[2])
    worst = min(rows, key=lambda x: x[2])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", best[0], f"{best[2]*100:.1f}% confidence")
    with col2:
        st.metric("Worst Model", worst[0], f"{worst[2]*100:.1f}% confidence")
    with col3:
        avg_conf = np.mean([r[2] for r in rows])
        st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
    with col4:
        st.metric("Number of Models", len(rows))

    st.markdown('</div>', unsafe_allow_html=True)

def render_probability_chart(rows):
    if not rows:
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Probability Distribution</div>', unsafe_allow_html=True)

    df = pd.DataFrame({
        "Model": [r[0] for r in rows],
        "Confidence": [r[2] for r in rows],
        "Label": [r[1] for r in rows]
    })
    df["Color"] = df["Label"].map({"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"}).fillna("#6B7280")

    fig = px.bar(
        df,
        x="Model",
        y="Confidence",
        color="Label",
        color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
        title="Confidence by Model",
        labels={"Confidence": "Confidence Score"},
        text_auto=".2f"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#111827",
        showlegend=False,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_lime_explanation(text, model_fn, explain_toggle):
    if not explain_toggle or not text:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Explainability (LIME)</div>', unsafe_allow_html=True)

    with st.spinner("Generating explanation..."):
        try:
            explanation = explain_with_lime(text, model_fn, num_features=10)
            # Display explanation as HTML
            html = explanation.as_html()
            st.components.v1.html(html, height=400, scrolling=True)
        except Exception as e:
            st.error(f"LIME explanation failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def render_comparison_tab(metrics):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    if not metrics:
        st.info("No metrics available. Run the training notebook first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "confusion_matrix"}
        for name, r in metrics.items()
    }).T.sort_values("f1", ascending=False)
    df.insert(0, "Rank", range(1, len(df)+1))

    st.dataframe(df.style.format({
        "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
        "f1": "{:.3f}", "roc_auc": "{:.3f}", "train_time_sec": "{:.1f}"
    }).background_gradient(subset=["f1", "accuracy"], cmap="Blues"))

    # Plotly chart
    numeric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    if all(c in df.columns for c in numeric_cols):
        fig = px.bar(
            df.reset_index(),
            x="index",
            y=numeric_cols,
            barmode="group",
            title="Metric Comparison",
            labels={"index": "Model", "value": "Score"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#111827",
            legend_title_text="Metric",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    with st.expander("📌 Confusion Matrices", expanded=False):
        model_names = list(metrics.keys())
        for i in range(0, len(model_names), 3):
            cols = st.columns(3)
            for col, name in zip(cols, model_names[i:i+3]):
                with col:
                    cm = np.array(metrics[name]["confusion_matrix"])
                    fig, ax = plt.subplots(figsize=(2.8, 2.8))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["neg", "pos"], yticklabels=["neg", "pos"],
                        ax=ax, cbar=False, annot_kws={"size": 10, "weight": "bold"}
                    )
                    ax.set_title(name, fontsize=10, color="#111827")
                    ax.tick_params(colors="#6B7280", labelsize=8)
                    ax.set_xlabel("Predicted", color="#6B7280", fontsize=8)
                    ax.set_ylabel("True", color="#6B7280", fontsize=8)
                    plt.tight_layout()
                    col.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Main App
# ------------------------------
def main():
    render_hero()

    # Sidebar
    model_choice, explain_toggle, confidence_threshold = render_sidebar()

    # Check artifacts
    if not os.path.exists(os.path.join(MODELS_DIR, "config.json")):
        st.markdown("""
        <div class="alert-error">
            ⚠️ <strong>Artifacts not found.</strong> Run the training notebook first.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    config = load_config()
    metrics = load_metrics()

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Live Demo", "📊 Comparison"])

    with tab1:
        # Input
        default_text = "This film was an absolute masterpiece, the acting and story blew me away."
        input_text = render_input_section(default_text, st.session_state.get("input_text_area", default_text))

        # Model info
        render_model_info(model_choice, config)

        # Analyze button
        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Running models..."):
                    cleaner = get_cleaner()
                    clean_text = cleaner(input_text)
                    keras_tokenizer = load_keras_tokenizer()
                    padded_seq, raw_seq = encode_and_pad(clean_text, keras_tokenizer, config["max_len"])
                    run_all = model_choice == "All models (compare side-by-side)"
                    rows = []

                    # Traditional ML
                    for name in TRADITIONAL_ML_MODELS:
                        if run_all or model_choice == name:
                            t0 = time.perf_counter()
                            label, conf = predict_traditional_ml(name, clean_text)
                            rows.append((name, label, conf, time.perf_counter() - t0))
                    # Recurrent
                    for name in RECURRENT_MODELS:
                        if run_all or model_choice == name:
                            t0 = time.perf_counter()
                            label, conf = predict_recurrent(name, padded_seq)
                            rows.append((name, label, conf, time.perf_counter() - t0))
                    # HMM
                    if run_all or model_choice == "HMM":
                        t0 = time.perf_counter()
                        label, conf = predict_hmm(np.array(raw_seq), config["hmm_vocab"])
                        rows.append(("HMM", label, conf, time.perf_counter() - t0))
                    # Transformer
                    if run_all or model_choice == "Transformer (DistilBERT)":
                        t0 = time.perf_counter()
                        label, conf = predict_transformer(input_text, config["transformer_max_len"])
                        rows.append(("Transformer (DistilBERT)", label, conf, time.perf_counter() - t0))
                    # Ensemble
                    if TF_AVAILABLE and (run_all or model_choice == "Ensemble (Stacking)"):
                        t0 = time.perf_counter()
                        label, conf = predict_ensemble(clean_text, np.array(raw_seq), padded_seq, input_text)
                        rows.append(("Ensemble (Stacking)", label, conf, time.perf_counter() - t0))

                # Display results
                render_results(rows, confidence_threshold)
                render_metrics(rows)
                render_probability_chart(rows)

                # LIME explanation
                if explain_toggle and rows:
                    # Use the first model that is not an ensemble for explanation
                    # We'll use the best performing model or the first available
                    best_model_name = max(rows, key=lambda x: x[2])[0]
                    # Determine which model to use for LIME
                    if best_model_name in TRADITIONAL_ML_MODELS:
                        # Use a wrapper for traditional ML
                        def lime_predict(texts):
                            X = load_tfidf_vectorizer().transform(texts)
                            model = load_traditional_ml_model(best_model_name)
                            return model.predict_proba(X)
                    elif best_model_name in RECURRENT_MODELS:
                        def lime_predict(texts):
                            cleaned = [get_cleaner()(t) for t in texts]
                            padded = encode_and_pad(cleaned, load_keras_tokenizer(), config["max_len"])[0]
                            model = load_recurrent_model(best_model_name)
                            probs = model.predict(padded, verbose=0)
                            return np.column_stack([1-probs.ravel(), probs.ravel()])
                    elif best_model_name == "HMM":
                        def lime_predict(texts):
                            # HMM prediction for multiple texts
                            probs = []
                            for t in texts:
                                clean_t = get_cleaner()(t)
                                _, raw_seq = encode_and_pad(clean_t, load_keras_tokenizer(), config["max_len"])
                                _, conf = predict_hmm(np.array(raw_seq), config["hmm_vocab"])
                                probs.append([1-conf, conf])
                            return np.array(probs)
                    elif best_model_name == "Transformer (DistilBERT)":
                        def lime_predict(texts):
                            tokenizer, model = load_transformer()
                            enc = tokenizer(texts, truncation=True, padding=True, max_length=config["transformer_max_len"], return_tensors="tf")
                            logits = model(enc).logits
                            return tf.nn.softmax(logits, axis=1).numpy()
                    elif best_model_name == "Ensemble (Stacking)":
                        def lime_predict(texts):
                            # Ensemble prediction for multiple texts
                            probs = []
                            for t in texts:
                                clean_t = get_cleaner()(t)
                                _, raw_seq = encode_and_pad(clean_t, load_keras_tokenizer(), config["max_len"])
                                padded_seq, _ = encode_and_pad(clean_t, load_keras_tokenizer(), config["max_len"])
                                _, conf = predict_ensemble(clean_t, np.array(raw_seq), padded_seq, t)
                                probs.append([1-conf, conf])
                            return np.array(probs)
                    else:
                        # Fallback: use a simple wrapper
                        def lime_predict(texts):
                            return np.array([[0.5, 0.5] for _ in texts])

                    render_lime_explanation(input_text, lime_predict, explain_toggle)

    with tab2:
        render_comparison_tab(metrics)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.8rem; padding: 1.5rem 0 0.5rem 0;">
        NLP Course Project — 11 models · Stacking Ensemble · Explainable AI · Premium Dashboard
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()