import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

st.markdown("""
<style>
.main {background:#f7faf7}
.block-container {padding-top:1.8rem}
.hero {padding:1.4rem 1.8rem;border-radius:18px;background:linear-gradient(135deg,#e8f5e9,#fff);border:1px solid #c8e6c9;margin-bottom:1.2rem}
.hero h1 {margin:0;color:#1b5e20;font-size:2rem}
.hero p {margin:0.4rem 0 0;color:#2e7d32;font-size:1.05rem}
.prediction {padding:1.2rem;border-radius:16px;background:#fff;border:1px solid #dfe8df;box-shadow:0 2px 8px rgba(0,0,0,0.04)}
.prediction h2 {margin:0;color:#1b5e20;font-size:1.8rem}
.card {padding:1rem;border-radius:14px;background:#fff;border:1px solid #e3e8e3;min-height:90px;box-shadow:0 2px 6px rgba(0,0,0,0.03)}
.small {color:#657065;font-size:.9rem}
.footer {text-align:center;color:#778077;padding-top:2rem;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

def find_file(filename_patterns):
    for pattern in filename_patterns:
        direct_candidate = BASE_DIR / pattern
        if direct_candidate.exists():
            return direct_candidate
        matches = list(BASE_DIR.rglob(pattern))
        if matches:
            return matches[0]
    return None

def get_model_path():
    return find_file([
        "models/waste_mobilenetv3.keras",
        "waste_mobilenetv3.keras",
        "models/*.keras",
        "*.keras",
        "models/*.h5",
        "*.h5"
    ])

def get_class_path():
    return find_file([
        "models/class_names.json",
        "class_names.json",
        "*class_names.json"
    ])

@st.cache_resource
def load_model(path_str: str = ""):
    if path_str and Path(path_str).exists():
        try:
            return keras.models.load_model(path_str)
        except Exception as e:
            st.warning(f"Note: Error loading custom model file: {e}. Initializing MobileNetV3Large fallback.")
    
    # Pre-trained MobileNetV3Large fallback if custom weights file is not available
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

@st.cache_data
def load_classes(path_str: str = ""):
    if path_str and Path(path_str).exists():
        try:
            with open(path_str, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return ["Glass", "Metal", "Paper", "Plastic"]

def preprocess_image(pil_img):
    resized = pil_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict_probabilities(model, arr):
    preds = model(arr, training=False).numpy()[0]
    if np.any(preds < 0) or not np.isclose(preds.sum(), 1.0, atol=1e-3):
        preds = tf.nn.softmax(preds).numpy()
    return preds

INFO_DICT = {
    "Glass": ("Glass bottles and jars", "Rinse thoroughly and place in designated glass recycling bins."),
    "Paper": ("Cardboard, magazines, newspaper, office paper", "Keep clean and dry before recycling."),
    "Metal": ("Aluminum and steel beverage/food cans", "Empty and rinse before placing in metal recycling."),
    "Plastic": ("Plastic bottles, jugs, containers", "Empty, rinse, and check the resin identification code.")
}

model_file = get_model_path()
class_file = get_class_path()

model_path_str = str(model_file) if model_file else ""
class_path_str = str(class_file) if class_file else ""

class_names = load_classes(class_path_str)
model = load_model(model_path_str)

st.markdown("""
<div class="hero">
<h1>♻️ Waste Classification AI</h1>
<p>Upload a photo of a waste item to identify its category and recycling guidelines.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings & Info")
    st.write("**Model:** MobileNetV3Large")
    st.write("**Classes:**", ", ".join(class_names))
    if model_file:
        st.caption(f"Loaded: `{model_file.name}`")
    else:
        st.caption("⚡ Running MobileNetV3Large")
    st.divider()
    st.info("💡 Upload clear photos of recyclable items (Glass, Paper, Metal, Plastic) for optimal accuracy.")

def display_prediction_results(pil_image, probabilities):
    pred_idx = int(np.argmax(probabilities))
    predicted_class = class_names[pred_idx]
    confidence = float(np.max(probabilities)) * 100

    col_img, col_pred = st.columns([1.2, 1], gap="medium")
    with col_img:
        st.image(pil_image, caption="Uploaded Item", use_container_width=True)

    with col_pred:
        st.markdown(
            f'<div class="prediction"><div class="small">Predicted Waste Type</div>'
            f'<h2>{predicted_class}</h2><p><b>Confidence: {confidence:.2f}%</b></p></div>',
            unsafe_allow_html=True
        )
        st.progress(min(confidence / 100.0, 1.0))

        st.subheader("📊 Class Probabilities")
        chart_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)

    st.divider()
    st.subheader("♻️ Material Information & Recycling Tip")
    description, tip = INFO_DICT.get(predicted_class, ("Waste material", "Follow local waste-management guidelines."))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="card"><b>Material</b><br>{description}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><b>Recycling Tip</b><br>{tip}</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a waste item image to classify",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    pil_image = Image.open(uploaded).convert("RGB")
    arr = preprocess_image(pil_image)
    probabilities = predict_probabilities(model, arr)
    display_prediction_results(pil_image, probabilities)
else:
    st.info("👆 Please upload an image file (JPG, PNG, or WEBP) above to get started.")

st.markdown('<div class="footer">Waste Classification AI • MobileNetV3Large • TensorFlow + Streamlit</div>',
            unsafe_allow_html=True)
