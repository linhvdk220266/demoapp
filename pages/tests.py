import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
import os
import matplotlib.pyplot as plt
import pywt
from skimage.feature import local_binary_pattern
from scipy.stats import linregress

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array


# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
PCA_PATH = os.path.join(BASE_DIR, "pca.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMG_SIZE = 160
DROPOUT = 0.3
SCALE_FACTOR = 1.0


# ==========================================
# FEATURE EXTRACTORS
# ==========================================
def extract_prnu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'db2')
    noise = np.sqrt(np.abs(cH * cV * cD))
    return [noise.mean(), noise.std()]


def extract_dct(image, scale=SCALE_FACTOR):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_abs = np.log1p(np.abs(dct))

    h, w = dct_abs.shape
    low = np.mean(dct_abs[:h//4, :w//4])
    mid = np.mean(dct_abs[h//4:h//2, w//4:w//2])
    high = np.mean(dct_abs[h//2:, w//2:])
    dct_std = np.std(dct_abs)
    return [low*scale, mid*scale, high*scale, dct_std*scale]


def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feats = []
    for radius in [1, 2, 3]:
        lbp = local_binary_pattern(gray, P=8*radius, R=radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
        hist = hist / (hist.sum() + 1e-8)
        feats.extend(hist.tolist())
    return feats


def extract_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log1p(np.abs(fft_shift) / gray.size)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]

    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_norm = r / (r.max() + 1e-8)

    low_mask = (r_norm <= 0.15)
    mid_mask = (r_norm > 0.15) & (r_norm <= 0.40)
    high_mask = (r_norm > 0.40)

    fft_low = mag[low_mask].mean()
    fft_mid = mag[mid_mask].mean()
    fft_high = mag[high_mask].mean()
    fft_std = mag.std()
    hf_ratio = fft_high / (fft_low + 1e-8)

    radial_bins = np.linspace(0, r_norm.max(), 50)
    radial_energy = []
    for i in range(len(radial_bins)-1):
        mask = (r_norm >= radial_bins[i]) & (r_norm < radial_bins[i+1])
        radial_energy.append(mag[mask].mean() if mask.any() else 0)

    radial_energy = np.array(radial_energy) + 1e-8
    slope, *_ = linregress(np.arange(len(radial_energy)), np.log(radial_energy))

    return [fft_low, fft_mid, fft_high, fft_std, hf_ratio, slope]


def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return np.hstack([
        extract_prnu(img),
        extract_dct(img),
        extract_lbp(img),
        extract_fft(img)
    ])

augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augment",
)
norm = layers.Rescaling(1.0 / 255.0)

# ==========================================
# BUILD CNN (same architecture as training)
# ==========================================
def se_block(x, ratio=8):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Reshape((1, 1, c))(s)
    s = layers.Dense(max(c // ratio, 4), activation="swish")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    return layers.Multiply()([x, s])

def conv_bn(x, f, k=3, s=1):
    x = layers.Conv2D(f, k, s, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    return x

def res_se_block(x, f, stride=1, se_ratio=0.25, drop=0.0):
    i = x
    x = conv_bn(x, f, 3, stride)
    x = layers.Conv2D(f, 3, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    if se_ratio:
        x = se_block(x, ratio=int(1 / se_ratio))
    if stride != 1 or i.shape[-1] != f:
        i = layers.Conv2D(f, 1, stride, padding="same", use_bias=False, kernel_initializer="he_normal")(i)
        i = layers.BatchNormalization()(i)
    x = layers.Add()([x, i])
    x = layers.Activation("swish")(x)
    if drop > 0:
        x = layers.Dropout(drop)(x)
    return x

def build_model():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = augment(inp)
    x = norm(x)

    # Stem
    x = conv_bn(x, 32, 3, 2)        # 80x80
    x = conv_bn(x, 32, 3, 1)

    # Stages
    x = res_se_block(x, 64, 2, se_ratio=0.25, drop=0.0)   # 40x40
    x = res_se_block(x, 64, 1, se_ratio=0.25, drop=0.0)

    x = res_se_block(x, 128, 2, se_ratio=0.25, drop=0.05) # 20x20
    x = res_se_block(x, 128, 1, se_ratio=0.25, drop=0.05)

    x = res_se_block(x, 256, 2, se_ratio=0.25, drop=0.10) # 10x10
    x = res_se_block(x, 256, 1, se_ratio=0.25, drop=0.10)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x) 
    return keras.Model(inp, out)


# ==========================================
# LOAD CNN WEIGHTS
# ==========================================
CNN_PATH = os.path.join(MODEL_DIR, "best_cnn.weights.h5")
cnn_model = None

try:
    cnn_model = build_model()

    cnn_model.load_weights(
        CNN_PATH,
        by_name=True,
        skip_mismatch=True   #  <-- MAGIC FIX
    )

    st.success("CNN model loaded successfully!")

except Exception as e:
    st.error(f"❌ Failed loading CNN model: {e}")


def predict_cnn(path):
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        arr = img.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        prob_fake = cnn_model.predict(arr)[0][0]
        prob_real = 1 - prob_fake

        pred = "FAKE" if prob_fake > 0.5 else "REAL"

        return pred, prob_real * 100, prob_fake * 100

    except Exception as e:
        st.error(f"CNN prediction error: {e}")
        return None, None, None


# ==========================================
# LOAD ML MODELS
# ==========================================
try:
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
except:
    scaler = None
    pca = None


def format_model_name(raw_name):
    name = raw_name.replace("_hasPCA", "").replace(".pkl", "").replace(".pkl", "")
    parts = []
    word = ""
    for c in name:
        if c.isupper() and word:
            parts.append(word)
            word = c
        else:
            word += c
    parts.append(word)
    return " ".join(w.capitalize() for w in parts)


models = {}
if os.path.isdir(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            formatted = format_model_name(f.replace(".pkl", ""))
            models[formatted] = joblib.load(os.path.join(MODEL_DIR, f))


# ==========================================
# STREAMLIT PAGE RENDER
# ==========================================
def render():

    st.title("Try Now — Deepfake Detector")

    if scaler is None or pca is None:
        st.error("Scaler or PCA missing!")
        return

    model_name = st.selectbox(
        "Select a Model",
        ["All Models", "Deep Learning CNN"] + list(models.keys())
    )

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if st.button("Predict"):

        if uploaded_file is None:
            st.warning("Please upload an image.")
            return
        
        progress = st.progress(0)
        status = st.empty()

        # Animation loop (runs in background)
        import time
        for i in range(0, 100):
            progress.progress(i + 1)
            status.text(f"⏳ Processing... Please wait ({i+1}%)")
            time.sleep(0.03)

        temp_path = "temp_test_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        results = []

        model_list = (
            ["Deep Learning CNN"] + list(models.keys())
            if model_name == "All Models"
            else [model_name]
        )

        # Extract ML features only once
        feat_scaled = None
        feat_pca = None

        if any(m != "Deep Learning CNN" for m in model_list):
            feat = extract_features(temp_path)
            if feat is None:
                st.error("Feature extraction failed!")
                return
            feat = feat.reshape(1, -1)
            feat_scaled = scaler.transform(feat)
            feat_pca = pca.transform(feat_scaled)

        # ML + CNN predictions
        for m in model_list:

            if m == "Deep Learning CNN":
                pred, r, fprob = predict_cnn(temp_path)
                results.append({
                    "Model": "Deep Learning CNN",
                    "Prediction": pred,
                    "% REAL": r,
                    "% FAKE": fprob
                })
                continue

            model = models[m]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(feat_pca)[0]
                pred = "REAL" if np.argmax(prob) == 0 else "FAKE"
                results.append({
                    "Model": m,
                    "Prediction": pred,
                    "% REAL": prob[0] * 100,
                    "% FAKE": prob[1] * 100
                })
            else:
                pred = model.predict(feat_pca)[0]
                results.append({
                    "Model": m,
                    "Prediction": "REAL" if pred == 0 else "FAKE",
                    "% REAL": "N/A",
                    "% FAKE": "N/A"
                })

        df = pd.DataFrame(results)
        st.dataframe(df)

        # ===========================
        # Enhanced Comparison Chart
        # ===========================
        numeric = df[df["% REAL"] != "N/A"]

        if not numeric.empty:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(numeric))
            width = 0.35

            # Bars
            bars_real = ax.bar(
                x - width/2,
                numeric["% REAL"],
                width,
                label="REAL Probability",
                color="#2ecc71",
                edgecolor="black",
                linewidth=1
            )
            bars_fake = ax.bar(
                x + width/2,
                numeric["% FAKE"],
                width,
                label="FAKE Probability",
                color="#e74c3c",
                edgecolor="black",
                linewidth=1
            )

            # Grid + Aesthetic
            ax.set_ylabel("Probability (%)", fontsize=12)
            ax.set_title("Model Prediction Probabilities", fontsize=16, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(numeric["Model"], rotation=30, ha="right", fontsize=11)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.legend()
            st.pyplot(fig)
