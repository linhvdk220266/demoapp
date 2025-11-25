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


# ==========================================
# Path Configuration 
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
PCA_PATH = os.path.join(BASE_DIR, "pca.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALE_FACTOR = 1.0


# ==========================================
# Feature Extractors (same as app.py)
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

    # spectral decay slope
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


# ===================================================
# Load Models (Loaded ONCE globally – performance OK)
# ===================================================
try:
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
except Exception as e:
    scaler = None
    pca = None

models = {}
if os.path.isdir(MODEL_DIR):
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    for f in files:
        name = f.replace("_hasPCA.pkl", "")
        try:
            models[name] = joblib.load(os.path.join(MODEL_DIR, f))
        except Exception as e:
            st.error(f"❌ Failed loading {f}: {e}")


# ===================================================
# RENDER FUNCTION (Streamlit page entry point)
# ===================================================
def render():

    st.title("Try Now — Deepfake Detector")

    if scaler is None or pca is None:
        st.error("Scaler or PCA not found! Make sure they exist in root folder.")
        return

    # --- Model Selection ---
    model_name = st.selectbox(
        "Select a Model",
        ["All Models"] + list(models.keys())
    )

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # ---------------------------
    # Predict
    # ---------------------------
    if st.button("Predict"):

        if uploaded_file is None:
            st.warning("Please upload an image.")
            return

        # Save temp
        temp_path = "temp_test_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        feat = extract_features(temp_path)

        if feat is None:
            st.error("Feature extraction failed. Invalid image?")
            return

        feat = feat.reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        feat_pca = pca.transform(feat_scaled)

        results = []

        model_list = list(models.keys()) if model_name == "All Models" else [model_name]

        for m in model_list:
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

        # Optional chart
        numeric = df[df["% REAL"] != "N/A"]
        if not numeric.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(numeric))
            ax.bar(x - 0.3, numeric["% REAL"], width=0.3, label="REAL", color="green")
            ax.bar(x + 0.3, numeric["% FAKE"], width=0.3, label="FAKE", color="red")
            ax.set_xticks(x)
            ax.set_xticklabels(numeric["Model"], rotation=45)
            ax.legend()
            st.pyplot(fig)
