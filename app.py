import streamlit as st
from utils.style import inject_style
from pages import home, dataset, processing, models, tests

st.set_page_config(page_title="Deepfake Detection App", layout="wide")
inject_style()

# ===== Top Navigation (Tabs) =====
tabs = st.tabs(["Home", "Data Exploration", "Feature Extraction", "Models", "Test Your Image"])

with tabs[0]:
    home.render()

with tabs[1]:
    dataset.render()

with tabs[2]:
    processing.render()

with tabs[3]:
    models.render()

with tabs[4]:
    tests.render()