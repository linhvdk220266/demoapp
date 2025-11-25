import streamlit as st

def render():
    st.markdown("<div class='content-box'>", unsafe_allow_html=True)

    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    Visualizations and analysis of dataset distribution, faces, PCA, etc.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
