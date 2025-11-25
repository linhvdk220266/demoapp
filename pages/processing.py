import streamlit as st

def render():
    
    st.markdown("""
    <style>
        html { scroll-behavior: smooth; }

        /* Fade-in animation */
        .fade-in-section {
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.9s ease forwards;
        }
        @keyframes fadeInUp {
            to { opacity: 1; transform: translateY(0); }
        }

        /* Blue info box */
        .info-box {
            background-color: #e6f2ff;
            padding: 18px;
            border-radius: 12px;
            margin-top: 10px;
            text-align: left;
        }

        /* Headings alignment */
        h1, h2, h3, p, li {
            text-align: left !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Pre-Processing & Feature Extraction")

    # Section 1 — Overview
    st.header("Overview of Pre-Processing")
    st.write("""
    Pre-processing prepares all images before feeding them into Machine Learning (ML) 
    and Deep Learning (DL) models. The goal is to clean, standardize, and transform 
    raw face images so that feature extraction and model training become more reliable.
    """)

    # Section 2 — Steps
    st.header("Pre-Processing Steps")
    st.subheader("• Face Detection & Cropping")


    st.subheader("• Resizing & Normalization")
 

    st.subheader("• Data Augmentation (DL models only)")
 

    # Section 3 — Feature Extraction
    st.header("Feature Extraction (ML Models)")
    

    st.subheader("• PRNU (Photo-Response Non-Uniformity)")
 

    st.subheader("• LBP (Local Binary Patterns)")
  

    st.subheader("• DCT (Discrete Cosine Transform)")
 

    st.subheader("• FFT (Fast Fourier Transform)")
  

    # Section 4 — Summary
    st.header("Summary")
 


