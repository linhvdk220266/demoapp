import streamlit as st

def render():

    # ===== GLOBAL STYLE =====
    st.markdown("""
    <style>
        html { 
            scroll-behavior: smooth; 
        }

        /* Fade-in animation */
        .fade-in-section {
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 1s ease forwards;
        }
        @keyframes fadeInUp {
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }

        /* Blue references box */
        .ref-box {
            background-color: #e6f2ff;
            padding: 20px;
            border-radius: 12px;
            text-align: left;
        }

        /* Page content alignment */
        h1, h2, h3, p {
            text-align: left !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ===== PAGE TITLE =====
    st.markdown("<h1 class='fade-in-section'>Machine Learning & Deep Learning Models</h1>", unsafe_allow_html=True)

    # ===== TITLES ONLY (no content) =====
    st.markdown("Overview of Implemented Models")

    st.markdown("Machine Learning (ML) Models")

    st.markdown("Deep Learning (DL) Model")

    st.markdown("ML vs. DL Performance Comparison")

