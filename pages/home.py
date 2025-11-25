import streamlit as st

def render():
    # ===== GLOBAL PAGE CSS =====
    st.markdown("""
    <style>
        html { scroll-behavior: smooth; }

        /* Fade-in animation */
        .fade-in-section {
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 1s ease forwards;
        }
        @keyframes fadeInUp {
            to { opacity: 1; transform: translateY(0); }
        }

        /* Blue references box */
        .ref-box {
            background-color: #e6f2ff;
            padding: 20px;
            border-radius: 12px;
            text-align: left;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- Introduction ----
    st.markdown("<h2 id='intro' class='fade-in-section' style='color:#191970;'>Introduction to Deepfakes</h2>", unsafe_allow_html=True)
    st.write("""
    Deepfakes are AI-generated synthetic media where a person's likeness is swapped 
    onto another individual using Deep Learning models.  
    These techniques can create realistic images and videos that are extremely difficult to distinguish from genuine content.
    """)

    # ---- Impact ----
    st.markdown("<h2 id='impact' class='fade-in-section' style='color:#191970;'>Impact of Deepfakes</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Large-scale misinformation and fake news  
    - Threats to personal privacy and security  
    - Political manipulation & election interference  
    - Fraud and identity scams  
    - Reputation damage & reduced public trust  
    """)

    # ---- Project Objectives ----
    st.markdown("<h2 id='objectives' class='fade-in-section' style='color:#191970;'>Project Objectives</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Build ML & DL models for detecting deepfake images  
    - Evaluate accuracy and performance of detection algorithms  
    - Conduct thorough EDA and preprocessing  
    - Create an interactive Streamlit-based testing demo  
    """)

    # ---- References ----
    st.markdown("<h2 id='references' class='fade-in-section' style='color:#191970;'>References</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ref-box">
        <ul>
            <li>Korshunov, P., & Marcel, S. (2018). Deepfakes: A new threat to face recognition. <i>arXiv:1812.08685</i></li>
            <li>Nguyen, H., et al. (2019). Deep learning for deepfakes creation and detection. <i>IEEE TIFS</i></li>
            <li>Rossler, A., et al. (2019). FaceForensics++: Manipulated facial image detection. <i>ICCV</i></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
