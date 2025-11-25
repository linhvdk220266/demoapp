import streamlit as st

def inject_style():
    st.markdown("""
    <style>
    /* ---- Hide Streamlit Default Header & Footer ---- */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none !important;}
    button[kind="header"] {display: none !important;}

    /* ---- Body & Block Container Full Height ---- */
    .block-container {
        padding: 10 !important;
        margin: 0 auto !important;
        height: 100vh !important; /* full screen height */
        display: flex;
        align-items: center;      /* vertical center */
        flex-direction: column;   /* stack elements vertically */
        text-align: left;
    }

    /* ---- Content Box ---- */
    .content-box {
        max-width: 900px;
        padding: 20px 40px;
    }

    /* ---- Tabs Styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 50px;
        border-bottom: 2px solid #191970;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 24px !important;
        padding: 16px 32px !important;
        color: #191970 !important;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: white !important;
        background-color: #191970 !important;
        border-radius: 8px 8px 0px 0px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #19197040 !important;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
