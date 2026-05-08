# master_app.py
"""
Master Streamlit app for ECO 317 Capstone Macroeconomic Engine (Assignment 4).
Entry point: streamlit run master_app.py
"""
import streamlit as st
from modules import solow, empirical, micro, dsge
st.set_page_config(page_title="Capstone Macroeconomic Engine", layout="wide")

# Custom CSS: dark navy background, Garamond font
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Garamond');
    html, body, [class*="css"]  {font-family: Garamond, serif;}
    .stApp { background-color: #071033; color: white; }
    .block-container { padding: 1rem 2rem; }
    .sidebar .sidebar-content { background-color: #071033; color: white; }
    .streamlit-expanderHeader { color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = {
    "Empirical Data Suite": empirical.app,
    "Solow Growth Model": solow.app,
    "Micro-Founded Models": micro.app,
    "DSGE & Fiscal Policy": dsge.app,
}

st.sidebar.title("Capstone Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("ECO 317 — Capstone Macroeconomic Engine")

# Render page
PAGES[page]()
