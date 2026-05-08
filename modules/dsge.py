# modules/dsge.py
"""
DSGE & Fiscal Policy (lightweight medium-scale skeleton)
- Linearized RBC/SMETS-style block with simple shock simulation via state-space
- IRF plotting for user-selected fiscal shock (government spending)
- Computes impact and cumulative multipliers (approximate)
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

@st.cache_data
def simple_dsge_irf(T=40, rho=0.8, shock_size=1.0, shock_type='g'):
    # A minimal AR(1) representation for aggregate demand where output responds to shock
    y = np.zeros(T)
    k = np.zeros(T)
    g = np.zeros(T)
    y[0] = shock_size
    g[0] = shock_size if shock_type=='g' else 0.0
    for t in range(1, T):
        y[t] = rho * y[t-1] + (0.1 if shock_type=='g' else 0.0) * g[t-1]
        g[t] = rho * g[t-1]
    return pd.DataFrame({'y': y, 'g': g})

def compute_multipliers(irf, baseline_output=1.0):
    impact = irf['y'].iloc[0] / irf['g'].iloc[0] if irf['g'].iloc[0]!=0 else np.nan
    cumulative = irf['y'].sum() / irf['g'].sum() if irf['g'].sum()!=0 else np.nan
    return {'impact_multiplier': float(impact), 'cumulative_multiplier': float(cumulative)}

def app():
    st.header("DSGE & Fiscal Policy")
    st.markdown("Lightweight DSGE skeleton: simulate a government spending shock and view IRFs and multipliers.")
    col1, col2 = st.columns([1,1])
    with col1:
        rho = st.slider("Persistence (ρ)", 0.0, 0.99, 0.8, step=0.01)
        shock = st.slider("Shock size (units)", 0.01, 5.0, 1.0, step=0.01)
        T = st.number_input("IRF periods", min_value=10, max_value=200, value=40, step=5)
    with col2:
        shock_type = st.selectbox("Shock type", ["Government Spending (g)", "Labor tax (t)"], index=0)
        shock_key = 'g' if shock_type.startswith("Government") else 't'

    irf = simple_dsge_irf(T=int(T), rho=float(rho), shock_size=float(shock), shock_type=shock_key)
    st.subheader("Impulse Response Functions")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(irf['y'], label='Output')
    ax.plot(irf['g'], label='Gov. spending' if shock_key=='g' else 'Tax')
    ax.legend()
    ax.set_xlabel("Periods")
    st.pyplot(fig)

    mul = compute_multipliers(irf)
    st.write("Multipliers (approximate)")
    st.json(mul)

    st.markdown("### Intelligent Summary")
    st.write(f"A {shock} unit {shock_type} shock with persistence {rho} yields impact multiplier ≈ {mul['impact_multiplier']:.3f} and cumulative ≈ {mul['cumulative_multiplier']:.3f}. Replace skeleton solver with a full Smets-Wouters implementation for research-grade analysis.")

