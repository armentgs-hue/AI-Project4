# modules/solow.py
"""
Solow module: simulate Solow growth model with Harrod-neutral tech progress.
Functions:
- simulate_solow(): simulate time-series of k_t, y_t, c_t
- steady_state_k(): analytic steady-state for k*
- app(): Streamlit page
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def steady_state_k(s, delta, n, g, alpha):
    """
    Compute steady-state capital per effective worker k*
    for Cobb-Douglas Y = K^alpha (A L)^(1-alpha)
    k* = [ s / (n + g + delta) ]^(1/(1-alpha))
    """
    denom = n + g + delta
    return (s / denom) ** (1.0 / (1.0 - alpha))

@st.cache_data
def simulate_solow(s, delta, n, g, alpha, k0, T=200):
    """
    Simulate Solow transition in per-effective-worker terms.
    Uses discrete-time law of motion:
    k_{t+1} = (1/(1+g)) * [ (1 - delta) * k_t + s * f(k_t) ] / (1 + n)
    But more standard discrete form:
    k_{t+1} = ( (1 - delta) * k_t + s * k_t**alpha ) / (1 + n) / (1 + g)
    We'll implement the canonical per-effective-worker update:
    k_{t+1} = ( (1 - delta) * k_t + s * k_t**alpha ) / (1 + n) / (1 + g)
    """
    k = np.zeros(T+1)
    y = np.zeros(T+1)
    c = np.zeros(T+1)
    k[0] = k0
    for t in range(T):
        y[t] = k[t] ** alpha
        invest = s * y[t]
        k_next = ((1 - delta) * k[t] + invest) / ((1 + n) * (1 + g))
        k[t+1] = max(k_next, 1e-12)
        c[t] = (1 - s) * y[t]
    y[-1] = k[-1] ** alpha
    c[-1] = (1 - s) * y[-1]
    df = pd.DataFrame({
        'k': k, 'y': y, 'c': c
    }, index=np.arange(T+1))
    return df

def golden_rule_alpha_dependent(alpha):
    """
    For Cobb-Douglas, golden-rule saving rate s_gr = alpha.
    Returns s_gr.
    """
    return alpha

def solow_diagram(s, delta, n, g, alpha, k_grid=None):
    """
    Returns matplotlib figure with s*f(k) and break-even line.
    """
    if k_grid is None:
        k_grid = np.linspace(1e-6, 10, 500)
    f = k_grid ** alpha
    invest = s * f
    break_even = (n + g + delta) * k_grid

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(k_grid, invest, label=r"$s f(k)$", color="#1f77b4")
    ax.plot(k_grid, break_even, label=r"$(n+g+\delta)k$", color="#ff7f0e")
    # steady state
    try:
        kss = steady_state_k(s, delta, n, g, alpha)
        ax.axvline(kss, linestyle='--', color='green', label=r"$k^*$")
    except Exception:
        kss = None
    ax.set_xlabel("k (capital per effective worker)")
    ax.set_ylabel("Investment / Break-even")
    ax.set_title("Solow Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, kss

def app():
    st.header("Solow Growth Model")
    st.markdown("Interactive Solow model with Harrod-neutral technological progress.")
    col1, col2 = st.columns([1,1])

    with col1:
        s = st.slider("Saving rate, s", 0.01, 0.6, 0.2, step=0.01)
        delta = st.slider("Depreciation rate, δ", 0.001, 0.2, 0.05, step=0.001)
        n = st.slider("Population growth rate, n", 0.0, 0.05, 0.01, step=0.001)
        g = st.slider("Tech growth rate, g", 0.0, 0.05, 0.02, step=0.001)
    with col2:
        alpha = st.slider("Capital share, α", 0.1, 0.5, 0.33, step=0.01)
        T = st.number_input("Simulation periods (T)", min_value=50, max_value=2000, value=200, step=10)
        k0 = st.number_input("Initial k0", min_value=1e-6, value=0.5, step=0.1, format="%.6f")

    # Simulate
    df = simulate_solow(s, delta, n, g, alpha, k0, T=int(T))
    kss = steady_state_k(s, delta, n, g, alpha)
    sgr = golden_rule_alpha_dependent(alpha)

    st.subheader("Transition Dynamics")
    st.write(f"Steady-state k*: **{kss:.4f}** — Golden-rule saving rate s_gr = **{sgr:.3f}**")

    # Time series plots
    fig, ax = plt.subplots(3,1, figsize=(8,9), sharex=True)
    ax[0].plot(df.index, df['k'], color="#66c2a5"); ax[0].set_ylabel("k_t")
    ax[1].plot(df.index, df['y'], color="#fc8d62"); ax[1].set_ylabel("y_t")
    ax[2].plot(df.index, df['c'], color="#8da0cb"); ax[2].set_ylabel("c_t"); ax[2].set_xlabel("Period")
    ax[0].axhline(kss, color='green', linestyle='--', label='k*')
    ax[0].legend()
    st.pyplot(fig)

    # Solow diagram
    fig2, kss_line = solow_diagram(s, delta, n, g, alpha)
    st.subheader("Solow Diagram")
    st.pyplot(fig2)

    # Button: shock
    if st.button("Simulate sudden capital destruction (50% instantaneous loss)"):
        k_shock = df.copy()
        k_shock['k'] = k_shock['k'].values
        k_shock.loc[1:, 'k'] = k_shock['k'].values
        k_shock.loc[0, 'k'] = k_shock.loc[0, 'k'] * 0.5
        # re-simulate from shocked k0
        df2 = simulate_solow(s, delta, n, g, alpha, k_shock.loc[0,'k'], T=int(T))
        fig3, ax3 = plt.subplots(1,1, figsize=(8,3))
        ax3.plot(df['k'], label='Baseline k')
        ax3.plot(df2['k'], label='After shock k', linestyle='--')
        ax3.axhline(kss, color='green', linestyle=':', label='k*')
        ax3.set_ylabel('k')
        ax3.set_xlabel('Period')
        ax3.legend()
        st.pyplot(fig3)

    # Intelligent summary
    st.markdown("### Intelligent Summary")
    last_k = df['k'].iloc[-1]
    reached = abs(last_k - kss) < 1e-3
    if reached:
        st.write(f"The economy converges close to steady state by period {int(T)} (k ≈ {last_k:.4f}, k* = {kss:.4f}).")
    else:
        st.write(f"The economy has not fully converged in {int(T)} periods (k_T = {last_k:.4f}, k* = {kss:.4f}). Changing s, n, g, or δ shifts k* as shown in the Solow diagram.")
