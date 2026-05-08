# modules/micro.py
"""
Micro-Founded Dynamic Models
- Implements a Stochastic Consumption-Savings model solved by VFI (discrete asset grid)
- Provides simulation, moments (mean, var, autocorr), and simple shock path forecasting
"""
import streamlit as st
import numpy as np
import pandas as pd
from functools import lru_cache

@st.cache_data
def solve_consumption_savings(beta=0.96, r=0.04, y_states=(0.5, 1.0, 1.5), pi=None, grid_a=None, ucoef=1.0, maxiter=1000, tol=1e-6):
    # CRRA utility with sigma=2
    sigma = 2.0
    if pi is None:
        pi = np.ones((len(y_states), len(y_states))) / len(y_states)
    if grid_a is None:
        grid_a = np.linspace(0, 20, 200)
    na = len(grid_a)
    ny = len(y_states)
    V = np.zeros((ny, na))
    policy = np.zeros((ny, na))
    for it in range(maxiter):
        Vp = V.copy()
        for iy, y in enumerate(y_states):
            for ia, a in enumerate(grid_a):
                c_choices = (1 + r) * a + y - grid_a
                c_choices[c_choices <= 0] = 1e-12
                if sigma == 1:
                    u = np.log(c_choices)
                else:
                    u = (c_choices**(1 - sigma)) / (1 - sigma)
                cont = beta * (pi[iy] @ Vp)
                vals = u + cont
                policy[iy, ia] = grid_a[np.argmax(vals)]
                V[iy, ia] = np.max(vals)
        diff = np.max(np.abs(V - Vp))
        if diff < tol:
            break
    return {'V': V, 'policy': policy, 'grid_a': grid_a, 'y_states': y_states}

@st.cache_data
def simulate_policy(policy, grid_a, y_states, pi, a0=0.0, y0_idx=1, T=200):
    rng = np.random.default_rng(12345)
    ny = len(y_states)
    idx = y0_idx
    a_idx = np.argmin(np.abs(grid_a - a0))
    a_path = np.zeros(T)
    c_path = np.zeros(T)
    y_path = np.zeros(T)
    for t in range(T):
        a_now = grid_a[a_idx]
        # choose next asset as per policy (policy gives asset next period)
        a_next = policy[idx, a_idx]
        a_idx = np.argmin(np.abs(grid_a - a_next))
        y = y_states[idx]
        c = (1 + 0.04) * a_now + y - a_next
        a_path[t] = a_now
        c_path[t] = c
        y_path[t] = y
        # transition shock
        probs = pi[idx]
        idx = rng.choice(ny, p=probs)
    df = pd.DataFrame({'a': a_path, 'c': c_path, 'y': y_path})
    return df

def moments(df):
    return {
        'mean_c': float(df['c'].mean()),
        'var_c': float(df['c'].var()),
        'autocorr_c': float(df['c'].autocorr(lag=1))
    }

def app():
    st.header("Micro-Founded Dynamic Models")
    st.markdown("Stochastic Consumption-Savings model solved by Value Function Iteration (VFI).")
    col1, col2 = st.columns([1,1])
    with col1:
        beta = st.slider("Discount factor, β", 0.90, 0.995, 0.96, step=0.001)
        r = st.slider("Interest rate, r", 0.0, 0.10, 0.04, step=0.001)
        T = st.number_input("Simulation periods", min_value=50, max_value=2000, value=200, step=10)
    with col2:
        grid_size = st.selectbox("Asset grid size", [50, 100, 200, 500], index=2)
        sigma = st.slider("CRRA sigma", 1.0, 5.0, 2.0, step=0.1)

    # Build discrete income states and simple Markov
    y_states = np.array([0.5, 1.0, 1.5])
    pi = np.array([[0.9,0.08,0.02],[0.05,0.9,0.05],[0.02,0.08,0.9]])

    sol = solve_consumption_savings(beta=beta, r=r, y_states=y_states, pi=pi, grid_a=np.linspace(0,20,grid_size))
    st.success("VFI completed (cached).")
    if st.button("Simulate 100 period policy"):
        df = simulate_policy(sol['policy'], sol['grid_a'], sol['y_states'], pi, a0=0.1, y0_idx=1, T=int(T))
        st.line_chart(df[['a','c']])
        m = moments(df)
        st.write("Key moments")
        st.json(m)

    st.markdown("### Forecasting shock path")
    st.write("Enter a short shock path to force income states (indexes 0..2).")
    path = st.text_input("Shock path (comma-separated indexes)", value="2,0,1")
    if st.button("Run forecast"):
        path_idx = [int(x.strip()) for x in path.split(",") if x.strip()!='']
        # apply policy path deterministically for len(path)
        grid_a = sol['grid_a']
        policy = sol['policy']
        idx = 1
        a_idx = np.argmin(np.abs(grid_a - 0.1))
        sim_T = len(path_idx)
        a_path = []
        c_path = []
        for t in range(sim_T):
            idx = path_idx[t]
            a_now = grid_a[a_idx]
            a_next = policy[idx, a_idx]
            c = (1 + r) * a_now + y_states[idx] - a_next
            a_path.append(a_now); c_path.append(c)
            a_idx = np.argmin(np.abs(grid_a - a_next))
        df2 = pd.DataFrame({'a': a_path, 'c': c_path})
        st.write(df2)

