# modules/empirical.py
"""
Empirical Data Suite
- Safe local demo using pandas_datareader (fallback) and yfinance (optional)
- Handles missing values, log-diff and YoY transforms
- Simple OLS regression widget (statsmodels)
- Recession shading using NBER series if available (optional)
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache

# Optional imports (lazy)
@st.cache_data
def fetch_yfinance(ticker, start):
    import yfinance as yf
    df = yf.download(ticker, start=start)
    df = df[['Adj Close']].rename(columns={'Adj Close': ticker})
    return df

@st.cache_data
def fetch_fred(series, start):
    # User must provide optional fredapi key in secret config for live calls.
    try:
        from fredapi import Fred
        import os
        key = st.secrets.get("FRED_API_KEY", None)
        if key is None:
            raise RuntimeError("FRED API key not found in secrets")
        fred = Fred(api_key=key)
        s = fred.get_series(series, observation_start=start)
        return pd.DataFrame({series: s})
    except Exception as e:
        raise

def transform_series(df, transform):
    if transform == "Level":
        return df
    if transform == "Log":
        return np.log(df).replace([np.inf, -np.inf], np.nan)
    if transform == "Log-diff (growth)":
        return np.log(df).diff().replace([np.inf, -np.inf], np.nan)
    if transform == "Year-over-Year %":
        return df.pct_change(12).replace([np.inf, -np.inf], np.nan)
    return df

def run_ols(y, X):
    import statsmodels.api as sm
    Xc = sm.add_constant(X)
    model = sm.OLS(y.dropna(), Xc.loc[y.dropna().index])
    res = model.fit()
    return res

def app():
    st.header("Empirical Data Suite")
    st.markdown("Fetch demo series via YahooFinance or use local upload. Live FRED calls require API key in Streamlit secrets (FRED_API_KEY).")
    col1, col2 = st.columns([2,1])
    with col1:
        source = st.selectbox("Data source", ["Demo (built-in)", "YahooFinance", "FRED (requires API key)"])
        start = st.date_input("Start date", value=pd.to_datetime("2000-01-01"))
        if source == "Demo (built-in)":
            # simple demo: US 10y yield proxy (constructed) and SP500
            dates = pd.date_range(start=start, end=datetime.today(), freq='ME')
            sp = pd.Series(1000 * (1 + 0.005) ** np.arange(len(dates)), index=dates, name="SP500")
            y10 = pd.Series(2 + 0.01 * np.sin(np.linspace(0, 10, len(dates))) , index=dates, name="T10Y")
            df = pd.concat([sp, y10], axis=1)
        elif source == "YahooFinance":
            tickers = st.text_input("Tickers (comma-separated)", value="^GSPC,GC=F")
            tickers = [t.strip() for t in tickers.split(",")][:4]
            dfs = []
            for t in tickers:
                try:
                    dfs.append(fetch_yfinance(t, start))
                except Exception as e:
                    st.error(f"Failed to fetch {t}: {e}")
            df = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()
        else:
            series = st.text_input("FRED series (comma-separated)", value="GDP,UNRATE")
            series = [s.strip() for s in series.split(",")][:4]
            dfs = []
            for s in series:
                try:
                    dfs.append(fetch_fred(s, start))
                except Exception as e:
                    st.error(f"FRED fetch failed for {s}: {e}")
            df = pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

        if df is None or df.empty:
            st.warning("No data available.")
            return

        st.write("First rows:")
        st.dataframe(df.dropna().head())

        transform = st.selectbox("Transformation", ["Level", "Log", "Log-diff (growth)", "Year-over-Year %"])
        df_t = transform_series(df, transform)
        st.line_chart(df_t.fillna(method='ffill'))

    with col2:
        st.subheader("Econometrics")
        y_var = st.selectbox("Dependent variable (y)", df.columns.tolist())
        x_vars = st.multiselect("Independent variables (X)", [c for c in df.columns if c != y_var], default=[c for c in df.columns if c != y_var][:1])
        if st.button("Run OLS"):
            y = df_t[y_var]
            X = df_t[x_vars]
            try:
                res = run_ols(y, X)
                st.write("Regression results")
                st.text(res.summary().as_text())
            except Exception as e:
                st.error(f"OLS failed: {e}")

    # Intelligent summary
    st.markdown("### Intelligent Summary")
    st.write(f"Data fetched: {', '.join(df.columns.tolist())}. Transform: **{transform}**. Check series frequency and missing values before inference.")
