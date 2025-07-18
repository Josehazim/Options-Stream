import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.covariance import LedoitWolf
import time

# Set API keys (hidden in UI, but can be set via env)
os.environ['ALPHA_VANTAGE_API_KEY'] = os.getenv('ALPHA_VANTAGE_API_KEY', '8TMX28PUWT08NRVQ')
os.environ['FRED_API_KEY'] = os.getenv('FRED_API_KEY', 'ae459a7a4cfcda809bac0750dbba86e3')
os.environ['NASDAQ_API_KEY'] = os.getenv('NASDAQ_API_KEY', 'n7MKRy2LWyKMbJxwzsGB')
os.environ['IEX_API_KEY'] = os.getenv('IEX_API_KEY', '14d67d65a7de42698134f02cdf8752aa')

st.set_page_config(page_title="Tres Esperti Dashboard", layout="wide", page_icon="📈")
st.title("📈 Tres Esperti Options Analytics Dashboard")
st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 8px;}
    .stDataFrame {background-color: #fff; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_yfinance_universe():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VXX']
    return list(sp500['Symbol']) + etfs

@st.cache_data(show_spinner=False)
def get_cboe_universe():
    url = "https://cdn.cboe.com/api/global/delayed_quotes/symbol_directory/options_symbols_list.csv"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        return list(df['Root Symbol'].unique())
    except Exception as e:
        return []

y_universe = get_yfinance_universe()
cboe_universe = get_cboe_universe()
UNIVERSE = sorted(set(y_universe).union(set(cboe_universe)))

# --- Analytics Functions ---
def mavd_signal(df, N=20):
    returns = df['Close'].pct_change()
    mom_score = returns.rolling(N).mean() / returns.rolling(N).std()
    hv_20 = returns.rolling(20).std() * np.sqrt(252)
    hv_rank = hv_20.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    return (mom_score > 0) & (hv_rank < 0.7)

def risk_parity_gauge(df):
    hv_30 = df['Close'].pct_change().rolling(30).std() * np.sqrt(252)
    iv_rank = hv_30.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    skew = df['Close'].rolling(5).mean() / df['Close'].rolling(30).mean()
    skew_rank = skew.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    std_iv = iv_rank.rolling(90).std()
    std_skew = skew_rank.rolling(90).std()
    w_iv = 1 / (std_iv + 1e-6)
    w_skew = 1 / (std_skew + 1e-6)
    score = (w_iv * iv_rank + w_skew * skew_rank) / (w_iv + w_skew)
    return score < 0.6

def zscore_flow(df, window=20):
    volume_z = zscore(df['Volume'].fillna(0).rolling(window).mean())
    flow = ((df['Close'] - df['Open']) * df['Volume'].fillna(0))
    flow_z = zscore(flow.rolling(window).mean())
    return (volume_z > 1.5) & (flow_z > 1.5)

def screen_option_contracts(symbol, dte_min=30, dte_max=50, min_vol=500, min_oi=1000, max_spread=0.08):
    try:
        tk = yf.Ticker(symbol)
        chains = []
        for expiry in tk.options:
            dte = (pd.to_datetime(expiry) - pd.Timestamp.now()).days
            if not (dte_min <= dte <= dte_max):
                continue
            options = tk.option_chain(expiry)
            for typ, chain in [('calls', options.calls)]:
                for _, row in chain.iterrows():
                    mid = (row['ask'] + row['bid']) / 2 if row['ask'] + row['bid'] > 0 else 0.01
                    spread = (row['ask'] - row['bid']) / mid if mid else 0
                    if (row['volume'] >= min_vol and row['openInterest'] >= min_oi and spread < max_spread):
                        chains.append({
                            'expiry': expiry, 'strike': row['strike'],
                            'bid': row['bid'], 'ask': row['ask'],
                            'dte': dte, 'spread': spread,
                            'volume': row['volume'], 'oi': row['openInterest']
                        })
        return pd.DataFrame(chains)
    except Exception as e:
        return pd.DataFrame()

def hrp_weights(returns):
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    clusters = linkage(dist, 'ward')
    order = dendrogram(clusters, no_plot=True)['leaves']
    returns = returns.iloc[:, order]
    lw = LedoitWolf().fit(returns.fillna(0))
    var = np.diag(lw.covariance_)
    inv_var = 1 / var
    weights = inv_var / inv_var.sum()
    p = weights
    rho = corr.values
    neff = 1 / (p @ rho @ p)
    return dict(zip(returns.columns, weights)), neff

def monte_carlo_cvar(spot, mu, sigma, long_strike, short_strike, dte, n=5000):
    dt = dte / 252
    pnl = []
    for _ in range(n):
        path = spot * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn())
        payoff = max(path - long_strike, 0) - max(path - short_strike, 0)
        pnl.append(payoff)
    pnl = np.array(pnl)
    tail_losses = np.sort(pnl)[:int(0.05 * n)]
    cvar = tail_losses.mean()
    return cvar

def kelly_fraction(expect_return, cvar, cap=0.5):
    f = expect_return / (cvar ** 2 + 1e-6)
    return min(max(f, 0), cap)

# --- UI Controls ---
st.sidebar.header("Universe & Filters")
symbols = st.sidebar.multiselect("Select symbols to screen", UNIVERSE, default=UNIVERSE[:10])
dte_min = st.sidebar.slider("Min DTE", 10, 90, 30)
dte_max = st.sidebar.slider("Max DTE", dte_min, 120, 50)
min_vol = st.sidebar.number_input("Min Option Volume", 0, 5000, 500)
min_oi = st.sidebar.number_input("Min Open Interest", 0, 10000, 1000)
max_spread = st.sidebar.slider("Max Bid-Ask Spread", 0.01, 0.5, 0.08)

run_screen = st.sidebar.button("Run Tres Esperti Pipeline 🚀")

if run_screen:
    st.info("Screening symbols, please wait... (this may take a few minutes)")
    qualified = []
    progress = st.progress(0)
    for i, sym in enumerate(symbols):
        st.write(f"Screening {sym}...")
        try:
            df = yf.download(sym, period='2y', interval='1d')
        except Exception as e:
            st.warning(f"Failed to download {sym}: {e}")
            continue
        if len(df) < 252:
            continue
        try:
            if not (mavd_signal(df).iloc[-1].item()):
                continue
            if not (risk_parity_gauge(df).iloc[-1].item()):
                continue
            if not (zscore_flow(df)[-1].item()):
                continue
        except Exception as e:
            continue
        contracts = screen_option_contracts(sym, dte_min, dte_max, min_vol, min_oi, max_spread)
        if contracts.empty:
            continue
        contract = contracts.sort_values('spread').iloc[0]
        qualified.append({'symbol': sym, 'contract': contract})
        progress.progress((i+1)/len(symbols))
        time.sleep(0.5)
    progress.empty()

    if not qualified:
        st.error("No qualified trades at this time.")
    else:
        st.success(f"Found {len(qualified)} qualified trades!")
        assets = [q['symbol'] for q in qualified]
        returns = pd.DataFrame({
            a: yf.download(a, period='3mo')['Close'].pct_change().dropna()
            for a in assets
        })
        weights, neff = hrp_weights(returns)
        st.subheader("Portfolio Weights & Analytics")
        st.dataframe(pd.DataFrame({'Weight': weights}))
        st.metric("Effective Number of Bets (Neff)", f"{neff:.2f}")
        st.divider()
        st.subheader("Qualified Option Contracts")
        for q in qualified:
            sym = q['symbol']
            contract = q['contract']
            try:
                spot = yf.Ticker(sym).history(period='1d')['Close'].iloc[-1]
            except Exception as e:
                spot = np.nan
            est_mu = returns[sym].mean() * 252 if sym in returns else np.nan
            est_sigma = returns[sym].std() * np.sqrt(252) if sym in returns else np.nan
            cvar = monte_carlo_cvar(
                spot, est_mu, est_sigma,
                contract['strike'], contract['strike'] + 5,
                contract['dte']
            ) if not np.isnan(spot) else np.nan
            expect_return = (contract['ask'] - contract['bid'])
            f_kelly = kelly_fraction(expect_return, cvar) if not np.isnan(cvar) else np.nan
            with st.expander(f"{sym} | {contract['expiry']} | Strike: {contract['strike']}"):
                st.write(contract)
                st.metric("Kelly Fraction", f"{f_kelly:.2%}" if not np.isnan(f_kelly) else "N/A")
                st.metric("CVaR", f"${cvar:.2f}" if not np.isnan(cvar) else "N/A")
                st.metric("Expected Return (Bid-Ask)", f"${expect_return:.2f}")
else:
    st.info("Select symbols and click 'Run Tres Esperti Pipeline' to begin.")
