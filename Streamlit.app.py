import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import hashlib

# ==================== CONFIGURATION ====================

BENCHMARK_NAME = 'NIFTY 50'

SECTORS_CONFIG = {
    # (Your existing sectors config from original code here)
    # For brevity, add sectors dict as in your original code snippet
}

# ==================== USER CACHE KEY FOR PER-USER DATA ====================

def get_user_cache_key(api_key):
    return hashlib.md5(api_key.encode()).hexdigest()[:8]

# ==================== LOAD BENCHMARK FROM INVESTING.COM UPLOAD ====================

def load_nifty50_from_investing(file=None):
    if file is None:
        file = st.file_uploader("Upload Nifty 50 Historical CSV from Investing.com", type=['csv'])
        if file is None:
            st.info("Please upload Nifty 50 historical CSV data from Investing.com to proceed")
            return None
    try:
        df = pd.read_csv(file)
        # Expect columns: Date, Open, High, Low, Close
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df[['Close']].rename(columns={'Close': BENCHMARK_NAME})
        return df
    except Exception as e:
        st.error(f"Failed to load Nifty 50 data: {e}")
        return None

# ==================== TWELVE DATA API ====================

def fetch_twelve_data(symbol, api_key, start_date, end_date, user_key):
    @st.cache_data(ttl=86400, show_spinner=False)
    def _fetch_cached(symbol, api_key_hash, start_str, end_str):
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': '1day',
            'start_date': start_str,
            'end_date': end_str,
            'apikey': api_key,
            'exchange': 'NSE',
            'format': 'JSON',
            'outputsize': 5000
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'code' in data:
                        if data['code'] == 429:
                            return {'error': 'rate_limit'}
                        elif data['code'] == 400:
                            return {'error': 'not_found', 'symbol': symbol}
                    if 'values' in data and data['values']:
                        df = pd.DataFrame(data['values'])
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df.set_index('datetime').sort_index()
                        df['close'] = pd.to_numeric(df['close'])
                        df = df[['close']].rename(columns={'close': symbol})
                        return {'success': True, 'data': df}
                elif response.status_code == 429:
                    time.sleep(5)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return {'error': 'fetch_failed', 'message': str(e)}
        return {'error': 'fetch_failed', 'message': 'Max retries exceeded'}
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    result = _fetch_cached(symbol, user_key, start_str, end_str)
    if isinstance(result, dict):
        if 'success' in result:
            return result['data']
        elif 'error' in result:
            if result['error'] == 'rate_limit':
                st.warning(f"⚠️ API rate limit reached for your key. Please wait...")
            elif result['error'] == 'not_found':
                pass
            return None
    return None

def fetch_all_twelve_data(api_key, user_key, start_date, end_date):
    all_data = dict()
    failed_symbols = []

    # Fetch sectors
    for sector_key, sector_info in SECTORS_CONFIG.items():
        sector_df = fetch_twelve_data(sector_info['symbol'], api_key, start_date, end_date, user_key)
        if sector_df is not None:
            all_data[sector_key] = sector_df
        else:
            failed_symbols.append(sector_info['name'])
        # Fetch stocks
        for stock in sector_info['stocks']:
            stock_df = fetch_twelve_data(stock, api_key, start_date, end_date, user_key)
            if stock_df is not None:
                all_data[stock] = stock_df
            else:
                failed_symbols.append(stock)
    if len(all_data) < 10:
        st.error("Not enough data fetched from Twelve Data.")
        return None
    combined_df = pd.concat(all_data.values(), axis=1, join='outer')
    combined_df.columns = list(all_data.keys())
    combined_df = combined_df.sort_index()
    return combined_df

# ==================== RRG CALCULATIONS ====================

def calculate_rrg_metrics(data, benchmark=BENCHMARK_NAME, window=26):
    if data is None or benchmark not in data.columns:
        return None, None
    benchmark_prices = data[benchmark]
    rsr_data, rsm_data = {}, {}
    for col in data.columns:
        if col == benchmark:
            continue
        rs = data[col] / benchmark_prices
        rs_mean = rs.rolling(window=window).mean()
        rs_mean_of_mean = rs_mean.rolling(window=window).mean()
        rs_std = rs_mean.rolling(window=window).std().clip(lower=1e-10)
        z_rsr = (rs_mean - rs_mean_of_mean) / rs_std
        rsr_data[col] = 100 + z_rsr
        roc_rs = rs_mean.pct_change() * 100
        roc_rs_mean = roc_rs.rolling(window=window).mean()
        roc_rs_std = roc_rs.rolling(window=window).std().clip(lower=1e-10)
        z_rsm = (roc_rs - roc_rs_mean) / roc_rs_std
        rsm_data[col] = 100 + z_rsm
    rsr_df = pd.DataFrame(rsr_data).dropna()
    rsm_df = pd.DataFrame(rsm_data).dropna()
    return rsr_df, rsm_df

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Indian Market RRG - Hybrid Edition",
        page_icon="🇮🇳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🇮🇳 Indian Market RRG Analysis - Hybrid Team Edition")

    # Step 1: Load Nifty 50 benchmark from Investing.com CSV
    nifty_df = load_nifty50_from_investing()
    if nifty_df is None:
        return

    # Step 2: Get Twelve Data API Key
    api_key = st.sidebar.text_input("Enter your Twelve Data API Key", type="password")
    if not api_key:
        st.sidebar.info("Please enter your Twelve Data API key to fetch stocks/sectors data")
        return
    user_key = get_user_cache_key(api_key)

    # Step 3: Fetch Twelve Data sector and stock data for last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    twelve_df = fetch_all_twelve_data(api_key, user_key, start_date, end_date)
    if twelve_df is None:
        return

    # Step 4: Combine benchmark (Nifty 50) and Twelve Data datasets
    combined_df = pd.concat([nifty_df, twelve_df], axis=1, join='outer').sort_index()
    combined_df = combined_df.ffill().bfill()

    # Step 5: Calculate RRG metrics
    rsr_df, rsm_df = calculate_rrg_metrics(combined_df)
    if rsr_df is None or rsm_df is None:
        st.error("Failed to calculate RRG metrics.")
        return

    # Step 6: Visualization - reuse your original RRG plot functions (not repeated here for brevity)
    # Example: fig = create_rrg_plot(rsr_df, rsm_df, list(SECTORS_CONFIG.keys()))
    # st.plotly_chart(fig, use_container_width=True)

    st.success("Data fetched and RRG metrics calculated successfully! Implement chart rendering here.")

if __name__ == "__main__":
    main()
