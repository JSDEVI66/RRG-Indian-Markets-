import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import hashlib

# ==================== CONFIG ====================
BENCHMARK_NAME = 'NIFTY 50'
SECTOR_SYMBOL = 'NIFTY BANK'
STOCK_SYMBOL = 'HDFCBANK'

def get_user_cache_key(api_key):
    return hashlib.md5(api_key.encode()).hexdigest()[:8]

# ==================== FETCH NIFTY 50 ====================
@st.cache_data(ttl=3600)
def fetch_nifty50_yfinance():
    ticker = '^NSEI'
    df = yf.Ticker(ticker).history(period='1y')
    df = df[['Close']].rename(columns={'Close': BENCHMARK_NAME})
    df.index.name = 'Date'
    return df

# ==================== TWELVE DATA FETCH ====================
def fetch_twelve_data(symbol, api_key, start_date, end_date, user_key):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': '1day',
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'apikey': api_key,
        'exchange': 'NSE',
        'format': 'JSON',
        'outputsize': 300
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if 'values' not in data or not data['values']:
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df['close'] = pd.to_numeric(df['close'])
        df = df[['close']].rename(columns={'close': symbol})
        return df
    except Exception as e:
        st.error(f"Fetch error for {symbol}: {e}")
        return None

# ==================== MAIN APP ====================
def main():
    st.title("Minimal Indian Market RRG Test (Nifty 50, 1 Sector, 1 Stock)")

    nifty_df = fetch_nifty50_yfinance()
    st.success(f"Nifty 50 data loaded: {len(nifty_df)} rows.")

    api_key = st.sidebar.text_input("Enter Twelve Data API Key", type="password")
    if not api_key:
        st.stop()
    user_key = get_user_cache_key(api_key)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Fetch 1 sector
    sector_df = fetch_twelve_data(SECTOR_SYMBOL, api_key, start_date, end_date, user_key)
    if sector_df is not None:
        st.success(f"Sector '{SECTOR_SYMBOL}' fetched: {len(sector_df)} rows.")
        st.dataframe(sector_df.tail())
    else:
        st.error(f"Failed to fetch sector '{SECTOR_SYMBOL}'.")

    # Fetch 1 stock
    stock_df = fetch_twelve_data(STOCK_SYMBOL, api_key, start_date, end_date, user_key)
    if stock_df is not None:
        st.success(f"Stock '{STOCK_SYMBOL}' fetched: {len(stock_df)} rows.")
        st.dataframe(stock_df.tail())
    else:
        st.error(f"Failed to fetch stock '{STOCK_SYMBOL}'.")

    # Merge for sample
    combined_df = pd.concat([nifty_df, sector_df, stock_df], axis=1, join='outer').sort_index()
    st.subheader("Merged Sample Data")
    st.dataframe(combined_df.tail())

if __name__ == "__main__":
    main()
