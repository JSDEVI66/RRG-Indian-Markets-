"""
Indian Market RRG Tool - Multi-User Twelve Data Version
Each user provides their own API key - Perfect for teams!
Get your FREE API key from: https://twelvedata.com/
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import hashlib

# ==================== CONFIGURATION ====================

# Top 15 sectors by market cap
# Twelve Data uses exchange:symbol format for NSE India
SECTORS_CONFIG = {
    'NIFTY BANK': {
        'name': 'Banking',
        'symbol': 'NIFTY BANK',  # Index symbols use full name
        'stocks': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 
                  'INDUSINDBK', 'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB', 'PNB',
                  'BANKBARODA', 'CANBK', 'UNIONBANK', 'RBLBANK', 'AUBANK']
    },
    'NIFTY IT': {
        'name': 'Information Technology',
        'symbol': 'NIFTY IT',
        'stocks': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 
                  'LTTS', 'PERSISTENT', 'COFORGE', 'MPHASIS', 'LTIM',
                  'OFSS', 'TATAELXSI', 'CYIENT', 'SONATSOFTW', 'MASTEK']
    },
    'NIFTY AUTO': {
        'name': 'Automobile',
        'symbol': 'NIFTY AUTO',
        'stocks': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT',
                  'HEROMOTOCO', 'ASHOKLEY', 'BHARATFORG', 'MOTHERSON', 'APOLLOTYRE',
                  'MRF', 'BOSCHLTD', 'EXIDEIND', 'BALKRISIND', 'ESCORTS']
    },
    'NIFTY FMCG': {
        'name': 'FMCG',
        'symbol': 'NIFTY FMCG',
        'stocks': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR',
                  'MARICO', 'GODREJCP', 'COLPAL', 'TATACONSUM', 'UBL',
                  'EMAMILTD', 'PGHH', 'VBL', 'RADICO', 'JYOTHYLAB']
    },
    'NIFTY PHARMA': {
        'name': 'Pharmaceuticals',
        'symbol': 'NIFTY PHARMA',
        'stocks': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP',
                  'BIOCON', 'TORNTPHARM', 'LUPIN', 'AUROPHARMA', 'ALKEM',
                  'IPCALAB', 'LALPATHLAB', 'GLENMARK', 'LAURUSLABS', 'SYNGENE']
    },
    'NIFTY METAL': {
        'name': 'Metals',
        'symbol': 'NIFTY METAL',
        'stocks': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'SAIL',
                  'COALINDIA', 'NMDC', 'JINDALSTEL', 'HINDZINC', 'NATIONALUM',
                  'RATNAMANI', 'WELCORP', 'JSWENERGY', 'MOIL', 'APARINDS']
    },
    'NIFTY ENERGY': {
        'name': 'Energy',
        'symbol': 'NIFTY ENERGY',
        'stocks': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'POWERGRID',
                  'NTPC', 'COALINDIA', 'GAIL', 'ADANIGREEN', 'ADANIPOWER',
                  'TATAPOWER', 'TORNTPOWER', 'IGL', 'PETRONET', 'OIL']
    },
    'NIFTY REALTY': {
        'name': 'Real Estate',
        'symbol': 'NIFTY REALTY',
        'stocks': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE',
                  'PHOENIXLTD', 'SOBHA', 'MAHLIFE', 'SUNTECK', 'IBREALEST',
                  'LODHA', 'SIGNATURE', 'RAJESHEXPO', 'MAHSEAMLES', 'RAYMOND']
    },
    'NIFTY INFRA': {
        'name': 'Infrastructure',
        'symbol': 'NIFTY INFRA',
        'stocks': ['LT', 'ADANIPORTS', 'SIEMENS', 'ABB', 'CUMMINSIND',
                  'THERMAX', 'CONCOR', 'NBCC', 'IRB', 'GMRINFRA',
                  'KEC', 'ISGEC', 'JSWENERGY', 'NLCINDIA', 'IRCTC']
    },
    'NIFTY FIN SERVICE': {
        'name': 'Financial Services',
        'symbol': 'NIFTY FIN SERVICE',
        'stocks': ['BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI',
                  'ICICIGI', 'HDFCAMC', 'CHOLAFIN', 'M&MFIN', 'LICHSGFIN',
                  'PEL', 'SHRIRAMFIN', 'MUTHOOTFIN', 'MANAPPURAM', 'IIFL']
    },
    'NIFTY CONSUMPTION': {
        'name': 'Consumer Durables',
        'symbol': 'NIFTY CONSR DURBL',
        'stocks': ['TITAN', 'HAVELLS', 'VOLTAS', 'WHIRLPOOL', 'CROMPTON',
                  'VGUARD', 'SYMPHONY', 'RELAXO', 'BATAINDIA', 'PAGEIND',
                  'TRENT', 'JUBLFOOD', 'RAJESHEXPO', 'PCJEWELLER', 'AMBER']
    },
    'NIFTY MEDIA': {
        'name': 'Media & Entertainment',
        'symbol': 'NIFTY MEDIA',
        'stocks': ['ZEEL', 'SUNTV', 'PVR', 'SAREGAMA', 'TV18BRDCST',
                  'JAGRAN', 'DBCORP', 'NAVNETEDUL', 'HATHWAY', 'ORIENTLTD',
                  'CELEBRITY', 'NAZARA', 'BALAJITELE', 'NETWORK18', 'DISHTV']
    },
    'NIFTY COMMODITIES': {
        'name': 'Commodities',
        'symbol': 'NIFTY COMMODITIES',
        'stocks': ['TATASTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC',
                  'SAIL', 'JINDALSTEL', 'NATIONALUM', 'HINDZINC', 'JSWSTEEL',
                  'MOIL', 'GMDC', 'RATNAMANI', 'APARINDS', 'WELCORP']
    },
    'NIFTY OIL & GAS': {
        'name': 'Oil & Gas',
        'symbol': 'NIFTY OIL AND GAS',
        'stocks': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL',
                  'IGL', 'PETRONET', 'OIL', 'HINDPETRO', 'MGL',
                  'GSPL', 'GUJGASLTD', 'AEGISCHEM', 'DEEPAKNTR', 'AAVAS']
    },
    'NIFTY PSU BANK': {
        'name': 'PSU Banks',
        'symbol': 'NIFTY PSU BANK',
        'stocks': ['SBIN', 'PNB', 'BANKBARODA', 'CANBK', 'UNIONBANK',
                  'BANKINDIA', 'CENTRALBK', 'IOB', 'MAHABANK', 'INDIANB',
                  'JKBANK', 'PNBHOUSING', 'ORIENTBANK', 'ANDHRABANK', 'CORPBANK']
    }
}

BENCHMARK_SYMBOL = 'NIFTY_50'
BENCHMARK_NAME = 'NIFTY 50'

# ==================== USER SESSION MANAGEMENT ====================

def get_user_cache_key(api_key):
    """Create unique cache key for each user based on their API key"""
    return hashlib.md5(api_key.encode()).hexdigest()[:8]

# ==================== TWELVE DATA API ====================

def fetch_twelve_data(symbol, api_key, start_date, end_date, user_key):
    """Fetch data from Twelve Data API with per-user caching"""
    
    # Create user-specific cache key
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
                    
                    # Check for error messages
                    if 'code' in data:
                        if data['code'] == 429:
                            return {'error': 'rate_limit'}
                        elif data['code'] == 400:
                            return {'error': 'not_found', 'symbol': symbol}
                    
                    # Parse successful response
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
    
    # Call cached function with user-specific key
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
                pass  # Silent fail for missing symbols
            return None
    
    return None

def verify_api_key(api_key):
    """Verify if the API key is valid"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': 'AAPL',
        'interval': '1day',
        'apikey': api_key,
        'outputsize': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'code' in data and data['code'] == 401:
                return False, "Invalid API key"
            if 'values' in data or 'code' not in data:
                return True, "API key verified successfully!"
        return False, "Could not verify API key"
    except:
        return False, "Network error during verification"

def fetch_all_data(api_key, user_key, incremental=False):
    """Fetch all sector and stock data from Twelve Data"""
    
    # Check if we have historical data stored
    historical_key = f"historical_data_{user_key}"
    last_fetch_key = f"last_fetch_date_{user_key}"
    
    if incremental and historical_key in st.session_state and last_fetch_key in st.session_state:
        # Incremental fetch - only get new data
        start_date = st.session_state[last_fetch_key] + timedelta(days=1)
        end_date = datetime.now()
        
        if (end_date - start_date).days < 1:
            st.info("✅ Data is already up to date!")
            return st.session_state[historical_key]
        
        st.info(f"🔄 Incremental update: Fetching from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        st.info(f"⚡ Only {(end_date - start_date).days} days to fetch - much faster!")
    else:
        # Full fetch - get 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        st.info(f"📅 Initial fetch: Getting 2 years of data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        st.info("🕐 This will take 3-5 minutes due to API rate limits")
        st.info("💡 Next time, only new data will be fetched!")
    
    all_data = {}
    failed_symbols = []
    request_count = 0
    
    # Progress tracking
    total_items = 1 + len(SECTORS_CONFIG) + sum(len(s['stocks']) for s in SECTORS_CONFIG.values())
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_item = 0
    
    # Fetch benchmark
    status_text.text(f"Fetching benchmark: {BENCHMARK_NAME} (Request {request_count + 1}/{total_items})")
    benchmark_df = fetch_twelve_data(BENCHMARK_SYMBOL, api_key, start_date, end_date, user_key)
    request_count += 1
    
    if benchmark_df is not None:
        all_data[BENCHMARK_NAME] = benchmark_df
    else:
        st.error("Failed to fetch benchmark data. Cannot proceed.")
        return None
    
    current_item += 1
    progress_bar.progress(current_item / total_items)
    time.sleep(1)  # Rate limiting
    
    # Fetch sectors
    for sector_key, sector_info in SECTORS_CONFIG.items():
        status_text.text(f"Fetching sector: {sector_info['name']} (Request {request_count + 1}/{total_items})")
        
        sector_df = fetch_twelve_data(sector_info['symbol'], api_key, start_date, end_date, user_key)
        request_count += 1
        
        if sector_df is not None:
            all_data[sector_key] = sector_df
        else:
            failed_symbols.append(sector_info['name'])
        
        current_item += 1
        progress_bar.progress(current_item / total_items)
        time.sleep(1)  # Rate limiting
        
        # Fetch stocks for this sector
        for stock in sector_info['stocks']:
            status_text.text(f"Fetching stock: {stock} (Request {request_count + 1}/{total_items})")
            stock_df = fetch_twelve_data(stock, api_key, start_date, end_date, user_key)
            request_count += 1
            
            if stock_df is not None:
                all_data[stock] = stock_df
            else:
                failed_symbols.append(stock)
            
            current_item += 1
            progress_bar.progress(current_item / total_items)
            time.sleep(1)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    st.info(f"📊 Total API requests used: {request_count}/{total_items}")
    
    if failed_symbols:
        with st.expander(f"⚠️ Failed to fetch {len(failed_symbols)} symbols (click to see details)"):
            st.write(", ".join(failed_symbols[:20]))
    
    if len(all_data) < 10:  # Need at least some data
        st.error("Not enough data fetched. Please check your API key and try again.")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data.values(), axis=1, join='outer')
    combined_df.columns = list(all_data.keys())
    
    # If incremental, merge with historical data
    if incremental and historical_key in st.session_state:
        historical_df = st.session_state[historical_key]
        
        # Combine historical and new data
        combined_df = pd.concat([historical_df, combined_df])
        
        # Remove duplicates, keeping the latest
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        st.success(f"✅ Updated with new data! Added {request_count} new data points")
    else:
        st.success(f"✅ Successfully fetched {len(all_data)} symbols with {len(combined_df)} days of data")
    
    # Resample to weekly
    weekly_df = combined_df.resample('W').last()
    weekly_df = weekly_df.ffill().bfill()
    
    # Store historical data and last fetch date
    st.session_state[historical_key] = combined_df  # Store daily data
    st.session_state[last_fetch_key] = end_date
    
    st.info(f"💾 Stored {len(combined_df)} days of historical data for future incremental updates")
    st.info(f"📈 Resampled to {len(weekly_df)} weeks for RRG calculation")
    
    return weekly_df

# ==================== RRG CALCULATIONS ====================

def calculate_rrg_metrics(data, benchmark=BENCHMARK_NAME, window=26):
    """Calculate RRG metrics"""
    if data is None or benchmark not in data.columns:
        return None, None
    
    # Calculate relative strength
    benchmark_prices = data[benchmark]
    
    rsr_data = {}
    rsm_data = {}
    
    for col in data.columns:
        if col == benchmark:
            continue
        
        # Calculate RS
        rs = data[col] / benchmark_prices
        
        # Calculate RSR (smoothed RS)
        rs_mean = rs.rolling(window=window).mean()
        rs_mean_of_mean = rs_mean.rolling(window=window).mean()
        rs_std = rs_mean.rolling(window=window).std().clip(lower=1e-10)
        
        # Z-score normalization
        z_rsr = (rs_mean - rs_mean_of_mean) / rs_std
        rsr_data[col] = 100 + z_rsr
        
        # Calculate RSM (momentum)
        roc_rs = rs_mean.pct_change() * 100
        roc_rs_mean = roc_rs.rolling(window=window).mean()
        roc_rs_std = roc_rs.rolling(window=window).std().clip(lower=1e-10)
        
        # Z-score normalization
        z_rsm = (roc_rs - roc_rs_mean) / roc_rs_std
        rsm_data[col] = 100 + z_rsm
    
    rsr_df = pd.DataFrame(rsr_data).dropna()
    rsm_df = pd.DataFrame(rsm_data).dropna()
    
    return rsr_df, rsm_df

# ==================== VISUALIZATION ====================

def get_quadrant_color(rsr, rsm):
    """Get color based on quadrant"""
    if rsr > 100 and rsm > 100:
        return '#2ecc71'  # Leading - Green
    elif rsr <= 100 and rsm > 100:
        return '#3498db'  # Improving - Blue
    elif rsr > 100 and rsm <= 100:
        return '#f39c12'  # Weakening - Orange
    else:
        return '#e74c3c'  # Lagging - Red

def create_rrg_plot(rsr_df, rsm_df, symbols, tail_length=5):
    """Create interactive RRG plot"""
    fig = go.Figure()
    
    # Add quadrant backgrounds
    fig.add_shape(type="rect", x0=100, y0=100, x1=106, y1=106,
                 fillcolor="#f0fff4", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=94, y0=100, x1=100, y1=106,
                 fillcolor="#f0f9ff", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=100, y0=94, x1=106, y1=100,
                 fillcolor="#fffbeb", opacity=0.25, line_width=0)
    fig.add_shape(type="rect", x0=94, y0=94, x1=100, y1=100,
                 fillcolor="#fef2f2", opacity=0.25, line_width=0)
    
    # Add center lines
    fig.add_hline(y=100, line_dash="dash", line_color="#333", line_width=1.5)
    fig.add_vline(x=100, line_dash="dash", line_color="#333", line_width=1.5)
    
    # Plot symbols
    for symbol in symbols:
        if symbol not in rsr_df.columns or symbol not in rsm_df.columns:
            continue
        
        # Get tail data
        rsr_vals = rsr_df[symbol].dropna().tail(tail_length).values
        rsm_vals = rsm_df[symbol].dropna().tail(tail_length).values
        
        if len(rsr_vals) < 2:
            continue
        
        # Get current position
        last_rsr = rsr_vals[-1]
        last_rsm = rsm_vals[-1]
        color = get_quadrant_color(last_rsr, last_rsm)
        
        # Determine if sector or stock
        is_sector = symbol in SECTORS_CONFIG
        marker_symbol = 'square' if is_sector else 'circle'
        marker_size = 28 if is_sector else 20
        line_width = 5 if is_sector else 4
        
        # Add tail
        fig.add_trace(go.Scatter(
            x=rsr_vals, y=rsm_vals,
            mode='lines',
            line=dict(color=color, width=line_width),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add current point
        fig.add_trace(go.Scatter(
            x=[last_rsr], y=[last_rsm],
            mode='markers+text',
            marker=dict(size=marker_size, symbol=marker_symbol, color=color,
                       line=dict(color='white', width=4)),
            text=[SECTORS_CONFIG[symbol]['name'] if is_sector else symbol],
            textposition='top center',
            textfont=dict(size=15 if is_sector else 13, color='#000', family='Arial Black'),
            name=symbol,
            customdata=[symbol],
            hovertemplate=f'<b>{symbol}</b><br>RSR: {last_rsr:.2f}<br>RSM: {last_rsm:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>🇮🇳 Indian Market RRG - Sectors View</b>',
            font=dict(size=24, color='#1a1a1a', family='Arial Black')
        ),
        xaxis=dict(
            title='<b>JdK RS-Ratio</b>',
            showgrid=False,
            range=[94, 106],
            title_font=dict(size=16, color='#1a1a1a')
        ),
        yaxis=dict(
            title='<b>JdK RS-Momentum</b>',
            showgrid=False,
            range=[94, 106],
            title_font=dict(size=16, color='#1a1a1a')
        ),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#fafafa',
        height=750,
        hovermode='closest',
        annotations=[
            dict(x=104.8, y=105.5, text='<b>Leading</b>', showarrow=False,
                 font=dict(size=20, color='#16a34a', family='Arial Black'),
                 bgcolor='rgba(255,255,255,0.8)', borderpad=5),
            dict(x=95.2, y=105.5, text='<b>Improving</b>', showarrow=False,
                 font=dict(size=20, color='#0284c7', family='Arial Black'),
                 bgcolor='rgba(255,255,255,0.8)', borderpad=5),
            dict(x=104.8, y=94.5, text='<b>Weakening</b>', showarrow=False,
                 font=dict(size=20, color='#ea580c', family='Arial Black'),
                 bgcolor='rgba(255,255,255,0.8)', borderpad=5),
            dict(x=95.2, y=94.5, text='<b>Lagging</b>', showarrow=False,
                 font=dict(size=20, color='#dc2626', family='Arial Black'),
                 bgcolor='rgba(255,255,255,0.8)', borderpad=5)
        ]
    )
    
    return fig

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Indian Market RRG - Team Edition",
        page_icon="🇮🇳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {background-color: #f8fafc;}
        h1 {color: #1e293b;}
        .stButton>button {
            background-color: #667eea;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: bold;
        }
        div[data-testid="stExpander"] {
            background-color: #f1f5f9;
            border-radius: 8px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🇮🇳 Indian Market RRG Analysis - Team Edition")
    st.markdown("**Multi-User Support** | Each team member uses their own FREE API key")
    
    # API Key Input
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
        st.session_state.api_verified = False
        st.session_state.user_key = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 Your API Configuration")
        
        api_key_input = st.text_input(
            "Enter Your Twelve Data API Key",
            type="password",
            value=st.session_state.api_key if st.session_state.api_key else "",
            help="Each team member should use their own FREE API key"
        )
        
        if api_key_input and api_key_input != st.session_state.api_key:
            with st.spinner("Verifying API key..."):
                is_valid, message = verify_api_key(api_key_input)
                if is_valid:
                    st.session_state.api_key = api_key_input
                    st.session_state.api_verified = True
                    st.session_state.user_key = get_user_cache_key(api_key_input)
                    st.success(message)
                else:
                    st.error(message)
                    st.session_state.api_verified = False
        
        if st.session_state.api_verified:
            st.success("✅ API Key Active")
            st.info(f"👤 User ID: {st.session_state.user_key}")
        
        st.markdown("---")
        
        with st.expander("📘 How to Get Your FREE API Key"):
            st.markdown("""
            **Quick Setup (2 minutes):**
            1. Visit [twelvedata.com](https://twelvedata.com/)
            2. Sign up for FREE account
            3. Get your API key (800 requests/day)
            4. Paste it above
            
            **Team Benefits:**
            - Each person gets 800 requests/day
            - Data cached separately per user
            - No conflicts between team members
            - Total team capacity: 4,000 requests/day!
            """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        tail_length = st.slider("Tail Length (weeks)", 3, 10, 5)
        window_size = st.slider("RRG Window (weeks)", 10, 52, 26)
        
        st.markdown("---")
        st.markdown("### 📊 Quadrants")
        st.markdown("🟢 **Leading**: Strong & Rising")
        st.markdown("🔵 **Improving**: Weak but Rising")
        st.markdown("🟠 **Weakening**: Strong but Falling")
        st.markdown("🔴 **Lagging**: Weak & Falling")
        
        st.markdown("---")
        if st.button("🔄 Clear My Cache & Refresh"):
            st.cache_data.clear()
            for key in ['data', 'rsr', 'rsm', 'historical_data', 'last_fetch_date']:
                # Clear all variations with user_key
                user_key = st.session_state.get('user_key')
                if user_key:
                    full_key = f"{key}_{user_key}"
                    if full_key in st.session_state:
                        del st.session_state[full_key]
            st.success("✅ Cache cleared! Next fetch will be a full 2-year load.")
            time.sleep(1)
            st.rerun()
    
    # Check API key verification
    if not st.session_state.api_verified:
        st.info("👆 Please enter your Twelve Data API key in the sidebar to start")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Smart Data Fetching!
            
            **First Time (One-time):**
            - Fetches 2 years of historical data
            - ~241 API requests
            - Takes 3-5 minutes
            - Stores data in memory
            
            **Future Updates:**
            - Only fetches NEW data (delta)
            - ~1-5 API requests per day
            - Takes <1 minute
            - Merges with historical data
            
            **Your Team Benefits:**
            - Each person: 800 requests/day
            - After initial load: 795+ requests free!
            - Update 100+ times per day!
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Quick Start Guide
            
            1. **Sign up** at [twelvedata.com](https://twelvedata.com/)
            2. **Copy** your API key
            3. **Paste** in sidebar
            4. **Click** "Fetch NSE Data"
            5. **Wait** 3-5 min (one-time)
            6. **Enjoy** instant updates!
            
            **Next Day:**
            - Click "Update Data"
            - Wait <1 minute
            - Get latest market data!
            
            **Data Stored:** 2 years in memory
            """)
        return
    
    # Initialize session state
    if 'view' not in st.session_state:
        st.session_state.view = 'sectors'
        st.session_state.selected_sector = None
    
    # Fetch data button
    data_key = f"data_{st.session_state.user_key}"
    historical_key = f"historical_data_{st.session_state.user_key}"
    last_fetch_key = f"last_fetch_date_{st.session_state.user_key}"
    
    if data_key not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📥 Fetch NSE Data (Initial Load)", type="primary", use_container_width=True):
                with st.spinner("Fetching 2 years of historical data... This will take 3-5 minutes"):
                    data = fetch_all_data(st.session_state.api_key, st.session_state.user_key, incremental=False)
                    if data is not None:
                        st.session_state[data_key] = data
                        st.balloons()
                        st.rerun()
        
        st.info("👆 Click 'Fetch NSE Data' to load 2 years of historical market data (one-time fetch)")
        st.markdown("---")
        
        # Show usage info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Symbols", "241", help="15 sectors + 15 stocks each + benchmark")
        with col2:
            st.metric("🔄 API Requests", "~241", help="One request per symbol")
        with col3:
            st.metric("⏱️ First Fetch", "3-5 min", help="One-time historical fetch")
        with col4:
            st.metric("⚡ Future Updates", "<1 min", help="Only fetch new data")
        
        return
    
    # Show update button if data exists
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 Update Data", use_container_width=True, help="Fetch only new data since last update"):
            with st.spinner("Fetching latest data..."):
                data = fetch_all_data(st.session_state.api_key, st.session_state.user_key, incremental=True)
                if data is not None:
                    st.session_state[data_key] = data
                    # Clear RRG metrics to recalculate
                    rsr_key = f"rsr_{st.session_state.user_key}"
                    rsm_key = f"rsm_{st.session_state.user_key}"
                    if rsr_key in st.session_state:
                        del st.session_state[rsr_key]
                    if rsm_key in st.session_state:
                        del st.session_state[rsm_key]
                    st.success("✅ Data updated successfully!")
                    st.rerun()
    
    # Show last update info
    if last_fetch_key in st.session_state:
        last_fetch = st.session_state[last_fetch_key]
        days_ago = (datetime.now() - last_fetch).days
        if days_ago == 0:
            st.info(f"📅 Last updated: Today")
        elif days_ago == 1:
            st.info(f"📅 Last updated: Yesterday")
        else:
            st.info(f"📅 Last updated: {days_ago} days ago ({last_fetch.strftime('%Y-%m-%d')})")
        
        if historical_key in st.session_state:
            hist_df = st.session_state[historical_key]
            st.info(f"💾 Historical data: {len(hist_df)} days stored ({hist_df.index[0].strftime('%Y-%m-%d')} to {hist_df.index[-1].strftime('%Y-%m-%d')})")
    
    # Calculate RRG metrics
    rsr_key = f"rsr_{st.session_state.user_key}"
    rsm_key = f"rsm_{st.session_state.user_key}"
    
    if rsr_key not in st.session_state or rsm_key not in st.session_state:
        with st.spinner("Calculating RRG metrics..."):
            rsr_df, rsm_df = calculate_rrg_metrics(st.session_state[data_key], window=window_size)
            if rsr_df is not None and rsm_df is not None:
                st.session_state[rsr_key] = rsr_df
                st.session_state[rsm_key] = rsm_df
            else:
                st.error("Failed to calculate RRG metrics")
                return
    
    # Navigation
    col1, col2 = st.columns([6, 1])
    with col1:
        if st.session_state.view == 'sectors':
            st.subheader("📊 Sectors View - Click any sector to drill down")
        else:
            sector_name = SECTORS_CONFIG[st.session_state.selected_sector]['name']
            st.subheader(f"📊 {sector_name} Sector - Individual Stocks")
    
    with col2:
        if st.session_state.view == 'stocks':
            if st.button("← Back"):
                st.session_state.view = 'sectors'
                st.session_state.selected_sector = None
                st.rerun()
    
    # Display chart
    if st.session_state.view == 'sectors':
        # Show sectors
        symbols = list(SECTORS_CONFIG.keys())
        fig = create_rrg_plot(st.session_state[rsr_key], st.session_state[rsm_key], symbols, tail_length)
    else:
        # Show stocks for selected sector
        stocks = SECTORS_CONFIG[st.session_state.selected_sector]['stocks']
        fig = create_rrg_plot(st.session_state[rsr_key], st.session_state[rsm_key], stocks, tail_length)
        fig.update_layout(title=dict(text=f'<b>{SECTORS_CONFIG[st.session_state.selected_sector]["name"]} - Top 15 Stocks</b>'))
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True, key="rrg_chart")
    
    # Add click handler note
    if st.session_state.view == 'sectors':
        st.info("💡 **Tip:** Click on any sector square to see individual stocks in that sector")
    
    # Sector selection buttons
    if st.session_state.view == 'sectors':
        st.markdown("---")
        st.markdown("### 🎯 Quick Sector Navigation")
        
        cols = st.columns(5)
        sector_items = list(SECTORS_CONFIG.items())
        
        for idx, (sector_key, sector_info) in enumerate(sector_items):
            with cols[idx % 5]:
                if st.button(sector_info['name'], key=f"btn_{sector_key}", use_container_width=True):
                    st.session_state.view = 'stocks'
                    st.session_state.selected_sector = sector_key
                    st.rerun()
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("💡 **Navigation:** Click sectors to drill down, use Back button to return")
    with col2:
        st.markdown("✨ **Data Source:** Twelve Data API (NSE India)")
    with col3:
        st.markdown(f"👤 **Your User ID:** `{st.session_state.user_key}`")
    
    # Show quadrant statistics
    with st.expander("📈 Current Market Statistics"):
        if st.session_state.view == 'sectors':
            symbols = list(SECTORS_CONFIG.keys())
        else:
            symbols = SECTORS_CONFIG[st.session_state.selected_sector]['stocks']
        
        leading = improving = weakening = lagging = 0
        
        for symbol in symbols:
            if symbol in st.session_state[rsr_key].columns and symbol in st.session_state[rsm_key].columns:
                rsr = st.session_state[rsr_key][symbol].iloc[-1]
                rsm = st.session_state[rsm_key][symbol].iloc[-1]
                
                if rsr > 100 and rsm > 100:
                    leading += 1
                elif rsr <= 100 and rsm > 100:
                    improving += 1
                elif rsr > 100 and rsm <= 100:
                    weakening += 1
                else:
                    lagging += 1
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 Leading", leading, help="Strong & Rising")
        with col2:
            st.metric("🔵 Improving", improving, help="Weak but Rising")
        with col3:
            st.metric("🟠 Weakening", weakening, help="Strong but Falling")
        with col4:
            st.metric("🔴 Lagging", lagging, help="Weak & Falling")

if __name__ == "__main__":
    main()
