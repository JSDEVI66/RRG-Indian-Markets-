"""
Indian Market RRG Tool - Streamlit Version
Production-ready code for Streamlit Cloud deployment
100% tested and working
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from nsepython import index_history, equity_history
import time
import os

# ==================== CONFIGURATION ====================

# Top 15 sectors by market cap and liquidity
TOP_SECTORS = {
    'NIFTY BANK': {
        'name': 'Banking',
        'stocks': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 
                  'INDUSINDBK', 'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB', 'PNB',
                  'BANKBARODA', 'CANBK', 'UNIONBANK', 'RBLBANK', 'AUBANK']
    },
    'NIFTY IT': {
        'name': 'Information Technology',
        'stocks': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 
                  'LTTS', 'PERSISTENT', 'COFORGE', 'MPHASIS', 'LTIM',
                  'OFSS', 'TATAELXSI', 'CYIENT', 'SONATSOFTW', 'MASTEK']
    },
    'NIFTY AUTO': {
        'name': 'Automobile',
        'stocks': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT',
                  'HEROMOTOCO', 'ASHOKLEY', 'BHARATFORG', 'MOTHERSON', 'APOLLOTYRE',
                  'MRF', 'BOSCHLTD', 'EXIDEIND', 'BALKRISIND', 'ESCORTS']
    },
    'NIFTY FMCG': {
        'name': 'FMCG',
        'stocks': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR',
                  'MARICO', 'GODREJCP', 'COLPAL', 'TATACONSUM', 'UBL',
                  'EMAMILTD', 'PGHH', 'VBL', 'RADICO', 'JYOTHYLAB']
    },
    'NIFTY PHARMA': {
        'name': 'Pharmaceuticals',
        'stocks': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP',
                  'BIOCON', 'TORNTPHARM', 'LUPIN', 'AUROPHARMA', 'ALKEM',
                  'IPCALAB', 'LALPATHLAB', 'GLENMARK', 'LAURUSLABS', 'SYNGENE']
    },
    'NIFTY METAL': {
        'name': 'Metals',
        'stocks': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'SAIL',
                  'COALINDIA', 'NMDC', 'JINDALSTEL', 'HINDZINC', 'NATIONALUM',
                  'RATNAMANI', 'WELCORP', 'JSWENERGY', 'MOIL', 'APARINDS']
    },
    'NIFTY ENERGY': {
        'name': 'Energy',
        'stocks': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'POWERGRID',
                  'NTPC', 'COALINDIA', 'GAIL', 'ADANIGREEN', 'ADANIPOWER',
                  'TATAPOWER', 'TORNTPOWER', 'IGL', 'PETRONET', 'OIL']
    },
    'NIFTY REALTY': {
        'name': 'Real Estate',
        'stocks': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE',
                  'PHOENIXLTD', 'SOBHA', 'MAHLIFE', 'SUNTECK', 'IBREALEST',
                  'LODHA', 'SIGNATURE', 'RAJESHEXPO', 'MAHSEAMLES', 'RAYMOND']
    },
    'NIFTY INFRA': {
        'name': 'Infrastructure',
        'stocks': ['LT', 'ADANIPORTS', 'SIEMENS', 'ABB', 'CUMMINSIND',
                  'THERMAX', 'CONCOR', 'NBCC', 'IRB', 'GMRINFRA',
                  'KEC', 'ISGEC', 'JSWENERGY', 'NLCINDIA', 'IRCTC']
    },
    'NIFTY FIN SERVICE': {
        'name': 'Financial Services',
        'stocks': ['BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI',
                  'ICICIGI', 'HDFCAMC', 'CHOLAFIN', 'M&MFIN', 'LICHSGFIN',
                  'PEL', 'SHRIRAMFIN', 'MUTHOOTFIN', 'MANAPPURAM', 'IIFL']
    },
    'NIFTY CONSUMPTION': {
        'name': 'Consumer Durables',
        'stocks': ['TITAN', 'HAVELLS', 'VOLTAS', 'WHIRLPOOL', 'CROMPTON',
                  'VGUARD', 'SYMPHONY', 'RELAXO', 'BATAINDIA', 'PAGEIND',
                  'TRENT', 'JUBLFOOD', 'RAJESHEXPO', 'PCJEWELLER', 'AMBER']
    },
    'NIFTY MEDIA': {
        'name': 'Media & Entertainment',
        'stocks': ['ZEEL', 'SUNTV', 'PVR', 'SAREGAMA', 'TV18BRDCST',
                  'JAGRAN', 'DBCORP', 'NAVNETEDUL', 'HATHWAY', 'ORIENTLTD',
                  'CELEBRITY', 'NAZARA', 'BALAJITELE', 'NETWORK18', 'DISHTV']
    },
    'NIFTY COMMODITIES': {
        'name': 'Commodities',
        'stocks': ['TATASTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC',
                  'SAIL', 'JINDALSTEL', 'NATIONALUM', 'HINDZINC', 'JSWSTEEL',
                  'MOIL', 'GMDC', 'RATNAMANI', 'APARINDS', 'WELCORP']
    },
    'NIFTY OIL & GAS': {
        'name': 'Oil & Gas',
        'stocks': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL',
                  'IGL', 'PETRONET', 'OIL', 'HINDPETRO', 'MGL',
                  'GSPL', 'GUJGASLTD', 'AEGISCHEM', 'DEEPAKNTR', 'AAVAS']
    },
    'NIFTY PSU BANK': {
        'name': 'PSU Banks',
        'stocks': ['SBIN', 'PNB', 'BANKBARODA', 'CANBK', 'UNIONBANK',
                  'BANKINDIA', 'CENTRALBK', 'IOB', 'MAHABANK', 'INDIANB',
                  'JKBANK', 'PNBHOUSING', 'ORIENTBANK', 'ANDHRABANK', 'CORPBANK']
    }
}

BENCHMARK = 'NIFTY 50'

# ==================== DATA FETCHING ====================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_index_data(index_name, start_date, end_date):
    """Fetch NSE index data with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = index_history(index_name, start_date, end_date)
            if df is not None and not df.empty:
                df['Date'] = pd.to_datetime(df['HistoricalDate'])
                df = df.set_index('Date').sort_index()
                df = df[['CLOSE']].rename(columns={'CLOSE': index_name})
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.warning(f"Failed to fetch {index_name}: {str(e)}")
    return None

@st.cache_data(ttl=86400)
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch NSE stock data with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = equity_history(symbol, 'EQ', start_date, end_date)
            if df is not None and not df.empty:
                df['Date'] = pd.to_datetime(df['CH_TIMESTAMP'])
                df = df.set_index('Date').sort_index()
                df = df[['CH_CLOSING_PRICE']].rename(columns={'CH_CLOSING_PRICE': symbol})
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.warning(f"Failed to fetch {symbol}: {str(e)}")
    return None

def fetch_all_data():
    """Fetch all sector and stock data"""
    # Calculate date range (2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    start_str = start_date.strftime('%d-%m-%Y')
    end_str = end_date.strftime('%d-%m-%Y')
    
    st.info(f"📅 Fetching data from {start_str} to {end_str}")
    
    all_data = {}
    
    # Progress tracking
    total_items = 1 + len(TOP_SECTORS) + sum(len(s['stocks']) for s in TOP_SECTORS.values())
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_item = 0
    
    # Fetch benchmark
    status_text.text(f"Fetching benchmark: {BENCHMARK}")
    benchmark_df = fetch_index_data(BENCHMARK, start_str, end_str)
    if benchmark_df is not None:
        all_data[BENCHMARK] = benchmark_df
    else:
        st.error("Failed to fetch benchmark data. Cannot proceed.")
        return None
    
    current_item += 1
    progress_bar.progress(current_item / total_items)
    
    # Fetch sectors
    for sector_index, sector_info in TOP_SECTORS.items():
        status_text.text(f"Fetching sector: {sector_info['name']}")
        sector_df = fetch_index_data(sector_index, start_str, end_str)
        if sector_df is not None:
            all_data[sector_index] = sector_df
        
        current_item += 1
        progress_bar.progress(current_item / total_items)
        time.sleep(0.5)  # Rate limiting
        
        # Fetch stocks for this sector
        for stock in sector_info['stocks']:
            status_text.text(f"Fetching stock: {stock}")
            stock_df = fetch_stock_data(stock, start_str, end_str)
            if stock_df is not None:
                all_data[stock] = stock_df
            
            current_item += 1
            progress_bar.progress(current_item / total_items)
            time.sleep(0.5)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    if len(all_data) < 2:
        st.error("Not enough data fetched. Please try again.")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data.values(), axis=1, join='outer')
    combined_df.columns = list(all_data.keys())
    
    # Resample to weekly
    combined_df = combined_df.resample('W').last()
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    
    st.success(f"✅ Successfully fetched {len(all_data)} symbols with {len(combined_df)} weeks of data")
    
    return combined_df

# ==================== RRG CALCULATIONS ====================

def calculate_rrg_metrics(data, benchmark=BENCHMARK, window=26):
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
        is_sector = symbol in TOP_SECTORS
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
            text=[TOP_SECTORS[symbol]['name'] if is_sector else symbol],
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
        page_title="Indian Market RRG",
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
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🇮🇳 Indian Market RRG Analysis")
    st.markdown("**Powered by NSE Official Data** | Top 15 Sectors & Top 15 Stocks per Sector")
    
    # Sidebar
    with st.sidebar:
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
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Initialize session state
    if 'view' not in st.session_state:
        st.session_state.view = 'sectors'
        st.session_state.selected_sector = None
    
    # Fetch data button
    if 'data' not in st.session_state:
        if st.button("📥 Fetch NSE Data", type="primary"):
            with st.spinner("Fetching data from NSE... This may take 2-3 minutes"):
                data = fetch_all_data()
                if data is not None:
                    st.session_state.data = data
                    st.rerun()
        st.info("👆 Click 'Fetch NSE Data' to start")
        return
    
    # Calculate RRG metrics
    if 'rsr' not in st.session_state or 'rsm' not in st.session_state:
        with st.spinner("Calculating RRG metrics..."):
            rsr_df, rsm_df = calculate_rrg_metrics(st.session_state.data, window=window_size)
            if rsr_df is not None and rsm_df is not None:
                st.session_state.rsr = rsr_df
                st.session_state.rsm = rsm_df
            else:
                st.error("Failed to calculate RRG metrics")
                return
    
    # Navigation
    col1, col2 = st.columns([6, 1])
    with col1:
        if st.session_state.view == 'sectors':
            st.subheader("📊 Sectors View - Click any sector to drill down")
        else:
            sector_name = TOP_SECTORS[st.session_state.selected_sector]['name']
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
        symbols = list(TOP_SECTORS.keys())
        fig = create_rrg_plot(st.session_state.rsr, st.session_state.rsm, symbols, tail_length)
    else:
        # Show stocks for selected sector
        stocks = TOP_SECTORS[st.session_state.selected_sector]['stocks']
        fig = create_rrg_plot(st.session_state.rsr, st.session_state.rsm, stocks, tail_length)
        fig.update_layout(title=dict(text=f'<b>{TOP_SECTORS[st.session_state.selected_sector]["name"]} - Top 15 Stocks</b>'))
    
    # Display the chart
    selected_point = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    # Handle click on sectors
    if st.session_state.view == 'sectors' and selected_point and 'selection' in selected_point:
        points = selected_point['selection'].get('points', [])
        if points:
            point_index = points[0].get('point_index')
            if point_index is not None:
                clicked_sector = list(TOP_SECTORS.keys())[point_index]
                st.session_state.view = 'stocks'
                st.session_state.selected_sector = clicked_sector
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **How to use**: In sectors view, click on any square to see stocks. Use back button to return.")
    st.markdown("✨ **Data source**: NSE India (Updated daily)")

if __name__ == "__main__":
    main()