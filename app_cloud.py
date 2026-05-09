"""
Streamlit Dashboard for Cloud Deployment
Consumes FastAPI endpoints from Azure App Service
"""
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Equity Factor Model Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_predictions(limit=1000):
    """Fetch predictions from API"""
    try:
        response = requests.get(f"{API_URL}/predictions?limit={limit}", timeout=30)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_top_stocks(n=20):
    """Fetch top stocks from API"""
    try:
        response = requests.get(f"{API_URL}/top-stocks?n={n}", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching top stocks: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_performance():
    """Fetch performance metrics from API"""
    try:
        response = requests.get(f"{API_URL}/performance", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching performance: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_features():
    """Fetch feature importance from API"""
    try:
        response = requests.get(f"{API_URL}/features", timeout=30)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data['features'])
    except Exception as e:
        st.error(f"Error fetching features: {e}")
        return None

def main():
    st.title("📈 Equity Factor Model Dashboard")
    st.markdown("**Multi-Factor Quantitative Strategy with Machine Learning**")
    st.markdown(f"*Data source: {API_URL}*")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        refresh = st.button("🔄 Refresh Data")
        if refresh:
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard displays predictions from a quantitative equity factor model.
        
        **Features:**
        - 239 stocks analyzed
        - 19 fundamental & technical factors
        - Ensemble ML model
        - Monthly updates
        """)
    
    # Fetch data
    predictions = fetch_predictions()
    performance = fetch_performance()
    
    if predictions is None:
        st.error("⚠️ Unable to load data from API. Please check the connection.")
        st.info(f"API URL: {API_URL}")
        st.info("Make sure the API is running and accessible.")
        return
    
    # Tabs
    tabs = st.tabs([
        "📊 Overview",
        "🎯 Stock Analysis",
        "🔍 Feature Importance",
        "📈 Performance"
    ])
    
    with tabs[0]:
        show_overview(predictions, performance)
    
    with tabs[1]:
        show_stock_analysis(predictions)
    
    with tabs[2]:
        show_features()
    
    with tabs[3]:
        show_performance_tab(performance)

def show_overview(predictions, performance):
    """Overview tab"""
    st.header("Model Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", len(predictions['ticker'].unique()))
    
    with col2:
        latest_date = predictions['date'].max()
        st.metric("Latest Month", latest_date.strftime('%Y-%m'))
    
    with col3:
        if performance:
            st.metric("Mean IC", f"{performance['mean_ic']:.4f}")
        else:
            st.metric("Mean IC", "N/A")
    
    with col4:
        if performance:
            st.metric("Sharpe Ratio", f"{performance['sharpe']:.3f}")
        else:
            st.metric("Sharpe Ratio", "N/A")
    
    st.markdown("---")
    
    # Latest predictions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction Distribution")
        latest = predictions[predictions['date'] == predictions['date'].max()]
        
        fig = px.histogram(
            latest,
            x='prediction',
            nbins=50,
            title=f"Latest Predictions ({latest['date'].iloc[0].strftime('%Y-%m')})"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 10 Stocks")
        top_stocks = fetch_top_stocks(10)
        if top_stocks:
            df = pd.DataFrame(top_stocks['stocks'])
            df['prediction'] = df['prediction'].apply(lambda x: f"{x:.4f}")
            st.dataframe(df[['ticker', 'sector', 'prediction']], hide_index=True, use_container_width=True)
    
    # Sector distribution
    st.subheader("📊 Sector Distribution")
    latest = predictions[predictions['date'] == predictions['date'].max()]
    sector_stats = latest.groupby('sector').agg({
        'prediction': ['count', 'mean']
    }).reset_index()
    sector_stats.columns = ['Sector', 'Count', 'Avg Prediction']
    
    fig = px.bar(
        sector_stats.sort_values('Avg Prediction', ascending=False),
        x='Avg Prediction',
        y='Sector',
        orientation='h',
        title="Average Prediction by Sector",
        color='Avg Prediction',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_stock_analysis(predictions):
    """Stock analysis tab"""
    st.header("Stock Analysis")
    
    tickers = sorted(predictions['ticker'].unique())
    
    selected_tickers = st.multiselect(
        "Select Stock(s) to Analyze",
        options=tickers,
        default=tickers[:3] if len(tickers) >= 3 else tickers
    )
    
    if not selected_tickers:
        st.warning("Please select at least one stock")
        return
    
    for ticker in selected_tickers:
        with st.expander(f"📊 {ticker}", expanded=len(selected_tickers) == 1):
            ticker_data = predictions[predictions['ticker'] == ticker].sort_values('date')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ticker_data['date'],
                    y=ticker_data['prediction'],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{ticker} - Prediction Time Series",
                    xaxis_title="Date",
                    yaxis_title="Predicted Return",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Latest Prediction**")
                latest = ticker_data.iloc[-1]
                st.metric("Prediction", f"{latest['prediction']:.4f}")
                st.metric("Sector", latest['sector'])
                st.metric("Date", latest['date'].strftime('%Y-%m'))

def show_features():
    """Feature importance tab"""
    st.header("Feature Importance")
    
    features = fetch_features()
    
    if features is None:
        st.warning("Feature importance data not available")
        return
    
    st.subheader("Top 15 Features by Mean IC")
    
    top_features = features.nlargest(15, 'mean_ic')
    
    fig = px.bar(
        top_features.sort_values('mean_ic'),
        x='mean_ic',
        y='feature',
        orientation='h',
        title="Feature Importance",
        color='mean_ic',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("All Features")
    st.dataframe(features.sort_values('mean_ic', ascending=False), hide_index=True, use_container_width=True)

def show_performance_tab(performance):
    """Performance tab"""
    st.header("Model Performance")
    
    if performance is None:
        st.warning("Performance metrics not available")
        return
    
    st.subheader("Out-of-Time Validation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean IC", f"{performance['mean_ic']:.4f}")
    with col2:
        st.metric("IC-IR", f"{performance['ic_ir']:.4f}")
    with col3:
        st.metric("Sharpe Ratio", f"{performance['sharpe']:.3f}")
    with col4:
        st.metric("Win Rate", f"{performance['win_rate']*100:.1f}%")
    
    st.markdown("---")
    
    st.info("""
    **Metrics Explanation:**
    - **Mean IC**: Information Coefficient - correlation between predictions and actual returns
    - **IC-IR**: IC Information Ratio - consistency of IC over time
    - **Sharpe Ratio**: Risk-adjusted returns
    - **Win Rate**: Percentage of months with positive IC
    """)

if __name__ == "__main__":
    main()
