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
import numpy as np

st.set_page_config(
    page_title="Equity Factor Model Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #48bb78;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffaf0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ed8936;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
    # Header with custom styling
    st.markdown('<h1 class="main-header">📈 Equity Factor Model Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Factor Quantitative Strategy with Machine Learning</p>', unsafe_allow_html=True)
    
    # Add explanation banner
    st.markdown("""
    <div class="info-box">
    <b>💡 What This Dashboard Shows:</b> The model predicts <b>next month's stock returns</b> using 19 factors. 
    Higher prediction = higher expected return. Predictions are ranked to select top stocks for portfolio.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stocks.png", width=80)
        st.markdown("### ⚙️ Settings")
        
        refresh = st.button("🔄 Refresh Data", use_container_width=True)
        if refresh:
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        <div class="info-box">
        <b>Universe:</b> 239 stocks<br>
        <b>Features:</b> 19 factors<br>
        <b>Model:</b> Ridge + LightGBM<br>
        <b>Rebalance:</b> Monthly<br>
        <b>Prediction:</b> Next month return<br>
        <b>API:</b> <code>{API_URL.split('//')[1][:20]}...</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ❓ What is Prediction?")
        st.markdown("""
        <div class="success-box">
        <b>Prediction</b> = Expected return for next month<br><br>
        • <b>Positive</b> = Stock expected to go up<br>
        • <b>Negative</b> = Stock expected to go down<br>
        • <b>Higher value</b> = Better investment<br><br>
        Example: 0.05 = +5% expected return
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📚 Quick Guide")
        with st.expander("📊 Overview Tab"):
            st.markdown("View key metrics, top stocks, and sector distribution")
        with st.expander("🎯 Stock Analysis Tab"):
            st.markdown("Analyze individual stock predictions over time")
        with st.expander("🔍 Feature Importance Tab"):
            st.markdown("See which factors drive predictions")
        with st.expander("📈 Performance Tab"):
            st.markdown("Out-of-time validation results")
        
        st.markdown("---")
        st.markdown("##### 💡 Tip")
        st.info("Click 🔄 Refresh to update data from the API")
    
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
    st.markdown("## 📊 Model Overview")
    
    # Add comprehensive explanation
    with st.expander("❓ What Do These Predictions Mean?", expanded=False):
        st.markdown("""
        ### 🎯 Understanding Predictions
        
        **What is a "Prediction"?**
        - The model predicts **next month's stock return** (gain or loss)
        - Example: Prediction of **0.05** means the model expects the stock to gain **+5%** next month
        - Example: Prediction of **-0.02** means the model expects the stock to lose **-2%** next month
        
        **How Are Predictions Used?**
        1. **Rank all 239 stocks** by their predictions
        2. **Select top stocks** (highest predictions) for portfolio
        3. **Rebalance monthly** based on new predictions
        
        **What Makes a Good Prediction?**
        - **Positive value** = Expected to go up (good for buying)
        - **Higher value** = Better expected return
        - **Top 20 stocks** = Best investment opportunities
        
        **Example Portfolio Strategy:**
        - Buy the **top 20 stocks** with highest predictions
        - Hold for **one month**
        - Rebalance based on **new predictions**
        
        **Important Notes:**
        - Predictions are **probabilities**, not guarantees
        - Past performance doesn't guarantee future results
        - Model is validated on **out-of-time data** to ensure reliability
        """)
    
    st.markdown("---")
    
    # Key Metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = len(predictions['ticker'].unique())
        st.metric(
            label="📈 Total Stocks",
            value=total_stocks,
            delta=f"{total_stocks} analyzed",
            help="Number of stocks in the investment universe"
        )
    
    with col2:
        latest_date = predictions['date'].max()
        st.metric(
            label="📅 Latest Month",
            value=latest_date.strftime('%b %Y'),
            delta=latest_date.strftime('%Y-%m-%d'),
            help="Most recent prediction date"
        )
    
    with col3:
        if performance:
            ic_value = performance['mean_ic']
            ic_quality = "Strong" if ic_value > 0.03 else "Good" if ic_value > 0.01 else "Moderate"
            st.metric(
                label="🎯 Mean IC",
                value=f"{ic_value:.4f}",
                delta=ic_quality,
                help="Information Coefficient - correlation between predictions and returns"
            )
        else:
            st.metric("🎯 Mean IC", "N/A")
    
    with col4:
        if performance:
            sharpe = performance['sharpe']
            sharpe_quality = "Excellent" if sharpe > 1.5 else "Good" if sharpe > 1.0 else "Fair"
            st.metric(
                label="📊 Sharpe Ratio",
                value=f"{sharpe:.2f}",
                delta=sharpe_quality,
                help="Risk-adjusted returns (>1.0 is good)"
            )
        else:
            st.metric("📊 Sharpe Ratio", "N/A")
    
    st.markdown("---")
    
    # Latest predictions with enhanced visuals
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Prediction Distribution")
        latest = predictions[predictions['date'] == predictions['date'].max()]
        
        fig = px.histogram(
            latest,
            x='prediction',
            nbins=50,
            title=f"Latest Predictions - {latest['date'].iloc[0].strftime('%B %Y')}",
            labels={'prediction': 'Predicted Return', 'count': 'Number of Stocks'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Return")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        positive_pct = (latest['prediction'] > 0).sum() / len(latest) * 100
        st.markdown(f"""
        <div class="info-box">
        <b>📈 {positive_pct:.1f}%</b> of stocks have positive predicted returns<br>
        <b>Mean Prediction:</b> {latest['prediction'].mean():.4f}<br>
        <b>Std Dev:</b> {latest['prediction'].std():.4f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏆 Top 10 Stocks")
        top_stocks = fetch_top_stocks(10)
        if top_stocks:
            df = pd.DataFrame(top_stocks['stocks'])
            df['prediction'] = df['prediction'].apply(lambda x: f"{x:.4f}")
            df['rank'] = range(1, len(df) + 1)
            
            # Style the dataframe
            st.dataframe(
                df[['rank', 'ticker', 'sector', 'prediction']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", help="Stock ranking"),
                    "ticker": st.column_config.TextColumn("Ticker", help="Stock symbol"),
                    "sector": st.column_config.TextColumn("Sector", help="Industry sector"),
                    "prediction": st.column_config.TextColumn("Prediction", help="Predicted return")
                }
            )
            
            st.markdown("""
            <div class="success-box">
            💡 <b>Tip:</b> These are the highest predicted returns for next month
            </div>
            """, unsafe_allow_html=True)
    
    # Sector distribution with enhanced visuals
    st.markdown("---")
    st.markdown("### 🏭 Sector Analysis")
    
    latest = predictions[predictions['date'] == predictions['date'].max()]
    sector_stats = latest.groupby('sector').agg({
        'prediction': ['count', 'mean', 'std']
    }).reset_index()
    sector_stats.columns = ['Sector', 'Count', 'Avg Prediction', 'Std Dev']
    sector_stats = sector_stats.sort_values('Avg Prediction', ascending=False)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = px.bar(
            sector_stats,
            x='Avg Prediction',
            y='Sector',
            orientation='h',
            title="Average Prediction by Sector",
            color='Avg Prediction',
            color_continuous_scale='RdYlGn',
            text='Avg Prediction',
            labels={'Avg Prediction': 'Average Predicted Return'}
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16,
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=False),
            coloraxis_showscale=False
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Sector Summary")
        sector_stats['Avg Prediction'] = sector_stats['Avg Prediction'].apply(lambda x: f"{x:.4f}")
        sector_stats['Std Dev'] = sector_stats['Std Dev'].apply(lambda x: f"{x:.4f}")
        st.dataframe(
            sector_stats,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Count": st.column_config.NumberColumn("Stocks", width="small"),
                "Avg Prediction": st.column_config.TextColumn("Avg Return", width="small"),
                "Std Dev": st.column_config.TextColumn("Volatility", width="small")
            }
        )
        
        best_sector = sector_stats.iloc[0]['Sector']
        st.markdown(f"""
        <div class="success-box">
        🎯 <b>Best Sector:</b> {best_sector}
        </div>
        """, unsafe_allow_html=True)

def show_stock_analysis(predictions):
    """Stock analysis tab"""
    st.markdown("## 🎯 Stock Analysis")
    
    st.markdown("""
    <div class="info-box">
    📌 <b>How to use:</b> Select stocks to view their <b>predicted returns over time</b>. 
    Prediction shows expected monthly return (e.g., 0.03 = +3% expected gain next month).
    </div>
    """, unsafe_allow_html=True)
    
    tickers = sorted(predictions['ticker'].unique())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_tickers = st.multiselect(
            "🔍 Select Stock(s) to Analyze",
            options=tickers,
            default=tickers[:3] if len(tickers) >= 3 else tickers,
            help="Choose stocks to analyze their prediction trends"
        )
    with col2:
        view_mode = st.radio("View Mode", ["Expanded", "Compact"], horizontal=True)
    
    if not selected_tickers:
        st.warning("⚠️ Please select at least one stock to begin analysis")
        return
    
    for ticker in selected_tickers:
        with st.expander(f"📊 {ticker}", expanded=(len(selected_tickers) == 1 or view_mode == "Expanded")):
            ticker_data = predictions[predictions['ticker'] == ticker].sort_values('date')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Enhanced time series chart
                fig = go.Figure()
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=ticker_data['date'],
                    y=ticker_data['prediction'],
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Prediction:</b> %{y:.4f}<extra></extra>'
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                # Add trend line (only if enough data points)
                if len(ticker_data) >= 3:
                    try:
                        z = np.polyfit(range(len(ticker_data)), ticker_data['prediction'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=ticker_data['date'],
                            y=p(range(len(ticker_data))),
                            mode='lines',
                            name='Trend',
                            line=dict(color='orange', width=2, dash='dot'),
                            hovertemplate='<b>Trend</b><extra></extra>'
                        ))
                    except:
                        pass  # Skip trend line if calculation fails
                
                fig.update_layout(
                    title=f"{ticker} - Predicted Monthly Returns Over Time",
                    xaxis_title="Date",
                    yaxis_title="Predicted Return (e.g., 0.05 = +5%)",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Latest Prediction")
                latest = ticker_data.iloc[-1]
                
                # Prediction with color coding and explanation
                pred_value = latest['prediction']
                pred_color = "🟢" if pred_value > 0 else "🔴"
                pred_pct = pred_value * 100
                
                st.metric(
                    "Expected Return",
                    f"{pred_value:.4f}",
                    delta=f"{pred_color} {pred_pct:+.2f}%",
                    help="Predicted return for next month"
                )
                
                st.markdown(f"""
                <div class="{'success-box' if pred_value > 0 else 'warning-box'}">
                <b>Interpretation:</b><br>
                Model expects this stock to {'gain' if pred_value > 0 else 'lose'} 
                <b>{abs(pred_pct):.2f}%</b> next month
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Sector", latest['sector'])
                st.metric("Date", latest['date'].strftime('%b %Y'))
                
                # Statistics
                st.markdown("#### 📊 Statistics")
                st.markdown(f"""
                <div class="info-box">
                <b>Mean:</b> {ticker_data['prediction'].mean():.4f}<br>
                <b>Std Dev:</b> {ticker_data['prediction'].std():.4f}<br>
                <b>Min:</b> {ticker_data['prediction'].min():.4f}<br>
                <b>Max:</b> {ticker_data['prediction'].max():.4f}<br>
                <b>Months:</b> {len(ticker_data)}
                </div>
                """, unsafe_allow_html=True)

def show_features():
    """Feature importance tab"""
    st.markdown("## 🔍 Feature Importance")
    
    st.markdown("""
    <div class="info-box">
    📊 <b>Feature Importance</b> shows which factors have the strongest predictive power (measured by Information Coefficient)
    </div>
    """, unsafe_allow_html=True)
    
    features = fetch_features()
    
    if features is None:
        st.warning("⚠️ Feature importance data not available")
        return
    
    # Top features visualization
    st.markdown("### 🏆 Top 15 Features by Mean IC")
    
    top_features = features.nlargest(15, 'mean_ic')
    
    fig = px.bar(
        top_features.sort_values('mean_ic'),
        x='mean_ic',
        y='feature',
        orientation='h',
        title="Feature Importance (Information Coefficient)",
        color='mean_ic',
        color_continuous_scale='Viridis',
        text='mean_ic',
        labels={'mean_ic': 'Mean IC', 'feature': 'Feature'}
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_font_size=16,
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 All Features")
        features_display = features.sort_values('mean_ic', ascending=False).copy()
        features_display['mean_ic'] = features_display['mean_ic'].apply(lambda x: f"{x:.4f}")
        features_display['ic_ir'] = features_display['ic_ir'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            features_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "feature": st.column_config.TextColumn("Feature", width="medium"),
                "mean_ic": st.column_config.TextColumn("Mean IC", width="small"),
                "ic_ir": st.column_config.TextColumn("IC-IR", width="small")
            }
        )
    
    with col2:
        st.markdown("### 📊 Feature Statistics")
        
        total_features = len(features)
        strong_features = (features['mean_ic'] > 0.02).sum()
        positive_features = (features['mean_ic'] > 0).sum()
        
        st.markdown(f"""
        <div class="success-box">
        <b>Total Features:</b> {total_features}<br>
        <b>Strong (IC > 0.02):</b> {strong_features}<br>
        <b>Positive IC:</b> {positive_features}<br>
        <b>Best Feature:</b> {features.nlargest(1, 'mean_ic')['feature'].values[0]}<br>
        <b>Best IC:</b> {features['mean_ic'].max():.4f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 💡 Interpretation")
        st.markdown("""
        <div class="info-box">
        <b>Mean IC:</b> Higher is better (>0.02 is strong)<br>
        <b>IC-IR:</b> Consistency of IC over time<br>
        <b>Positive IC:</b> Feature predicts returns correctly
        </div>
        """, unsafe_allow_html=True)

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


def show_performance_tab(performance):
    """Performance tab - Enhanced version"""
    st.markdown("## 📈 Model Performance")
    
    if performance is None:
        st.warning("⚠️ Performance metrics not available")
        return
    
    st.markdown("""
    <div class="info-box">
    📊 <b>Out-of-Time Validation:</b> Model tested on completely unseen future data to ensure real-world viability
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ic_value = performance['mean_ic']
        ic_delta = "Strong" if ic_value > 0.03 else "Good" if ic_value > 0.01 else "Moderate"
        st.metric(
            "Mean IC",
            f"{ic_value:.4f}",
            delta=ic_delta,
            help="Information Coefficient - correlation between predictions and returns (>0.02 is strong)"
        )
    
    with col2:
        ic_ir = performance['ic_ir']
        ir_delta = "Consistent" if ic_ir > 0.5 else "Variable"
        st.metric(
            "IC-IR",
            f"{ic_ir:.4f}",
            delta=ir_delta,
            help="IC Information Ratio - consistency of IC over time (>0.5 is good)"
        )
    
    with col3:
        sharpe = performance['sharpe']
        sharpe_delta = "Excellent" if sharpe > 1.5 else "Good" if sharpe > 1.0 else "Fair"
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta=sharpe_delta,
            help="Risk-adjusted returns (>1.0 is good, >1.5 is excellent)"
        )
    
    with col4:
        win_rate = performance['win_rate'] * 100
        wr_delta = "High" if win_rate > 60 else "Moderate"
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=wr_delta,
            help="Percentage of months with positive IC"
        )
    
    st.markdown("---")
    
    # Performance interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Performance Summary")
        
        # Determine overall quality
        quality_score = 0
        if ic_value > 0.02: quality_score += 1
        if ic_ir > 0.3: quality_score += 1
        if sharpe > 1.0: quality_score += 1
        if win_rate > 50: quality_score += 1
        
        if quality_score >= 3:
            quality = "🟢 Strong Performance"
            quality_class = "success-box"
        elif quality_score >= 2:
            quality = "🟡 Good Performance"
            quality_class = "info-box"
        else:
            quality = "🟠 Moderate Performance"
            quality_class = "warning-box"
        
        st.markdown(f"""
        <div class="{quality_class}">
        <h4>{quality}</h4>
        <b>Quality Score:</b> {quality_score}/4<br><br>
        <b>Strengths:</b><br>
        {'✓ Strong predictive power (IC > 0.02)<br>' if ic_value > 0.02 else ''}
        {'✓ Consistent performance (IC-IR > 0.3)<br>' if ic_ir > 0.3 else ''}
        {'✓ Excellent risk-adjusted returns (Sharpe > 1.0)<br>' if sharpe > 1.0 else ''}
        {'✓ High win rate (>50%)<br>' if win_rate > 50 else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Metrics Explained")
        st.markdown("""
        <div class="info-box">
        <b>Mean IC (Information Coefficient)</b><br>
        Measures correlation between predictions and actual returns<br>
        • >0.03: Strong<br>
        • 0.01-0.03: Good<br>
        • <0.01: Weak<br><br>
        
        <b>IC-IR (IC Information Ratio)</b><br>
        Measures consistency of IC over time<br>
        • >0.5: Consistent<br>
        • 0.2-0.5: Moderate<br>
        • <0.2: Variable<br><br>
        
        <b>Sharpe Ratio</b><br>
        Risk-adjusted returns<br>
        • >1.5: Excellent<br>
        • 1.0-1.5: Good<br>
        • <1.0: Fair<br><br>
        
        <b>Win Rate</b><br>
        % of months with positive IC<br>
        • >60%: High<br>
        • 50-60%: Moderate<br>
        • <50%: Low
        </div>
        """, unsafe_allow_html=True)
